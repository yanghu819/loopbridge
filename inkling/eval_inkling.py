"""Multi-GPU sharded batched evaluator for Inkling-2-Mini (native, training-free, LoopBridge, Distilled)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import string
from time import perf_counter

import torch
from transformers import AutoTokenizer

from inkling.model import concat_caches, load_inkling_runner, select_cache_rows
from inkling.scoring import score_row


@torch.inference_mode()
def greedy_batch(
    runner,
    tok,
    batch: list[dict],
    device: torch.device,
    temperature: float = 0.0,
    gen_rng: torch.Generator | None = None,
) -> list[tuple[str, int, str]]:
    """Left-padded batched decoding (greedy when temperature<=0, or multinomial sampling at temperature>0)."""
    n = len(batch)
    width = max(len(r["prompt_ids"]) for r in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((n, width), pad, dtype=torch.long)
    mask = torch.zeros((n, width), dtype=torch.long)
    for i, r in enumerate(batch):
        p = r["prompt_ids"]
        ids[i, width - len(p) :] = torch.tensor(p, dtype=torch.long)
        mask[i, width - len(p) :] = 1
    ids, mask = ids.to(device), mask.to(device)
    single = n == 1
    # Micro-batched prefill keeps eager fp32 attention <25GB while enabling huge decode batches (B=16..96)
    pre_bs = (
        1 if width > 65536 else
        2 if width > 32768 else
        4 if width > 16384 else
        8 if width > 8192 else
        16
    )
    if n <= pre_bs:
        logits, cache = runner.prefill(ids, None if single else mask)
    else:
        sub_logits = []
        sub_caches = []
        for s in range(0, n, pre_bs):
            e = min(s + pre_bs, n)
            lg, c = runner.prefill(ids[s:e], mask[s:e])
            sub_logits.append(lg)
            sub_caches.append(c)
        logits = torch.cat(sub_logits, dim=0)
        cache = concat_caches(sub_caches)
        del sub_logits, sub_caches

    eos_id = tok.eos_token_id
    gen = [[] for _ in batch]
    out: list[tuple[str, int, str] | None] = [None] * n
    active = list(range(n))
    max_steps = max(r["max_new_tokens"] for r in batch)

    for _ in range(max_steps):
        if temperature > 0.0:
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1, generator=gen_rng).squeeze(-1)
        else:
            tokens = logits.argmax(dim=-1)
        for j, i in enumerate(active):
            t = int(tokens[j])
            if t == eos_id:
                out[i] = (tok.decode(gen[i], skip_special_tokens=False), len(gen[i]), "eos")
                continue
            gen[i].append(t)
            if len(gen[i]) >= batch[i]["max_new_tokens"]:
                out[i] = (tok.decode(gen[i], skip_special_tokens=False), len(gen[i]), "max_new_tokens")
        keep = [j for j, i in enumerate(active) if out[i] is None]
        if not keep:
            break
        if len(keep) < len(active):
            idx = torch.tensor(keep, device=device, dtype=torch.long)
            select_cache_rows(cache, idx, len(active))
            tokens = tokens[idx]
            mask = mask[idx]
            active = [active[j] for j in keep]
        mask = torch.cat([mask, mask.new_ones((len(active), 1))], dim=1)
        logits, cache = runner.decode(tokens.view(-1, 1), cache, None if single else mask)
    return out  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/dev/shm/inkling/model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benches", nargs="+", required=True)
    parser.add_argument("--mode", choices=("native", "training_free", "loopbridge", "distilled"), required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-tokens", type=int, default=4194304)
    parser.add_argument("--max-batch", type=int, default=96)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stratified", action="store_true",
                        help="When --limit > 0, pick evenly spaced items sorted by (task, prompt_len) for unbiased fast alignment")
    parser.add_argument("--force-text-header", action="store_true",
                        help="Append <|message_model|><|content_text|> so effort=none answers directly without thinking")
    parser.add_argument("--max-new-tokens-map", default="",
                        help="Optional comma-separated bench=N overrides, e.g. gsm8k=2048,math500=4096")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Decoding temperature (0.0=greedy; 1.0=Inkling-2-Mini official generation_config.json)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_tok_overrides: dict[str, int] = {}
    for item in args.max_new_tokens_map.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            max_tok_overrides[k.strip()] = int(v.strip())

    device = torch.device("cuda:0")
    gen_rng = torch.Generator(device=device).manual_seed(args.seed + args.shard_index) if args.temperature > 0.0 else None
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    text_hdr = tok.encode("<|message_model|><|content_text|>", add_special_tokens=False) if args.force_text_header else []
    runner = load_inkling_runner(
        args.model_path,
        mode=args.mode,
        m=args.m,
        n=args.n,
        checkpoint=args.checkpoint,
        device=device,
        chunk_size=args.chunk_size,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    for bench in args.benches:
        t0 = perf_counter()
        all_rows = [json.loads(line) for line in (args.data / f"{bench}.jsonl").open()]
        if text_hdr:
            for r in all_rows:
                r["prompt_ids"] = r["prompt_ids"] + text_hdr
        if bench in max_tok_overrides:
            for r in all_rows:
                r["max_new_tokens"] = max_tok_overrides[bench]
        if args.limit and args.limit < len(all_rows):
            if args.stratified:
                ordered = sorted(all_rows, key=lambda r: (str(r.get("task", "")), len(r["prompt_ids"])))
                step = len(ordered) / args.limit
                all_rows = [ordered[int(k * step)] for k in range(args.limit)]
            else:
                all_rows = all_rows[: args.limit]
        shard_rows = all_rows[args.shard_index :: args.num_shards]
        out_path = args.output / f"{bench}.shard{args.shard_index}.jsonl" if args.num_shards > 1 else args.output / f"{bench}.jsonl"

        done = {}
        if out_path.exists():
            for line in out_path.open():
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()))

        # Also recognize items already completed in any exact-bench shard/pilot file (zero prefix collision)
        global_done = set(done.keys())
        exact_files = [
            p for p in [args.output / f"{bench}.jsonl", args.output / f"{bench}.pilot.jsonl"] if p.exists()
        ] + sorted(args.output.glob(f"{bench}.shard*.jsonl"))
        for existing in exact_files:
            if existing == out_path:
                continue
            for line in existing.open():
                if line.strip():
                    try:
                        global_done.add(json.loads(line)["id"])
                    except Exception:
                        pass

        todo = [r for r in shard_rows if r["id"] not in global_done]
        todo.sort(key=lambda r: len(r["prompt_ids"]) + r["max_new_tokens"])

        with out_path.open("a") as fh:
            i = 0
            while i < len(todo):
                # Micro-batched prefill decouples prefill attention VRAM from decode KV cache (~6.7GB@128K, 3.5GB@64K)
                bs = min(args.max_batch, len(todo) - i)
                while bs > 1:
                    max_plen = max(len(r["prompt_ids"]) for r in todo[i : i + bs])
                    max_span = max(len(r["prompt_ids"]) + r["max_new_tokens"] for r in todo[i : i + bs])
                    cap = (
                        16 if max_plen > 65536 else
                        32 if max_plen > 32768 else
                        48 if max_plen > 16384 else
                        64 if max_plen > 8192 else
                        96
                    )
                    if bs <= cap and bs * max_span <= args.batch_tokens:
                        break
                    bs = min(bs - 1, cap)
                batch = todo[i : i + bs]
                torch.cuda.synchronize(device)
                tb0 = perf_counter()
                outs = greedy_batch(runner, tok, batch, device, temperature=args.temperature, gen_rng=gen_rng)
                torch.cuda.synchronize(device)
                batch_wall = (perf_counter() - tb0) / max(len(batch), 1)
                for row, (raw_txt, n_gen, reason) in zip(batch, outs):
                    rec = score_row(bench, row, raw_txt, n_gen, reason)
                    rec["wall_s"] = round(batch_wall, 4)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[rec["id"]] = rec
                fh.flush()
                i += len(batch)
                print(
                    f"[shard {args.shard_index}/{args.num_shards}] {bench} progress: "
                    f"{i}/{len(todo)} new (bs={len(batch)}, {perf_counter()-tb0:.1f}s)",
                    flush=True,
                )

        vals = [float(r["correct"]) if "correct" in r else float(r["score"]) for r in done.values()]
        mean_sc = 100.0 * sum(vals) / max(len(vals), 1)
        print(
            f"[shard {args.shard_index}/{args.num_shards}] {bench} ({args.mode} {args.m}/{args.n}): "
            f"{mean_sc:.2f}% ({len(vals)} rows) in {perf_counter()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
