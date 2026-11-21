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

from inkling.model import load_inkling_runner, select_cache_rows
from inkling.scoring import bbh_answer, helmet_score, minerva_norm, ruler_score


def extract_visible_text(raw: str) -> str:
    """Extract Inkling-2-Mini's final assistant text after <|content_text|> (fallback to cleaned text)."""
    if "<|content_text|>" in raw:
        txt = raw.rsplit("<|content_text|>", 1)[-1]
    else:
        txt = raw
    for tag in ("<|content_model_end_sampling|>", "<|end_message|>", "<|message_model|>", "<|content_thinking|>"):
        txt = txt.replace(tag, "")
    return txt.strip()


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


def parse_gsm8k(text: str) -> str:
    b = extract_boxed(text)
    if b is not None:
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", b)
        if m:
            return m[-1].replace(",", "").rstrip(".")
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else "[invalid]"


def score_gsm8k(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def score_math500(text: str, gold_norm: str) -> tuple[str, bool]:
    b = extract_boxed(text)
    if b is None:
        m = re.search(r"(?:final answer is|answer is)\s*[:$]?\s*([^\n.]+)", text, re.I)
        b = m.group(1).strip() if m else ""
    if not b:
        return "[invalid]", False
    pred_norm = minerva_norm(b)
    if pred_norm == gold_norm:
        return pred_norm, True
    try:
        from math_verify import parse, verify
        if verify(parse(f"${gold_norm}$"), parse(f"${b}$")):
            return pred_norm, True
    except Exception:
        pass
    return pred_norm, False


def parse_mcq(text: str, valid_letters: str = "ABCD") -> str:
    # Check explicit "Answer: (X)" / "The correct answer is (X)" / "\boxed{X}" patterns first
    pats = [
        rf"(?:the\s+correct\s+answer\s+is|final\s+answer\s+is|answer\s*(?:is|:))\s*\(?\s*([{valid_letters}])\s*\)?",
        rf"\\boxed\{{\s*\(?\s*([{valid_letters}])\s*\)?\s*\}}",
        rf"\(\s*([{valid_letters}])\s*\)",
        rf"\b([{valid_letters}])\b",
    ]
    for pat in pats:
        ms = re.findall(pat, text, re.I)
        if ms:
            return ms[-1].upper()
    return "[invalid]"


def score_bbh(text: str, gold: str) -> tuple[str, bool]:
    pred = bbh_answer(text)
    if pred == "[invalidanswer]":
        b = extract_boxed(text)
        if b:
            pred = b.strip()
        else:
            m = re.search(r"answer\s*(?:is|:)\s*([^\n]+)", text, re.I)
            if m:
                pred = m.group(1).strip().rstrip(".")
    g = gold.strip()
    ok = pred.strip().lower() == g.lower() or pred.strip("() ").lower() == g.strip("() ").lower()
    return pred, ok


def score_row(bench: str, row: dict, raw_text: str, gen_tokens: int, reason: str) -> dict:
    vis = extract_visible_text(raw_text)
    base = {
        "id": row["id"],
        "task": row["task"],
        "generated_tokens": gen_tokens,
        "stop_reason": reason,
        "text": vis[:800],
    }
    if bench == "gsm8k":
        pred = parse_gsm8k(vis)
        return {**base, "pred": pred, "gold": row["answer"], "correct": score_gsm8k(pred, row["answer"])}
    if bench.startswith("math500"):
        pred, ok = score_math500(vis, row["answer"])
        return {**base, "pred": pred, "gold": row["answer"], "correct": ok}
    if bench == "mmlu_pro":
        pred = parse_mcq(vis, "ABCDEFGHIJ")
        return {**base, "pred": pred, "gold": row["answer"], "correct": pred == row["answer"]}
    if bench == "arc_c":
        pred = parse_mcq(vis, "ABCDE")
        return {**base, "pred": pred, "gold": row["answer"], "correct": pred == row["answer"]}
    if bench.startswith("longbench_v2"):
        pred = parse_mcq(vis, "ABCD")
        return {
            **base,
            "length_bucket": row.get("length_bucket"),
            "difficulty": row.get("difficulty"),
            "pred": pred,
            "gold": row["answer"],
            "correct": pred == row["answer"],
        }
    if bench == "bbh":
        pred, ok = score_bbh(vis, row["answer"])
        return {**base, "pred": pred, "gold": row["answer"], "correct": ok}
    if bench.startswith("ruler"):
        sc = float(ruler_score(row["task"], vis, row["answers"]))
        return {**base, "score": sc}
    if bench == "helmet":
        sc = float(helmet_score(row, vis))
        return {**base, "length": row["length"], "score": sc}
    raise ValueError(f"Unknown bench: {bench}")


@torch.inference_mode()
def greedy_batch(runner, tok, batch: list[dict], device: torch.device) -> list[tuple[str, int, str]]:
    """Left-padded batched greedy decoding with dynamic compaction when rows finish."""
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
    logits, cache = runner.prefill(ids, None if single else mask)

    eos_id = tok.eos_token_id
    gen = [[] for _ in batch]
    out: list[tuple[str, int, str] | None] = [None] * n
    active = list(range(n))
    max_steps = max(r["max_new_tokens"] for r in batch)

    for _ in range(max_steps):
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
    parser.add_argument("--batch-tokens", type=int, default=32768)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
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
        if args.limit:
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

        todo = [r for r in shard_rows if r["id"] not in done]
        todo.sort(key=lambda r: len(r["prompt_ids"]) + r["max_new_tokens"])

        with out_path.open("a") as fh:
            i = 0
            while i < len(todo):
                span = len(todo[i]["prompt_ids"]) + todo[i]["max_new_tokens"]
                bs = max(1, min(args.max_batch, args.batch_tokens // max(span, 1)))
                # Cap batch size on very long prompts (>12K) to keep attention memory crisp
                if len(todo[i]["prompt_ids"]) > 12000:
                    bs = min(bs, 4)
                batch = todo[i : i + bs]
                outs = greedy_batch(runner, tok, batch, device)
                for row, (raw_txt, n_gen, reason) in zip(batch, outs):
                    rec = score_row(bench, row, raw_txt, n_gen, reason)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[rec["id"]] = rec
                fh.flush()
                i += len(batch)

        vals = [float(r["correct"]) if "correct" in r else float(r["score"]) for r in done.values()]
        mean_sc = 100.0 * sum(vals) / max(len(vals), 1)
        print(
            f"[shard {args.shard_index}/{args.num_shards}] {bench} ({args.mode} {args.m}/{args.n}): "
            f"{mean_sc:.2f}% ({len(vals)} rows) in {perf_counter()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
