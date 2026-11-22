"""8-GPU sharded batched On-Policy Original 3/3 generator for zero-shift distillation rollouts.

Generates native Inkling-2-Mini `<|content_thinking|>`, `<|content_text|>`, and
`<|content_invoke_tool_json|>` responses from frozen Original 3/3 over a seed prompt JSONL
(strictly disjoint from all evaluation benchmarks), writing train_inkling.py-ready rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from inkling.eval_inkling import greedy_batch
from inkling.model import load_inkling_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/dev/shm/inkling/model")
    parser.add_argument("--prompts", type=Path, required=True, help="Input JSONL with id, source, effort, prompt_ids, max_new_tokens")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for shard JSONLs")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--batch-tokens", type=int, default=98304)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    runner = load_inkling_runner(args.model_path, mode="native", m=3, n=3, device=device, chunk_size=args.chunk_size)

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = [json.loads(l) for l in args.prompts.open() if l.strip()]
    shard_rows = all_rows[args.shard_index :: args.num_shards]
    out_path = args.output / f"onpolicy.shard{args.shard_index}.jsonl"

    done_ids = set()
    if out_path.exists():
        for l in out_path.open():
            if l.strip():
                done_ids.add(json.loads(l)["id"])

    todo = [r for r in shard_rows if r["id"] not in done_ids]
    todo.sort(key=lambda r: len(r["prompt_ids"]) + r["max_new_tokens"])

    t0 = perf_counter()
    kept, total_toks = 0, 0
    eos_id = tok.eos_token_id
    with out_path.open("a") as fh:
        i = 0
        while i < len(todo):
            bs = min(args.max_batch, len(todo) - i)
            while bs > 1:
                max_plen = max(len(r["prompt_ids"]) for r in todo[i : i + bs])
                max_span = max(len(r["prompt_ids"]) + r["max_new_tokens"] for r in todo[i : i + bs])
                cap = 2 if max_plen > 8000 else (4 if max_plen > 4096 else (8 if max_plen > 2048 else (24 if max_plen > 1024 else 64)))
                if bs <= cap and bs * max_span <= args.batch_tokens:
                    break
                bs = min(bs - 1, cap)
            batch = todo[i : i + bs]
            outs = greedy_batch(runner, tok, batch, device)
            for row, (raw_txt, n_gen, reason) in zip(batch, outs):
                s_ids = tok.encode(raw_txt, add_special_tokens=False)
                if reason == "eos" and eos_id is not None:
                    s_ids = s_ids + [eos_id]
                if len(s_ids) < 8:
                    continue
                rec = {
                    "id": row["id"],
                    "source": f"onpolicy_{row.get('source', 'mix')}",
                    "effort": row.get("effort", "low"),
                    "prefix_ids": row["prompt_ids"],
                    "suffix_ids": s_ids,
                    "effective_tokens": len(s_ids),
                }
                fh.write(json.dumps(rec) + "\n")
                kept += 1
                total_toks += len(s_ids)
            fh.flush()
            i += len(batch)
            if i % 64 == 0 or i >= len(todo):
                print(
                    f"[onpolicy shard {args.shard_index}/{args.num_shards}] {i}/{len(todo)} rows "
                    f"({total_toks/1e6:.2f}M tok) in {perf_counter()-t0:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
