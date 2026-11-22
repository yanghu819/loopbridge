"""Prepare official Inkling-2-Mini Section 5 benchmarks (100% programmatic, zero LLM-judge).

Covers four official public text benchmarks from Thinking Machines Lab's Inkling-2-Mini model card
(Section 5, Reasoning / Knowledge / Long Context):
  1. LongBench v2 (503 items, up to 128K context, effort=low)       — Official: 66.9%
  2. GraphWalks BFS (350 items, <=128K split, exact node-set F1)    — Official: 69.7% (256K)
  3. MRCR v2 8-needle (<=64K prompt tokens, exact SequenceMatcher)  — Official: 46.4% (256K-512K)
  4. Apex 2025 (12 olympiad problems, exact math_verify)            — Official: 34.4%
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from inkling.scoring import minerva_norm


def _chat_ids(tok, messages: list[dict], effort: str) -> list[int]:
    s = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
    )
    return tok.encode(s, add_special_tokens=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--raw-dir", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    parser.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/inkling/data/eval_sec5"))
    parser.add_argument("--mrcr-max-tokens", type=int, default=65536)
    parser.add_argument("--lb2-max-tokens", type=int, default=131072)
    parser.add_argument("--effort", default="high", help="Inkling-2-Mini reasoning_effort (official chat_template default: high=0.9)")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1. LongBench v2 (full 128K context + official default thinking effort)
    lb2_in = args.eval_raw / "longbench_v2.jsonl"
    if lb2_in.exists():
        rows = []
        for line in lb2_in.open():
            r = json.loads(line)
            body = (
                "Please read the following text and answer the question below.\n\n"
                f"<text>\n{r['context']}\n</text>\n\n"
                f"What is the correct answer to this question: {r['question']}\n"
                "Choices:\n"
                f"(A) {r['choice_A']}\n"
                f"(B) {r['choice_B']}\n"
                f"(C) {r['choice_C']}\n"
                f"(D) {r['choice_D']}\n\n"
                'Format your response as follows: "The correct answer is (insert answer here)".'
            )
            ids = _chat_ids(tok, [{"role": "user", "content": body}], effort=args.effort)
            if len(ids) > args.lb2_max_tokens:
                half = args.lb2_max_tokens // 2
                ids = ids[:half] + ids[-(args.lb2_max_tokens - half) :]
            rows.append(
                {
                    "id": f"lb2/{r['_id']}",
                    "task": f"{r['domain']}/{r['length']}/{r['difficulty']}",
                    "length_bucket": r["length"],
                    "difficulty": r["difficulty"],
                    "prompt_ids": ids,
                    "answer": r["answer"].strip().upper(),
                    "max_new_tokens": 8192,
                    "until": [],
                }
            )
        (args.output / "longbench_v2_128k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["longbench_v2_128k"] = {
            "n": len(rows),
            "effort": args.effort,
            "max_prompt_tok": max(len(r["prompt_ids"]) for r in rows),
        }

    # 2. GraphWalks BFS (all 350 BFS tasks from graphwalks_128k_and_shorter.parquet)
    gw_in = args.raw_dir / "graphwalks_128k.parquet"
    if gw_in.exists():
        tbl = pq.read_table(gw_in).to_pylist()
        rows = []
        for idx, r in enumerate(tbl):
            if r["problem_type"] != "bfs":
                continue
            ids = _chat_ids(tok, [{"role": "user", "content": r["prompt"]}], effort=args.effort)
            rows.append(
                {
                    "id": f"graphwalks_bfs/{idx}",
                    "task": "bfs",
                    "prompt_ids": ids,
                    "answer_nodes": list(r["answer_nodes"]),
                    "max_new_tokens": 8192,
                    "until": [],
                }
            )
        (args.output / "graphwalks_bfs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["graphwalks_bfs"] = {
            "n": len(rows),
            "effort": args.effort,
            "median_prompt_tok": sorted(len(r["prompt_ids"]) for r in rows)[len(rows) // 2],
            "max_prompt_tok": max(len(r["prompt_ids"]) for r in rows),
        }

    # 3. MRCR v2 8-needle (all items <= mrcr_max_tokens across available 8needle parquets)
    mrcr_files = sorted(args.raw_dir.glob("mrcr_8needle_*.parquet"))
    if mrcr_files:
        rows = []
        for fp in mrcr_files:
            tbl = pq.read_table(fp).to_pylist()
            for idx, r in enumerate(tbl):
                # Quick char filter before tokenizing huge 2M-char items
                if r["n_chars"] > args.mrcr_max_tokens * 5:
                    continue
                msgs = json.loads(r["prompt"])
                # Normalize assistant role names if needed
                norm_msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
                ids = _chat_ids(tok, norm_msgs, effort=args.effort)
                if len(ids) > args.mrcr_max_tokens:
                    continue
                rows.append(
                    {
                        "id": f"mrcr_8n/{fp.stem}/{idx}",
                        "task": f"8needle_len{((len(ids)-1)//16384 + 1)*16}k",
                        "prompt_ids": ids,
                        "answer": r["answer"],
                        "random_string_to_prepend": r["random_string_to_prepend"],
                        "max_new_tokens": 4096,
                        "until": [],
                    }
                )
        (args.output / "mrcr_8n.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["mrcr_8n"] = {
            "n": len(rows),
            "effort": args.effort,
            "median_prompt_tok": sorted(len(r["prompt_ids"]) for r in rows)[len(rows) // 2] if rows else 0,
            "max_prompt_tok": max((len(r["prompt_ids"]) for r in rows), default=0),
        }

    # 4. Apex 2025 (12 olympiad problems)
    apex_in = args.raw_dir / "apex_2025.parquet"
    if apex_in.exists():
        tbl = pq.read_table(apex_in).to_pylist()
        rows = []
        for r in tbl:
            prompt = (
                "Solve the following olympiad mathematics problem step by step. "
                "Put your final answer inside \\boxed{}.\n\n"
                f"{r['problem'].strip()}"
            )
            ids = _chat_ids(tok, [{"role": "user", "content": prompt}], effort=args.effort)
            rows.append(
                {
                    "id": f"apex_2025/{r['problem_idx']}",
                    "task": r.get("source", "apex_2025"),
                    "prompt_ids": ids,
                    "answer": minerva_norm(str(r["answer"])),
                    "max_new_tokens": 16384,
                    "until": [],
                }
            )
        (args.output / "apex_2025.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["apex_2025"] = {"n": len(rows)}

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
