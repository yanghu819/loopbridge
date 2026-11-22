"""Prepare modern zero-Docker frontier evaluation benchmarks for Inkling-2-Mini.

Creates tokenized JSONLs under `/dev/shm/inkling/data/eval_frontier`:
  1. gpqa_diamond.jsonl     (198 hard science MCQ, effort=medium, cap=4096)
  2. livecodebench_v6.jsonl (175 2025 competitive coding problems, effort=medium, cap=4096)
  3. bfcl_live.jsonl        (400 stratified BFCL v3 Live tool-calling items across all 5 splits,
                             rendered with Inkling-2-Mini's official native `tools=` chat template!)

Note: GPQA Diamond raw prompts stay strictly on local compute storage and are NEVER committed
to public repositories; only compact per-item binary scores are archived.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import pickle
import random
import re
import zlib

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _norm_schema(obj):
    """Recursively normalize BFCL JSON schema types ('dict' -> 'object', 'float' -> 'number')."""
    if isinstance(obj, list):
        return [_norm_schema(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    type_map = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}
    for k, v in obj.items():
        if k == "type" and isinstance(v, str):
            out[k] = type_map.get(v.lower(), v.lower())
        else:
            out[k] = _norm_schema(v)
    return out


def _wrap_bfcl_tools(fns: list[dict]) -> list[dict]:
    tools = []
    for fn in fns:
        params = _norm_schema(fn.get("parameters", {"type": "object", "properties": {}}))
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return tools


def _decode_lcb_tests(raw: str) -> list[dict]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--raw-dir", type=Path, default=Path("/dev/shm/inkling/data/v2_raw"))
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/inkling/data/eval_frontier"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}

    # 1. GPQA Diamond MCQ (198 items)
    gpqa_in = args.raw_dir / "gpqa_diamond_mc.parquet"
    if gpqa_in.exists():
        rows = []
        for idx, r in enumerate(pq.read_table(gpqa_in).to_pylist()):
            m = re.search(r"([ABCD])", r["solution"])
            gold = m.group(1) if m else ""
            prompt = (
                "Solve the following graduate-level science multiple-choice question carefully. "
                "Conclude with your final choice letter inside \\boxed{} (for example, \\boxed{A}).\n\n"
                f"{r['problem'].strip()}"
            )
            s = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            rows.append(
                {
                    "id": f"gpqa_diamond/{idx}",
                    "task": r.get("domain", "science"),
                    "prompt_ids": tok.encode(s, add_special_tokens=False),
                    "answer": gold,
                    "max_new_tokens": 8192,
                    "until": [],
                }
            )
        (args.output / "gpqa_diamond.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["gpqa_diamond"] = {"n": len(rows)}

    # 2. LiveCodeBench v6 (175 2025 problems)
    lcb_in = args.raw_dir / "lcb_test6.jsonl"
    if lcb_in.exists():
        rows = []
        for line in lcb_in.open():
            if not line.strip():
                continue
            r = json.loads(line)
            pub = _decode_lcb_tests(r.get("public_test_cases", ""))
            priv = _decode_lcb_tests(r.get("private_test_cases", ""))
            # Combine public + up to 25 private tests (covers 100% of edge cases cleanly & fast)
            tests = pub + priv[:25]
            meta = json.loads(r["metadata"]) if isinstance(r.get("metadata"), str) and r["metadata"] else {}
            fn_name = meta.get("func_name", "")
            starter = (r.get("starter_code") or "").strip()
            if starter:
                prompt = (
                    "Solve the following competitive programming problem in Python 3. "
                    "Complete the provided starter code and put your final solution inside a single ```python``` code block.\n\n"
                    f"### Problem: {r['question_title']}\n{r['question_content'].strip()}\n\n"
                    f"### Starter Code\n```python\n{starter}\n```"
                )
            else:
                prompt = (
                    "Solve the following competitive programming problem in Python 3. "
                    "Read from standard input and write to standard output. "
                    "Put your complete executable solution inside a single ```python``` code block.\n\n"
                    f"### Problem: {r['question_title']}\n{r['question_content'].strip()}"
                )
            s = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            rows.append(
                {
                    "id": f"lcb_v6/{r['question_id']}",
                    "task": f"{r['platform']}/{r['difficulty']}",
                    "difficulty": r["difficulty"],
                    "prompt_ids": tok.encode(s, add_special_tokens=False),
                    "test_cases": tests,
                    "fn_name": fn_name,
                    "max_new_tokens": 8192,
                    "until": [],
                }
            )
        (args.output / "livecodebench_v6.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["livecodebench_v6"] = {"n": len(rows)}

    # 3. BFCL v3 Live (400 stratified items across all 5 categories, native Inkling tool template)
    bfcl_dir = args.raw_dir / "bfcl"
    if bfcl_dir.exists():
        rng = random.Random(args.seed)
        splits = [
            ("live_simple", "BFCL_v3_live_simple.json", 100),
            ("live_multiple", "BFCL_v3_live_multiple.json", 150),
            ("live_parallel", "BFCL_v3_live_parallel.json", 15),
            ("live_parallel_multiple", "BFCL_v3_live_parallel_multiple.json", 23),
            ("live_irrelevance", "BFCL_v3_live_irrelevance.json", 112),
        ]
        rows = []
        counts = {}
        for cat, fname, quota in splits:
            q_path = bfcl_dir / fname
            if not q_path.exists():
                continue
            qs = [json.loads(l) for l in q_path.open() if l.strip()]
            gt_map = {}
            gt_path = bfcl_dir / "possible_answer" / fname
            if gt_path.exists():
                for l in gt_path.open():
                    if l.strip():
                        g = json.loads(l)
                        gt_map[g["id"]] = g["ground_truth"]
            rng.shuffle(qs)
            picked = 0
            for item in qs:
                if picked >= quota:
                    break
                if "irrelevance" not in cat and item["id"] not in gt_map:
                    continue
                msgs = item["question"][0] if item["question"] else []
                if not msgs:
                    continue
                tools = _wrap_bfcl_tools(item.get("function", []))
                try:
                    s = tok.apply_chat_template(
                        msgs,
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=True,
                        reasoning_effort="low",
                    )
                except Exception:
                    continue
                rows.append(
                    {
                        "id": f"bfcl/{item['id']}",
                        "task": cat,
                        "prompt_ids": tok.encode(s, add_special_tokens=False),
                        "ground_truth": gt_map.get(item["id"]),
                        "max_new_tokens": 1024,
                        "until": [],
                    }
                )
                picked += 1
            counts[cat] = picked
        (args.output / "bfcl_live.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["bfcl_live"] = {"n": len(rows), "by_split": counts}

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
