"""Build 100% official-exact Thinking Machines evaluation splits (matching model card length regimes).

Exact official splits:
  1. `mrcr_8n_256k_512k.jsonl`  : Exact 100 items in OpenAI MRCR 8-needle `(262144, 524288]` o200k_base bin
                                  (Official Inkling-2-Mini: 46.4%; TML-varloop 1p..4p: 27.2/65.2/89.2/93.4%).
  2. `graphwalks_bfs_256k.jsonl`: Exact 100 BFS items at ~256K from `graphwalks_256k_to_1mil.parquet`
                                  (Official Inkling-2-Mini: 69.7%; TML-varloop 1p..3p: 23.5/56.0/64.3%).
  3. `longbench_v2_1m.jsonl`    : All 503 LongBench v2 items UNTRUNCATED up to 990,000 tokens
                                  (middle-truncating only the ~31 extreme items >1M to fit 1,048,576 context;
                                   Official Inkling-2-Mini: 66.9%).
  4. Symlinks `gpqa_diamond_high.jsonl` (198) and `livecodebench_v6_high.jsonl` (175) from `eval_sec5_high`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import tiktoken
from transformers import AutoTokenizer

_TOK = None
_ENC = None
_EFFORT = "high"
_LB2_MAX = 990000


def _init_worker(model_path: str, effort: str, lb2_max: int) -> None:
    global _TOK, _ENC, _EFFORT, _LB2_MAX
    _TOK = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    _ENC = tiktoken.get_encoding("o200k_base")
    _EFFORT = effort
    _LB2_MAX = lb2_max


def _chat_ids(messages: list[dict]) -> list[int]:
    s = _TOK.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=_EFFORT,
    )
    return _TOK.encode(s, add_special_tokens=False)


def _proc_mrcr_item(arg: tuple[str, int, dict]) -> dict | None:
    stem, idx, r = arg
    msgs = json.loads(r["prompt"])
    nt = sum(len(_ENC.encode(m["content"], disallowed_special=())) for m in msgs) + len(
        _ENC.encode(r["answer"], disallowed_special=())
    )
    if not (262144 < nt <= 524288):
        return None
    norm_msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]
    ids = _chat_ids(norm_msgs)
    return {
        "id": f"mrcr_8n_256k_512k/{stem}/{idx}",
        "task": "8needle_256k_512k",
        "o200k_tokens": nt,
        "prompt_ids": ids,
        "answer": r["answer"],
        "random_string_to_prepend": r["random_string_to_prepend"],
        "max_new_tokens": 16384,
        "until": [],
    }


def _proc_gw_item(arg: tuple[int, dict]) -> dict | None:
    idx, r = arg
    if r["problem_type"] != "bfs":
        return None
    nt = len(_ENC.encode(r["prompt"], disallowed_special=()))
    if not (131072 < nt <= 262144):
        return None
    ids = _chat_ids([{"role": "user", "content": r["prompt"]}])
    return {
        "id": f"graphwalks_bfs_256k/{idx}",
        "task": "bfs_256k",
        "o200k_tokens": nt,
        "prompt_ids": ids,
        "answer_nodes": list(r["answer_nodes"]),
        "max_new_tokens": 32768,
        "until": [],
    }


def _proc_lb2_item(r: dict) -> dict:
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
    ids = _chat_ids([{"role": "user", "content": body}])
    orig_len = len(ids)
    if orig_len > _LB2_MAX:
        half = _LB2_MAX // 2
        ids = ids[:half] + ids[-(_LB2_MAX - half) :]
    return {
        "id": f"lb2/{r['_id']}",
        "task": f"{r['domain']}/{r['length']}/{r['difficulty']}",
        "length_bucket": r["length"],
        "difficulty": r["difficulty"],
        "orig_prompt_tokens": orig_len,
        "prompt_ids": ids,
        "answer": r["answer"].strip().upper(),
        "max_new_tokens": 32768,
        "until": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--raw-dir", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    parser.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    parser.add_argument("--sec5-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_sec5_high"))
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/inkling/data/eval_official_exact"))
    parser.add_argument("--effort", default="high")
    parser.add_argument("--lb2-max-tokens", type=int, default=990000)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(args.model_path), args.effort, args.lb2_max_tokens),
    ) as ex:
        # 1. MRCR v2 8-needle 256K-512K (exact 100 official items)
        mrcr_tasks = []
        for fp in sorted(args.raw_dir.glob("mrcr_8needle_*.parquet")):
            for idx, r in enumerate(pq.read_table(fp).to_pylist()):
                mrcr_tasks.append((fp.stem, idx, r))
        mrcr_rows = [x for x in ex.map(_proc_mrcr_item, mrcr_tasks) if x is not None]
        (args.output / "mrcr_8n_256k_512k.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in mrcr_rows)
        )
        m_lens = [len(r["prompt_ids"]) for r in mrcr_rows]
        summary["mrcr_8n_256k_512k"] = {
            "n": len(mrcr_rows),
            "min_tok": min(m_lens),
            "med_tok": sorted(m_lens)[len(m_lens) // 2],
            "max_tok": max(m_lens),
        }
        print("Built mrcr_8n_256k_512k:", summary["mrcr_8n_256k_512k"], flush=True)

        # 2. GraphWalks BFS 256K (exact 100 official 256K items)
        gw_fp = args.raw_dir / "graphwalks_256k_to_1mil.parquet"
        gw_tasks = list(enumerate(pq.read_table(gw_fp).to_pylist()))
        gw_rows = [x for x in ex.map(_proc_gw_item, gw_tasks) if x is not None]
        (args.output / "graphwalks_bfs_256k.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in gw_rows)
        )
        g_lens = [len(r["prompt_ids"]) for r in gw_rows]
        summary["graphwalks_bfs_256k"] = {
            "n": len(gw_rows),
            "min_tok": min(g_lens),
            "med_tok": sorted(g_lens)[len(g_lens) // 2],
            "max_tok": max(g_lens),
        }
        print("Built graphwalks_bfs_256k:", summary["graphwalks_bfs_256k"], flush=True)

        # 3. LongBench v2 1M (all 503 items untruncated up to 990K tokens)
        lb_raw = [json.loads(l) for l in (args.eval_raw / "longbench_v2.jsonl").open() if l.strip()]
        lb_rows = list(ex.map(_proc_lb2_item, lb_raw))
        (args.output / "longbench_v2_1m.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in lb_rows)
        )
        l_lens = [len(r["prompt_ids"]) for r in lb_rows]
        n_trunc = sum(1 for r in lb_rows if r["orig_prompt_tokens"] > args.lb2_max_tokens)
        summary["longbench_v2_1m"] = {
            "n": len(lb_rows),
            "min_tok": min(l_lens),
            "med_tok": sorted(l_lens)[len(l_lens) // 2],
            "max_tok": max(l_lens),
            "truncated_gt_990k": n_trunc,
        }
        print("Built longbench_v2_1m:", summary["longbench_v2_1m"], flush=True)

    # 4. Symlink GPQA Diamond & LiveCodeBench v6
    for fn in ("gpqa_diamond_high.jsonl", "livecodebench_v6_high.jsonl"):
        src = args.sec5_dir / fn
        dst = args.output / fn
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)

    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("ALL_OFFICIAL_EXACT_DONE:", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
