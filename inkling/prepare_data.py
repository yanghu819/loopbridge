"""Prepare native Inkling-2-Mini training stream and evaluation benchmarks.

1. Training data (`train` mode):
   Decodes the shared LoopBridge 48M-token distillation mix (`/dev/shm/inkling/data/mix-data-48m/gsm_train.jsonl`,
   containing Tulu 3 SFT, OpenThoughts-114k, and LongAlpaca-12k) from Ouro token IDs and re-tokenizes it
   into Inkling-2-Mini's native chat & thinking format (keeping LongAlpaca prefixes up to 3072 tokens so
   global full-attention layers >1024 receive direct long-range KV-map supervision).

2. Evaluation benchmarks (`eval` mode):
   Prepares Inkling-tokenized JSONLs for:
     - GSM8K (1,319 test)
     - MATH-500 (500 test, both effort=none and effort=low)
     - MMLU-Pro-420 (420 stratified: 30 per subject x 14 subjects, matching LoopBridge paper)
     - ARC-Challenge (1,172 test)
     - BBH-540 (540 stratified: 20 per task x 27 tasks)
     - RULER-4K / 8K / 16K (520 each, decoded from bench/ruler*.jsonl)
     - HELMET (1,108 from bench-mini/helmet.jsonl)
     - LongBench-v2 (503 test, official multiple-choice long-context benchmark from Inkling-2-Mini model card)
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from multiprocessing import Pool
from pathlib import Path
import random
import re

from transformers import AutoTokenizer


_OTOK = None
_ITOK = None
_MAX_PREFIX = 3072
_MAX_TARGET = 1024


def _init_workers(ouro_tok_path: str, inkling_tok_path: str, max_prefix: int, max_target: int):
    global _OTOK, _ITOK, _MAX_PREFIX, _MAX_TARGET
    _OTOK = AutoTokenizer.from_pretrained(ouro_tok_path, local_files_only=True, trust_remote_code=True)
    _ITOK = AutoTokenizer.from_pretrained(inkling_tok_path, local_files_only=True)
    _MAX_PREFIX = max_prefix
    _MAX_TARGET = max_target


def _strip_user_assistant(prompt_txt: str) -> str:
    s = prompt_txt.strip()
    if s.startswith("User:"):
        s = s[5:].lstrip()
    if s.endswith("Assistant:"):
        s = s[: -len("Assistant:")].rstrip()
    return s


def _convert_train_chunk(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        r = json.loads(line)
        src = r["source"]
        p_txt = _OTOK.decode(r["prompt_ids"], skip_special_tokens=False)
        t_txt = _OTOK.decode(r["target_ids"], skip_special_tokens=False)
        user_txt = _strip_user_assistant(p_txt)
        t_clean = t_txt.replace("<|endoftext|>", "").strip()
        if not user_txt or not t_clean:
            continue

        if src == "openthoughts" and "<|begin_of_thought|>" in t_clean:
            # Native Inkling-2-Mini reasoning format
            body = t_clean.split("<|begin_of_thought|>", 1)[1].strip()
            if "<|end_of_thought|>" in body:
                thought, ans = body.split("<|end_of_thought|>", 1)
                thought = thought.strip()
                ans = ans.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()
                target_str = (
                    f"<|message_model|><|content_thinking|>{thought}<|end_message|>"
                    f"<|message_model|><|content_text|>{ans}<|end_message|><|content_model_end_sampling|>"
                )
            else:
                target_str = f"<|message_model|><|content_thinking|>{body}"
            effort = "low"
        else:
            target_str = f"<|message_model|><|content_text|>{t_clean}<|end_message|><|content_model_end_sampling|>"
            effort = "none"

        prompt_str = _ITOK.apply_chat_template(
            [{"role": "user", "content": user_txt}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
        p_ids = _ITOK.encode(prompt_str, add_special_tokens=False)
        t_ids = _ITOK.encode(target_str, add_special_tokens=False)

        # Keep head+tail if prompt exceeds _MAX_PREFIX so system/effort header and final question both survive
        if len(p_ids) > _MAX_PREFIX:
            head = 128
            p_ids = p_ids[:head] + p_ids[-(_MAX_PREFIX - head) :]
        if len(t_ids) > _MAX_TARGET:
            t_ids = t_ids[:_MAX_TARGET]
        if len(p_ids) < 4 or len(t_ids) < 4:
            continue

        rec = {
            "source": src,
            "effort": effort,
            "prefix_ids": p_ids,
            "suffix_ids": t_ids,
            "effective_tokens": len(t_ids),
        }
        out.append(json.dumps(rec))
    return out


def prepare_train(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.mix_jsonl) as f:
        all_lines = f.readlines()
    print(f"Loaded {len(all_lines)} raw rows from {args.mix_jsonl}", flush=True)

    # Deterministic shuffle before slicing/converting so target budget preserves source balance
    rng = random.Random(args.seed)
    rng.shuffle(all_lines)
    if args.max_rows and len(all_lines) > args.max_rows:
        all_lines = all_lines[: args.max_rows]

    chunk_size = 512
    chunks = [all_lines[i : i + chunk_size] for i in range(0, len(all_lines), chunk_size)]
    counts = defaultdict(int)
    tokens = defaultdict(int)
    total_rows = 0
    total_tokens = 0
    out_path = args.output / "train.jsonl"

    with Pool(
        processes=args.workers,
        initializer=_init_workers,
        initargs=(str(args.ouro_tok), str(args.model_path), args.max_prefix, args.max_target),
    ) as pool, out_path.open("w") as out_f:
        for res in pool.imap(_convert_train_chunk, chunks):
            for line in res:
                r = json.loads(line)
                counts[r["source"]] += 1
                tokens[r["source"]] += r["effective_tokens"]
                total_rows += 1
                total_tokens += r["effective_tokens"]
                out_f.write(line + "\n")
                if args.target_tokens and total_tokens >= args.target_tokens:
                    break
            if args.target_tokens and total_tokens >= args.target_tokens:
                break

    manifest = {
        "rows": total_rows,
        "effective_tokens": total_tokens,
        "counts_by_source": dict(counts),
        "tokens_by_source": dict(tokens),
        "max_prefix": args.max_prefix,
        "max_target": args.max_target,
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def _chat_ids(itok, user_content: str, effort: str = "none") -> list[int]:
    s = itok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
    )
    return itok.encode(s, add_special_tokens=False)


def _extract_gsm8k_gold(ans: str) -> str:
    m = re.search(r"####\s*([^\n]+)", ans)
    return m.group(1).strip().replace(",", "") if m else ans.strip()


def prepare_eval(args):
    args.output.mkdir(parents=True, exist_ok=True)
    itok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    otok = AutoTokenizer.from_pretrained(args.ouro_tok, local_files_only=True, trust_remote_code=True)
    summary = {}

    # 1. GSM8K (1,319 test)
    gsm_in = args.eval_raw / "gsm8k.jsonl"
    if gsm_in.exists():
        rows = []
        for i, line in enumerate(gsm_in.open()):
            r = json.loads(line)
            prompt = (
                f"Solve the following math problem step by step. "
                f"Put your final numerical answer inside \\boxed{{}}.\n\n{r['question'].strip()}"
            )
            rows.append({
                "id": f"gsm8k/{i}",
                "task": "gsm8k",
                "prompt_ids": _chat_ids(itok, prompt, effort="none"),
                "answer": _extract_gsm8k_gold(r["answer"]),
                "max_new_tokens": 512,
                "until": [],
            })
        (args.output / "gsm8k.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["gsm8k"] = len(rows)

    # 2. MATH-500 (500 test, extracted cleanly from bench/math500.jsonl -> effort=none & effort=low)
    math_in = args.bench_dir / "math500.jsonl"
    if math_in.exists():
        for eff, fname, max_tok in [("none", "math500.jsonl", 1024), ("low", "math500_low.jsonl", 2048)]:
            rows = []
            for line in math_in.open():
                r = json.loads(line)
                txt = otok.decode(r["prompt_ids"], skip_special_tokens=True)
                # Extract the final problem after the 4 Minerva few-shot examples
                prob = txt.rsplit("Problem:\n", 1)[-1].split("\n\nSolution:", 1)[0].strip()
                prompt = (
                    f"Solve the following mathematics problem step by step. "
                    f"Put your final answer inside \\boxed{{}}.\n\n{prob}"
                )
                rows.append({
                    "id": r["id"],
                    "task": r["task"],
                    "level": r["level"],
                    "prompt_ids": _chat_ids(itok, prompt, effort=eff),
                    "answer": r["answer"],
                    "max_new_tokens": max_tok,
                    "until": [],
                })
            (args.output / fname).write_text("".join(json.dumps(r) + "\n" for r in rows))
            summary[fname[:-6]] = len(rows)

    # 3. MMLU-Pro-420 (stratified 30 per subject x 14 subjects, deterministic seed 42)
    mpro_in = args.eval_raw / "mmlu_pro.jsonl"
    if mpro_in.exists():
        by_cat = defaultdict(list)
        for line in mpro_in.open():
            r = json.loads(line)
            by_cat[r["category"]].append(r)
        rng = random.Random(42)
        rows = []
        letters = "ABCDEFGHIJ"
        for cat in sorted(by_cat):
            items = list(by_cat[cat])
            rng.shuffle(items)
            for i, r in enumerate(items[:30]):
                opts = "\n".join(f"{letters[k]}. {opt}" for k, opt in enumerate(r["options"]))
                prompt = (
                    f"Answer the following multiple-choice question about {cat}. "
                    f"Explain briefly and conclude with `Answer: (X)` where X is the correct option letter.\n\n"
                    f"Question: {r['question'].strip()}\n\nOptions:\n{opts}"
                )
                rows.append({
                    "id": f"mmlu_pro/{cat}/{r.get('question_id', i)}",
                    "task": cat,
                    "prompt_ids": _chat_ids(itok, prompt, effort="none"),
                    "answer": r["answer"].strip().upper(),
                    "max_new_tokens": 512,
                    "until": [],
                })
        (args.output / "mmlu_pro.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["mmlu_pro"] = len(rows)

    # 4. ARC-Challenge (1,172 test)
    arc_in = args.eval_raw / "arc_c.jsonl"
    if arc_in.exists():
        rows = []
        for i, line in enumerate(arc_in.open()):
            r = json.loads(line)
            labels = r["choices"]["label"]
            texts = r["choices"]["text"]
            # Normalize numeric labels 1..4 to A..D
            norm_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
            norm_labels = [norm_map.get(str(l), str(l).upper()) for l in labels]
            gold = norm_map.get(str(r["answerKey"]), str(r["answerKey"]).upper())
            opts = "\n".join(f"{l}. {t}" for l, t in zip(norm_labels, texts))
            prompt = (
                f"Answer the following science multiple-choice question. "
                f"Reply with `Answer: (X)` where X is the single correct option letter.\n\n"
                f"Question: {r['question'].strip()}\n\nOptions:\n{opts}"
            )
            rows.append({
                "id": f"arc_c/{r.get('id', i)}",
                "task": "arc_c",
                "prompt_ids": _chat_ids(itok, prompt, effort="none"),
                "answer": gold,
                "max_new_tokens": 256,
                "until": [],
            })
        (args.output / "arc_c.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["arc_c"] = len(rows)

    # 5. BBH-540 (20 per task x 27 tasks, deterministic)
    bbh_in = args.bench_dir / "bbh.jsonl"
    if bbh_in.exists():
        by_task = defaultdict(list)
        for line in bbh_in.open():
            r = json.loads(line)
            by_task[r["task"]].append(r)
        rng = random.Random(42)
        rows = []
        for task in sorted(by_task):
            items = list(by_task[task])
            rng.shuffle(items)
            for r in items[:20]:
                txt = otok.decode(r["prompt_ids"], skip_special_tokens=True)
                prompt = (
                    f"{txt}\n\nContinue the step-by-step reasoning for the final question above "
                    f"and end with `So the answer is <answer>.`"
                )
                rows.append({
                    "id": r["id"],
                    "task": task,
                    "prompt_ids": _chat_ids(itok, prompt, effort="none"),
                    "answer": r["answer"],
                    "max_new_tokens": 384,
                    "until": [],
                })
        (args.output / "bbh.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["bbh"] = len(rows)

    # 6. RULER-4K, 8K, 16K (520 each)
    for name in ("ruler4k", "ruler8k", "ruler16k"):
        p = args.bench_dir / f"{name}.jsonl"
        if not p.exists():
            continue
        rows = []
        for line in p.open():
            r = json.loads(line)
            txt = otok.decode(r["prompt_ids"], skip_special_tokens=True)
            rows.append({
                "id": r["id"],
                "task": r["task"],
                "prompt_ids": _chat_ids(itok, txt, effort="none"),
                "answers": r.get("answers", []),
                "max_new_tokens": r["max_new_tokens"],
                "until": [],
            })
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary[name] = len(rows)

    # 7. HELMET (1,108 from bench-mini/helmet.jsonl)
    hel_in = args.bench_mini_dir / "helmet.jsonl"
    if hel_in.exists():
        rows = []
        for line in hel_in.open():
            r = json.loads(line)
            txt = otok.decode(r["prompt_ids"], skip_special_tokens=True)
            rec = {
                "id": r["id"],
                "task": r["task"],
                "length": r["length"],
                "metric": r["metric"],
                "prompt_ids": _chat_ids(itok, txt, effort="none"),
                "answers": r.get("answers", []),
                "max_new_tokens": max(32, r["max_new_tokens"]),
                "until": [],
            }
            if "qrel" in r:
                rec["qrel"] = r["qrel"]
            rows.append(rec)
        (args.output / "helmet.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        summary["helmet"] = len(rows)

    # 8. LongBench-v2 (official Inkling-2-Mini model card benchmark: 503 rows, capped to 32K for fast eval & also short subset <= 32K)
    lb2_in = args.eval_raw / "longbench_v2.jsonl"
    if lb2_in.exists():
        for eff, fname, max_ctx, max_tok in [
            ("none", "longbench_v2.jsonl", 32768, 256),
            ("low", "longbench_v2_low.jsonl", 32768, 2048),
        ]:
            rows = []
            for line in lb2_in.open():
                r = json.loads(line)
                ctx = r["context"]
                # Official LongBench v2 template
                body = (
                    "Please read the following text and answer the question below.\n\n"
                    f"<text>\n{ctx}\n</text>\n\n"
                    "What is the correct answer to this question: "
                    f"{r['question']}\n"
                    "Choices:\n"
                    f"(A) {r['choice_A']}\n"
                    f"(B) {r['choice_B']}\n"
                    f"(C) {r['choice_C']}\n"
                    f"(D) {r['choice_D']}\n\n"
                    'Format your response as follows: "The correct answer is (insert answer here)".'
                )
                ids = _chat_ids(itok, body, effort=eff)
                # Middle truncation if > max_ctx (exact LongBench-v2 standard protocol)
                if len(ids) > max_ctx:
                    half = max_ctx // 2
                    ids = ids[:half] + ids[-(max_ctx - half) :]
                rows.append({
                    "id": f"lb2/{r['_id']}",
                    "task": f"{r['domain']}/{r['length']}/{r['difficulty']}",
                    "length_bucket": r["length"],
                    "difficulty": r["difficulty"],
                    "prompt_ids": ids,
                    "answer": r["answer"].strip().upper(),
                    "max_new_tokens": max_tok,
                    "until": [],
                })
            (args.output / fname).write_text("".join(json.dumps(r) + "\n" for r in rows))
            summary[fname[:-6]] = len(rows)

    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--ouro-tok", type=Path, default=Path("/dev/shm/inkling/ouro-tok"))
    parser.add_argument("--mix-jsonl", type=Path, default=Path("/dev/shm/inkling/data/mix-data-48m/gsm_train.jsonl"))
    parser.add_argument("--bench-dir", type=Path, default=Path("/dev/shm/inkling/data/bench"))
    parser.add_argument("--bench-mini-dir", type=Path, default=Path("/dev/shm/inkling/data/bench-mini"))
    parser.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prefix", type=int, default=3072)
    parser.add_argument("--max-target", type=int, default=1024)
    parser.add_argument("--target-tokens", type=int, default=65_000_000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=31847)
    args = parser.parse_args()
    if args.mode == "train":
        prepare_train(args)
    else:
        prepare_eval(args)


if __name__ == "__main__":
    main()
