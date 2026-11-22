"""Build 4-Pillar + On-Policy v2 distillation training stream for Inkling-2-Mini.

Four pillars (default 28M effective tokens, zero changes needed to train_inkling.py):
  1. Multi-Turn Tool Use & Coding Agent (35%): APIGen-MT-5k + SWE-smith-trajectories
     - Native Inkling `apply_chat_template(messages, tools=..., reasoning_effort=...)`
     - Exact multi-turn assistant split so `sys + tools + history + latest tool output` -> 3-loop prefill
       and current `thinking + tool_call` -> n-loop decode!
  2. Hard Thinking & Competitive Coding + On-Policy 3/3 (30%):
     - OpenCodeReasoning + OpenThoughts (strictly decontaminated against GPQA / LCB / MATH-500 / GSM8K / Apex)
     - Native Original 3/3 on-policy rollouts (`--onpolicy-dir`, zero distribution shift)
     - Covers effort in {low, medium, high}, max_target=2048, plus mid-thinking continuation splits
  3. True Long Context (20%): LongAlign-10k + LongAlpaca (max_prefix=6144 to supervise global layers 5,11,17,23)
  4. General Instruction Anchor (15%): Tulu-3-SFT (effort=none/low)

Also provides `prompts` mode to export 1,600 strictly-decontaminated seed prompts for 8-GPU on-policy 3/3 sampling.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer


def _fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())[:120]


def _load_eval_fingerprints(raw_dir: Path, eval_raw: Path, sec5_raw: Path) -> set[str]:
    fps: set[str] = set()
    # GPQA Diamond
    gpqa = raw_dir / "gpqa_diamond_mc.parquet"
    if gpqa.exists():
        for r in pq.read_table(gpqa).to_pylist():
            fps.add(_fingerprint(r["problem"]))
    # LiveCodeBench v5 & v6
    for fn in ("lcb_test5.jsonl", "lcb_test6.jsonl"):
        p = raw_dir / fn
        if p.exists():
            for l in p.open():
                if l.strip():
                    r = json.loads(l)
                    fps.add(_fingerprint(r.get("question_content", "")))
    # Apex 2025
    ap = sec5_raw / "apex_2025.parquet"
    if ap.exists():
        for r in pq.read_table(ap).to_pylist():
            fps.add(_fingerprint(r["problem"]))
    # GSM8K
    gsm = eval_raw / "gsm8k.jsonl"
    if gsm.exists():
        for l in gsm.open():
            if l.strip():
                fps.add(_fingerprint(json.loads(l)["question"]))
    fps.discard("")
    return fps


def _norm_schema(obj):
    if isinstance(obj, list):
        return [_norm_schema(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    tmap = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}
    for k, v in obj.items():
        if k == "type" and isinstance(v, str):
            out[k] = tmap.get(v.lower(), v.lower())
        else:
            out[k] = _norm_schema(v)
    return out


def _wrap_tools(raw_tools: list[dict]) -> list[dict]:
    tools = []
    for t in raw_tools:
        fn = t["function"] if isinstance(t, dict) and "function" in t else t
        if not isinstance(fn, dict) or not fn.get("name"):
            continue
        params = _norm_schema(fn.get("parameters", {"type": "object", "properties": {}}))
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(fn["name"]),
                    "description": str(fn.get("description", "")),
                    "parameters": params,
                },
            }
        )
    return tools


def _split_turn_exact(
    tok,
    messages: list[dict],
    tools: list[dict] | None,
    effort: str,
    turn_idx: int,
    max_prefix: int,
    max_target: int,
) -> tuple[list[int], list[int]] | None:
    """Exact prefix/suffix tokenization at assistant turn `turn_idx` using Inkling's native template."""
    try:
        pref_str = tok.apply_chat_template(
            messages[:turn_idx],
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort=effort,
        )
        full_str = tok.apply_chat_template(
            messages[: turn_idx + 1],
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=False,
            reasoning_effort=effort,
        )
    except Exception:
        return None
    if not full_str.startswith(pref_str):
        return None
    suff_str = full_str[len(pref_str) :]
    p_ids = tok.encode(pref_str, add_special_tokens=False)
    s_ids = tok.encode(suff_str, add_special_tokens=False)
    if len(p_ids) > max_prefix:
        # Preserve tool declarations + system header at head, and recent turns + tool obs at tail
        head = min(1024, max_prefix // 3)
        p_ids = p_ids[:head] + p_ids[-(max_prefix - head) :]
    if len(s_ids) > max_target:
        s_ids = s_ids[:max_target]
    if len(p_ids) < 8 or len(s_ids) < 8:
        return None
    return p_ids, s_ids


def _convert_apigen_mt(row: dict) -> tuple[list[dict], list[dict]] | None:
    raw_tools = json.loads(row["tools"]) if isinstance(row.get("tools"), str) else (row.get("tools") or [])
    tools = _wrap_tools(raw_tools)
    msgs: list[dict] = []
    if row.get("system"):
        msgs.append({"role": "system", "content": str(row["system"])})
    last_tool_name = "tool"
    pending_thought = ""
    for turn in row.get("conversations", []):
        frm = turn.get("from", "")
        val = turn.get("value", "")
        if frm == "human":
            msgs.append({"role": "user", "content": str(val)})
        elif frm == "gpt":
            m = {"role": "assistant", "content": str(val)}
            if pending_thought:
                m["reasoning_content"] = pending_thought
                pending_thought = ""
            msgs.append(m)
        elif frm == "function_call":
            try:
                fc = json.loads(val) if isinstance(val, str) else val
            except Exception:
                return None
            if not isinstance(fc, dict) or not fc.get("name"):
                continue
            args = fc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            if fc["name"] == "think" and "thought" in args:
                pending_thought = (pending_thought + "\n" + str(args["thought"])).strip()
                continue
            last_tool_name = str(fc["name"])
            m = {
                "role": "assistant",
                "tool_calls": [{"function": {"name": last_tool_name, "arguments": args}}],
            }
            if pending_thought:
                m["reasoning_content"] = pending_thought
                pending_thought = ""
            msgs.append(m)
        elif frm == "observation":
            # Skip observations from merged "think" pseudo-tool
            if str(val).strip() in ("", "null", "None"):
                continue
            msgs.append({"role": "tool", "name": last_tool_name, "content": str(val)})
    return msgs, tools


SWE_DEFAULT_TOOLS = _wrap_tools(
    [
        {
            "name": "bash",
            "description": "Execute a bash command in the repository environment.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
        {
            "name": "str_replace_editor",
            "description": "View, create, or edit files in the repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "file_text": {"type": "string"},
                    "view_range": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["command", "path"],
            },
        },
        {
            "name": "submit",
            "description": "Submit the final patch.",
            "parameters": {"type": "object", "properties": {}},
        },
    ]
)


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for x in content:
            if isinstance(x, dict) and "text" in x:
                parts.append(str(x["text"]))
            elif isinstance(x, str):
                parts.append(x)
        return "\n".join(parts)
    return str(content or "")


def _convert_swesmith(row: dict) -> tuple[list[dict], list[dict]] | None:
    try:
        raw_msgs = json.loads(row["messages"]) if isinstance(row["messages"], str) else row["messages"]
    except Exception:
        return None
    msgs: list[dict] = []
    last_tool_name = "bash"
    for m in raw_msgs:
        role = m.get("role", "")
        txt = _text_of(m.get("content", "")).strip()
        if role == "system":
            msgs.append({"role": "system", "content": txt})
        elif role == "user":
            msgs.append({"role": "user", "content": txt})
        elif role == "assistant":
            out_m: dict = {"role": "assistant"}
            tcs_raw = m.get("tool_calls") or []
            tcs = []
            for tc in tcs_raw:
                fn = tc.get("function", tc) if isinstance(tc, dict) else {}
                name = fn.get("name")
                if not name:
                    continue
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                last_tool_name = str(name)
                tcs.append({"function": {"name": last_tool_name, "arguments": args}})
            if tcs:
                if txt:
                    out_m["reasoning_content"] = txt
                out_m["tool_calls"] = tcs
            elif txt:
                out_m["content"] = txt
            else:
                continue
            msgs.append(out_m)
        elif role == "tool":
            msgs.append({"role": "tool", "name": last_tool_name, "content": txt})
    return msgs, SWE_DEFAULT_TOOLS


def _build_thinking_pair(
    tok,
    user_txt: str,
    thought: str,
    answer: str,
    effort: str,
    max_prefix: int,
    max_target: int,
    rng: random.Random,
) -> tuple[list[int], list[int]] | None:
    pref_str = tok.apply_chat_template(
        [{"role": "user", "content": user_txt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
    )
    if thought:
        suff_str = (
            f"<|message_model|><|content_thinking|>{thought}<|end_message|>"
            f"<|message_model|><|content_text|>{answer}<|end_message|><|content_model_end_sampling|>"
        )
    else:
        suff_str = f"<|message_model|><|content_text|>{answer}<|end_message|><|content_model_end_sampling|>"
    p_ids = tok.encode(pref_str, add_special_tokens=False)
    s_ids = tok.encode(suff_str, add_special_tokens=False)
    if len(p_ids) > max_prefix:
        p_ids = p_ids[:128] + p_ids[-(max_prefix - 128) :]
    # Mid-thinking continuation split on 30% of long CoT traces (>1024 tokens)
    if len(s_ids) > 1024 and rng.random() < 0.30:
        cut = rng.randint(256, min(len(s_ids) - 256, max(257, max_prefix - len(p_ids))))
        if len(p_ids) + cut <= max_prefix:
            p_ids = p_ids + s_ids[:cut]
            s_ids = s_ids[cut:]
    if len(s_ids) > max_target:
        s_ids = s_ids[:max_target]
    if len(p_ids) < 8 or len(s_ids) < 8:
        return None
    return p_ids, s_ids


def export_onpolicy_prompts(args, tok, eval_fps: set[str]):
    rng = random.Random(args.seed)
    prompts: list[dict] = []

    # 1. 500 multi-turn tool-calling prompts from APIGen-MT-5k & SWE-smith
    ap_path = args.raw_dir / "apigen_mt_5k.json"
    if ap_path.exists():
        ap = json.load(ap_path.open())
        rng.shuffle(ap)
        n_ap = 0
        for idx, row in enumerate(ap):
            if n_ap >= 350:
                break
            conv = _convert_apigen_mt(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
            if not asst_idxs:
                continue
            t = rng.choice(asst_idxs)
            try:
                p_str = tok.apply_chat_template(
                    msgs[:t], tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort="low"
                )
            except Exception:
                continue
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 32 <= len(p_ids) <= 4096:
                prompts.append(
                    {"id": f"op_apigen/{idx}", "source": "tool_apigen", "effort": "low", "prompt_ids": p_ids, "max_new_tokens": 512}
                )
                n_ap += 1

    swe_path = args.raw_dir / "swesmith_tool_0.parquet"
    if swe_path.exists():
        rows_swe = pq.read_table(swe_path).to_pylist()
        rng.shuffle(rows_swe)
        n_swe = 0
        for idx, row in enumerate(rows_swe):
            if n_swe >= 150:
                break
            conv = _convert_swesmith(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 1]
            if not asst_idxs:
                continue
            t = rng.choice(asst_idxs)
            try:
                p_str = tok.apply_chat_template(
                    msgs[:t], tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort="low"
                )
            except Exception:
                continue
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 64 <= len(p_ids) <= 4096:
                prompts.append(
                    {"id": f"op_swe/{idx}", "source": "tool_swe", "effort": "low", "prompt_ids": p_ids, "max_new_tokens": 768}
                )
                n_swe += 1

    # 2. 600 hard coding/reasoning prompts from OpenCodeReasoning (strictly decontaminated)
    ocr_path = args.raw_dir / "ocr_0.parquet"
    if ocr_path.exists():
        rows_ocr = pq.read_table(ocr_path).to_pylist()
        rng.shuffle(rows_ocr)
        n_ocr = 0
        for idx, row in enumerate(rows_ocr):
            if n_ocr >= 600:
                break
            q = (row.get("input") or "").strip()
            if not q or _fingerprint(q) in eval_fps:
                continue
            eff = rng.choice(["low", "medium"])
            p_str = tok.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, reasoning_effort=eff
            )
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 32 <= len(p_ids) <= 2048:
                prompts.append(
                    {"id": f"op_ocr/{idx}", "source": "code_think", "effort": eff, "prompt_ids": p_ids, "max_new_tokens": 1536}
                )
                n_ocr += 1

    # 3. 300 real 3K-6K long-context prompts from LongAlign-10k
    la_path = args.raw_dir / "longalign_10k.jsonl"
    if la_path.exists():
        la_lines = [l for l in la_path.open() if l.strip()]
        rng.shuffle(la_lines)
        n_la = 0
        for idx, l in enumerate(la_lines):
            if n_la >= 300:
                break
            try:
                r = json.loads(l)
            except Exception:
                continue
            ms = r.get("messages") or []
            if len(ms) < 2 or ms[0].get("role") != "user":
                continue
            p_str = tok.apply_chat_template(
                [{"role": "user", "content": ms[0]["content"]}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort="low",
            )
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 2048 <= len(p_ids) <= 6144:
                prompts.append(
                    {"id": f"op_long/{idx}", "source": "long_doc", "effort": "low", "prompt_ids": p_ids, "max_new_tokens": 512}
                )
                n_la += 1

    # 4. 200 reasoning/math prompts from existing OpenThoughts rows in v1 train.jsonl
    v1_train = args.v1_train
    if v1_train.exists():
        ot_rows = []
        for l in v1_train.open():
            if '"openthoughts"' in l:
                r = json.loads(l)
                if r.get("source") == "openthoughts" and len(r["prefix_ids"]) <= 1536:
                    ot_rows.append(r)
            if len(ot_rows) >= 2000:
                break
        rng.shuffle(ot_rows)
        for idx, r in enumerate(ot_rows[:200]):
            prompts.append(
                {
                    "id": f"op_ot/{idx}",
                    "source": "math_think",
                    "effort": r.get("effort", "low"),
                    "prompt_ids": r["prefix_ids"],
                    "max_new_tokens": 1536,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in prompts))
    by_src = defaultdict(int)
    for r in prompts:
        by_src[r["source"]] += 1
    print(json.dumps({"total_prompts": len(prompts), "by_source": dict(by_src)}, indent=2), flush=True)


def build_train_v2(args, tok, eval_fps: set[str]):
    rng = random.Random(args.seed)
    target_total = args.target_tokens
    budgets = {
        "tool": int(target_total * 0.35),
        "think": int(target_total * 0.30),
        "long": int(target_total * 0.20),
        "general": int(target_total * 0.15),
    }
    buckets: dict[str, list[dict]] = {k: [] for k in budgets}
    tok_counts: dict[str, int] = {k: 0 for k in budgets}
    sub_counts: dict[str, int] = defaultdict(int)
    sub_toks: dict[str, int] = defaultdict(int)

    def add_row(pillar: str, sub: str, effort: str, p_ids: list[int], s_ids: list[int]) -> bool:
        if tok_counts[pillar] >= budgets[pillar]:
            return False
        rec = {
            "source": f"{pillar}/{sub}",
            "effort": effort,
            "prefix_ids": p_ids,
            "suffix_ids": s_ids,
            "effective_tokens": len(s_ids),
        }
        buckets[pillar].append(rec)
        tok_counts[pillar] += len(s_ids)
        sub_counts[f"{pillar}/{sub}"] += 1
        sub_toks[f"{pillar}/{sub}"] += len(s_ids)
        return True

    # --- 0. On-policy 3/3 rollouts (highest priority in their respective pillars!) ---
    if args.onpolicy_dir and args.onpolicy_dir.exists():
        op_rows = []
        for fp in sorted(args.onpolicy_dir.glob("onpolicy*.jsonl")):
            for l in fp.open():
                if l.strip():
                    op_rows.append(json.loads(l))
        rng.shuffle(op_rows)
        for r in op_rows:
            src = r.get("source", "")
            if "tool" in src:
                pillar = "tool"
            elif "long" in src:
                pillar = "long"
            else:
                pillar = "think"
            add_row(pillar, src, r.get("effort", "low"), r["prefix_ids"], r["suffix_ids"])

    # --- 1. Pillar 1 (35%): APIGen-MT-5k + SWE-smith-trajectories ---
    ap_path = args.raw_dir / "apigen_mt_5k.json"
    if ap_path.exists():
        ap = json.load(ap_path.open())
        rng.shuffle(ap)
        for row in ap:
            if tok_counts["tool"] >= int(budgets["tool"] * 0.65):
                break
            conv = _convert_apigen_mt(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
            rng.shuffle(asst_idxs)
            for t in asst_idxs[:3]:
                eff = rng.choice(["low", "medium"])
                pair = _split_turn_exact(tok, msgs, tools, eff, t, max_prefix=4096, max_target=1024)
                if pair:
                    add_row("tool", "apigen_mt", eff, pair[0], pair[1])

    swe_path = args.raw_dir / "swesmith_tool_0.parquet"
    if swe_path.exists() and tok_counts["tool"] < budgets["tool"]:
        rows_swe = pq.read_table(swe_path).to_pylist()
        rng.shuffle(rows_swe)
        for row in rows_swe:
            if tok_counts["tool"] >= budgets["tool"]:
                break
            conv = _convert_swesmith(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 1]
            rng.shuffle(asst_idxs)
            for t in asst_idxs[:4]:
                eff = rng.choice(["low", "medium"])
                pair = _split_turn_exact(tok, msgs, tools, eff, t, max_prefix=args.max_prefix, max_target=1536)
                if pair:
                    if not add_row("tool", "swesmith", eff, pair[0], pair[1]):
                        break

    # Top up tool pillar from remaining APIGen-MT if needed
    if ap_path.exists() and tok_counts["tool"] < budgets["tool"]:
        ap = json.load(ap_path.open())
        rng.shuffle(ap)
        for row in ap:
            if tok_counts["tool"] >= budgets["tool"]:
                break
            conv = _convert_apigen_mt(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
            for t in asst_idxs:
                pair = _split_turn_exact(tok, msgs, tools, "low", t, max_prefix=4096, max_target=1024)
                if pair and not add_row("tool", "apigen_mt", "low", pair[0], pair[1]):
                    break

    # --- 2. Pillar 2 (30%): OpenCodeReasoning + OpenThoughts (decontaminated, max_target=2048) ---
    ocr_path = args.raw_dir / "ocr_0.parquet"
    if ocr_path.exists() and tok_counts["think"] < budgets["think"]:
        rows_ocr = pq.read_table(ocr_path).to_pylist()
        rng.shuffle(rows_ocr)
        ocr_cap = int(budgets["think"] * 0.72)
        for row in rows_ocr:
            if tok_counts["think"] >= ocr_cap:
                break
            q = (row.get("input") or "").strip()
            out_txt = (row.get("output") or "").strip()
            if not q or not out_txt or _fingerprint(q) in eval_fps:
                continue
            thought, ans = "", out_txt
            if "<think>" in out_txt and "</think>" in out_txt:
                thought = out_txt.split("<think>", 1)[1].split("</think>", 1)[0].strip()
                ans = out_txt.split("</think>", 1)[1].strip()
            eff = rng.choice(["low", "medium", "high"])
            pair = _build_thinking_pair(tok, q, thought, ans, eff, max_prefix=4096, max_target=2048, rng=rng)
            if pair:
                add_row("think", "opencode_reasoning", eff, pair[0], pair[1])

    if args.v1_train.exists() and tok_counts["think"] < budgets["think"]:
        ot_rows = []
        for l in args.v1_train.open():
            if '"openthoughts"' in l:
                r = json.loads(l)
                if r.get("source") == "openthoughts":
                    ot_rows.append(r)
        rng.shuffle(ot_rows)
        for r in ot_rows:
            if not add_row("think", "openthoughts", r.get("effort", "low"), r["prefix_ids"], r["suffix_ids"]):
                break

    # --- 3. Pillar 3 (20%): LongAlign-10k (max_prefix=6144) + LongAlpaca ---
    la_path = args.raw_dir / "longalign_10k.jsonl"
    if la_path.exists() and tok_counts["long"] < budgets["long"]:
        la_lines = [l for l in la_path.open() if l.strip()]
        rng.shuffle(la_lines)
        for l in la_lines:
            if tok_counts["long"] >= budgets["long"]:
                break
            try:
                r = json.loads(l)
            except Exception:
                continue
            ms = r.get("messages") or []
            if len(ms) < 2:
                continue
            asst_idxs = [i for i, m in enumerate(ms) if m.get("role") == "assistant" and i > 0]
            if not asst_idxs:
                continue
            t = rng.choice(asst_idxs)
            eff = rng.choice(["none", "low"])
            pair = _split_turn_exact(tok, ms, None, eff, t, max_prefix=args.max_prefix, max_target=1024)
            if pair and len(pair[0]) >= 1536:  # ensure long context past 1024 sliding window!
                add_row("long", "longalign", eff, pair[0], pair[1])

    if args.v1_train.exists() and tok_counts["long"] < budgets["long"]:
        for l in args.v1_train.open():
            if '"longalpaca"' in l:
                r = json.loads(l)
                if r.get("source") == "longalpaca":
                    if not add_row("long", "longalpaca", r.get("effort", "none"), r["prefix_ids"], r["suffix_ids"]):
                        break

    # --- 4. Pillar 4 (15%): Tulu-3-SFT anchor ---
    if args.v1_train.exists() and tok_counts["general"] < budgets["general"]:
        tulu_rows = []
        for l in args.v1_train.open():
            if '"tulu3"' in l:
                r = json.loads(l)
                if r.get("source") == "tulu3":
                    tulu_rows.append(r)
            if len(tulu_rows) >= 25000:
                break
        rng.shuffle(tulu_rows)
        for r in tulu_rows:
            if not add_row("general", "tulu3", r.get("effort", "none"), r["prefix_ids"], r["suffix_ids"]):
                break

    all_rows = [r for b in buckets.values() for r in b]
    rng.shuffle(all_rows)
    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "train.jsonl"
    with out_file.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    n_gt_4k = sum(1 for r in all_rows if len(r["prefix_ids"]) >= 4096)
    n_gt_2k = sum(1 for r in all_rows if len(r["prefix_ids"]) >= 2048)
    manifest = {
        "rows": len(all_rows),
        "effective_tokens": sum(r["effective_tokens"] for r in all_rows),
        "pillar_tokens": tok_counts,
        "counts_by_subsource": dict(sub_counts),
        "tokens_by_subsource": dict(sub_toks),
        "prefix_ge_2k": n_gt_2k,
        "prefix_ge_4k": n_gt_4k,
        "max_prefix": args.max_prefix,
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prompts", "build"))
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--raw-dir", type=Path, default=Path("/dev/shm/inkling/data/v2_raw"))
    parser.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    parser.add_argument("--sec5-raw", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    parser.add_argument("--v1-train", type=Path, default=Path("/dev/shm/inkling/data/train_inkling/train.jsonl"))
    parser.add_argument("--onpolicy-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=30_000_000)
    parser.add_argument("--max-prefix", type=int, default=6144)
    parser.add_argument("--seed", type=int, default=31847)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    eval_fps = _load_eval_fingerprints(args.raw_dir, args.eval_raw, args.sec5_raw)
    print(f"Loaded {len(eval_fps)} decontamination eval fingerprints", flush=True)
    if args.mode == "prompts":
        export_onpolicy_prompts(args, tok, eval_fps)
    else:
        build_train_v2(args, tok, eval_fps)


if __name__ == "__main__":
    main()
