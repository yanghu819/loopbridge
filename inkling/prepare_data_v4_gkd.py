"""Build LoopBridge v2 (v4) On-Policy GKD + Ultra-Long (16K-52K) Multi-Needle/Graph Datasets.

Targets the exact 3 root causes of residual quality drop vs full 3/3:
  1. Ultra-long LogScale (>8,192) retrieval gap on MRCR 256K-512K & GraphWalks 256K:
     Generates 1,200 100% synthetic, 13-gram-decontaminated 16K-52K multi-needle verbatim
     ordinal-retrieval (8-12 needles + hard distractors) and 16K-40K random-graph BFS prompts
     (which LoopBridge trains on effortlessly because prefill is frozen torch.no_grad()).
  2. Exposure-bias thinking-loop truncation on LiveCodeBench v6 & GPQA Diamond:
     Samples on-policy student rollouts (GKD) from the v1 LoopBridge student on hard STEM/coding
     prompts so the frozen 3/3 teacher scores and corrects the student's own token distribution
     under Hessian-matched Generalized JSD.
  3. Off-policy mismatch on TML-d24-varloop-research:
     Generates native 100% on-policy TML-varloop 3/3 teacher rollouts on STEM/coding/long/tool prompts.

Modes:
  - `prompts`      : Build `prompts_ultralong_v4.jsonl`, `prompts_gkd_hard_v4.jsonl`, `prompts_vl_teacher_v4.jsonl`.
  - `rollout_vllm` : High-concurrency least-loaded 8-replica vLLM rollout collector (teacher or student_gkd).
  - `build`        : Merge teacher + student GKD + 16K-52K ultra-long rollouts, verify 0 eval leaks, emit train.jsonl.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import string
import threading
from time import perf_counter
import urllib.request

from transformers import AutoTokenizer


def _make_ultralong_verbatim_recall(
    tok,
    paras: list[str],
    rng: random.Random,
    target_tokens: int,
    effort: str,
) -> tuple[list[int], str] | None:
    """Procedurally generate a 16K-52K multi-turn multi-needle verbatim-copy prompt (zero test overlap)."""
    from inkling.prepare_data_v3 import RECALL_FORMS, RECALL_TOPICS

    if len(paras) < 80:
        return None
    target_form = rng.choice(RECALL_FORMS)
    target_topic = rng.choice(RECALL_TOPICS)
    other_forms = [f for f in RECALL_FORMS if f != target_form]
    other_topics = [t for t in RECALL_TOPICS if t != target_topic]

    # 8-12 needles of the target (form, topic) interleaved across 16K-52K tokens of hard distractors
    n_needles = rng.randint(8, 12)
    est_total_turns = max(n_needles + 20, int(target_tokens / 195))
    needle_slots = set(rng.sample(range(est_total_turns), min(n_needles, est_total_turns)))

    msgs: list[dict] = []
    est_tokens = 0
    needle_turns = 0
    turn_idx = 0

    while est_tokens < target_tokens - 200 and turn_idx < est_total_turns + 60:
        snippet = paras[rng.randrange(len(paras))]
        if turn_idx in needle_slots or (needle_turns < n_needles and est_tokens > target_tokens * 0.85):
            form, topic = target_form, target_topic
            needle_turns += 1
        else:
            if rng.random() < 0.5:
                form, topic = target_form, rng.choice(other_topics)
            else:
                form, topic = rng.choice(other_forms), target_topic
        code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=6))
        user_msg = f"Write a short {form} about {topic} (reference tag {code})."
        asst_msg = f"[{code}] {snippet}"
        msgs.append({"role": "user", "content": user_msg})
        msgs.append({"role": "assistant", "content": asst_msg})
        est_tokens += int((len(user_msg) + len(asst_msg)) / 3.6) + 12
        turn_idx += 1

    if needle_turns < 6:
        return None

    k = rng.randint(1, needle_turns)
    ord_suffix = {1: "st", 2: "nd", 3: "rd"}.get(k if k < 20 else k % 10, "th")
    rand_prefix = "".join(rng.choices(string.ascii_letters + string.digits, k=10))
    final_q = (
        f"Prepend {rand_prefix} to your final answer (do not add any extra commentary before or after it) "
        f"and reproduce verbatim the {k}{ord_suffix} {target_form} about {target_topic} that you wrote in "
        f"the conversation above."
    )
    msgs.append({"role": "user", "content": final_q})

    p_str = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    p_ids = tok.encode(p_str, add_special_tokens=False)
    if len(p_ids) < 15000 or len(p_ids) > 54000:
        return None
    return p_ids, f"ultralong_recall_k{k}_n{needle_turns}"


def _make_ultralong_graph_bfs(
    tok,
    rng: random.Random,
    target_tokens: int,
    effort: str,
) -> tuple[list[int], str] | None:
    """Procedurally generate a 16K-40K synthetic random-graph BFS prompt (zero test overlap)."""
    n_nodes = max(700, min(1950, int(target_tokens / 19)))
    alphabet = string.ascii_lowercase + string.digits
    seen: set[str] = set()
    nodes: list[str] = []
    while len(nodes) < n_nodes:
        nid = "".join(rng.choices(alphabet, k=5))
        if nid not in seen:
            seen.add(nid)
            nodes.append(nid)

    directed = rng.random() < 0.65
    adj: dict[str, list[str]] = {u: [] for u in nodes}
    for u in nodes:
        deg = rng.randint(1, 3)
        nbrs = rng.sample(nodes, deg)
        for v in nbrs:
            if v == u:
                continue
            if v not in adj[u]:
                adj[u].append(v)
            if not directed and u not in adj[v]:
                adj[v].append(u)

    start = None
    target_dist = rng.choice([2, 3])
    for _ in range(30):
        cand = rng.choice(nodes)
        dist = {cand: 0}
        q = [cand]
        for u in q:
            if dist[u] >= target_dist:
                continue
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        frontier = [u for u, d in dist.items() if d == target_dist]
        if 2 <= len(frontier) <= 20:
            start = cand
            break
    if start is None:
        return None

    order = list(nodes)
    rng.shuffle(order)
    edge_word = "directed edges" if directed else "undirected edges"
    lines = [f"Below is a graph with {edge_word} specified as adjacency lists:"]
    for u in order:
        nbr_str = ", ".join(adj[u]) if adj[u] else "none"
        lines.append(f"Node {u} connects to: {nbr_str}")
    lines.append("")
    lines.append(
        f"Starting from node {start}, perform a breadth-first search (BFS) and list all nodes at exact "
        f"shortest-path distance {target_dist} from {start}. Return them as a Python list in the format:\n"
        f"Final Answer: [node1, node2, ...]"
    )
    user_txt = "\n".join(lines)
    p_str = tok.apply_chat_template(
        [{"role": "user", "content": user_txt}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=effort,
    )
    p_ids = tok.encode(p_str, add_special_tokens=False)
    if len(p_ids) < 15000 or len(p_ids) > 42000:
        return None
    return p_ids, f"ultralong_graph_bfs_d{target_dist}_n{n_nodes}"


def cmd_prompts(args) -> None:
    from inkling.prepare_data_v3 import _load_paragraph_pool, load_eval_13grams

    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    forbidden = load_eval_13grams(args.eval_dir, args.eval_raw, args.sec5_raw, tok=tok)
    if args.eval_exact_dir and args.eval_exact_dir.exists():
        forbidden |= load_eval_13grams(args.eval_exact_dir, args.eval_raw, args.sec5_raw, tok=tok)
    print(f"[v4 prompts] Loaded {len(forbidden):,} forbidden eval 13-grams", flush=True)

    rng = random.Random(args.seed)
    ul_path = args.output / "prompts_ultralong_v4.jsonl"
    ul_target = args.n_ultralong_recall + args.n_ultralong_graph
    if ul_path.exists() and sum(1 for _ in ul_path.open()) >= ul_target:
        print(f"[v4 prompts] Reusing existing {ul_path} (>= {ul_target} rows)", flush=True)
    else:
        paras = _load_paragraph_pool(args.v2_raw, forbidden, max_paras=16000)
        print(f"[v4 prompts] Harvested {len(paras):,} clean paragraphs from longalign_10k", flush=True)

        ultralong: list[dict] = []

        # 1. 800 ultra-long 16K-52K multi-needle verbatim ordinal recall prompts
        n_recall_target = args.n_ultralong_recall
        attempts = 0
        while len(ultralong) < n_recall_target and attempts < n_recall_target * 4:
            attempts += 1
            target_tok = rng.randint(16500, 50000)
            effort = rng.choice(["high", "medium", "high"])
            res = _make_ultralong_verbatim_recall(tok, paras, rng, target_tok, effort)
            if res is None:
                continue
            p_ids, tag = res
            ultralong.append(
                {
                    "id": f"v4_ul_recall/{len(ultralong):05d}",
                    "pillar": "ultralong_recall",
                    "tag": tag,
                    "effort": effort,
                    "prefix_ids": p_ids,
                    "max_tokens": 1536,
                }
            )
        print(f"[v4 prompts] Generated {len(ultralong)} ultra-long recall prompts (16K-52K)", flush=True)

        # 2. 400 ultra-long 16K-40K synthetic random-graph BFS prompts
        n_graph_done = 0
        attempts = 0
        while n_graph_done < args.n_ultralong_graph and attempts < args.n_ultralong_graph * 4:
            attempts += 1
            target_tok = rng.randint(16500, 38000)
            effort = rng.choice(["high", "medium", "high"])
            res = _make_ultralong_graph_bfs(tok, rng, target_tok, effort)
            if res is None:
                continue
            p_ids, tag = res
            ultralong.append(
                {
                    "id": f"v4_ul_graph/{n_graph_done:05d}",
                    "pillar": "ultralong_graph",
                    "tag": tag,
                    "effort": effort,
                    "prefix_ids": p_ids,
                    "max_tokens": 1536,
                }
            )
            n_graph_done += 1
        rng.shuffle(ultralong)
        ul_path.write_text("".join(json.dumps(r) + "\n" for r in ultralong))
        ul_lens = sorted(len(r["prefix_ids"]) for r in ultralong)
        print(
            f"[v4 prompts] Wrote {len(ultralong)} ultra-long prompts -> {ul_path} "
            f"(min={ul_lens[0]}, med={ul_lens[len(ul_lens)//2]}, max={ul_lens[-1]})",
            flush=True,
        )

    # 3. Slice prompts_v3.jsonl for (a) On-Policy Student OPD/GKD and (b) Native VarLoop 3/3 Teacher Rollouts
    v3_rows = [json.loads(l) for l in args.prompts_v3.open() if l.strip()]
    by_pillar: dict[str, list[dict]] = {}
    for r in v3_rows:
        by_pillar.setdefault(r.get("pillar", "other"), []).append(r)
    for rows in by_pillar.values():
        rng.shuffle(rows)

    # GKD/OPD hard subset: 2,000 STEM/coding 'reason' + 400 'long' = 2,400 prompts (~3-4 min on 8x B300)
    gkd_rows = by_pillar.get("reason", [])[: args.n_gkd_reason] + by_pillar.get("long", [])[: args.n_gkd_long]
    rng.shuffle(gkd_rows)
    gkd_out = []
    for idx, r in enumerate(gkd_rows):
        p_ids = r.get("prefix_ids") or r["prompt_ids"]
        gkd_out.append(
            {
                "id": f"v4_gkd/{r.get('id', idx)}",
                "pillar": f"gkd_{r.get('pillar', 'reason')}",
                "prefix_ids": p_ids,
                "max_tokens": 3072,
            }
        )
    gkd_path = args.output / "prompts_gkd_hard_v4.jsonl"
    gkd_path.write_text("".join(json.dumps(r) + "\n" for r in gkd_out))
    print(f"[v4 prompts] Wrote {len(gkd_out)} GKD/OPD hard prompts -> {gkd_path}", flush=True)

    # Native VarLoop teacher subset: 1,200 reason + 500 long + 450 procedural + 250 tool = 2,400 (~3 min on 8x B300)
    vl_rows = (
        by_pillar.get("reason", [])[: args.n_vl_reason]
        + by_pillar.get("long", [])[: args.n_vl_long]
        + by_pillar.get("procedural", [])[: args.n_vl_proc]
        + by_pillar.get("tool", [])[: args.n_vl_tool]
    )
    rng.shuffle(vl_rows)
    vl_out = []
    for idx, r in enumerate(vl_rows):
        p_ids = r.get("prefix_ids") or r["prompt_ids"]
        vl_out.append(
            {
                "id": f"v4_vl_t/{r.get('id', idx)}",
                "pillar": r.get("pillar", "reason"),
                "prefix_ids": p_ids,
                "max_tokens": 2048,
            }
        )
    vl_path = args.output / "prompts_vl_teacher_v4.jsonl"
    vl_path.write_text("".join(json.dumps(r) + "\n" for r in vl_out))
    print(f"[v4 prompts] Wrote {len(vl_out)} VarLoop teacher prompts -> {vl_path}", flush=True)


def _format_native_sft_pair(
    tok,
    user_text: str,
    reasoning_text: str,
    answer_text: str,
    effort: str,
    min_suffix: int,
    max_suffix: int,
    min_prefix: int = 16,
    max_prefix: int = 48000,
) -> tuple[list[int], list[int]] | None:
    """Wrap (user, reasoning, answer) into exact native Inkling chat_template tokens with guaranteed EOS."""
    user_text = user_text.strip()
    answer_text = answer_text.strip()
    reasoning_text = reasoning_text.strip()
    if not user_text or not answer_text:
        return None
    msgs_p = [{"role": "user", "content": user_text}]
    asst: dict[str, str] = {"role": "assistant", "content": answer_text}
    if reasoning_text:
        asst["reasoning_content"] = reasoning_text
    msgs_full = [msgs_p[0], asst]
    p_str = tok.apply_chat_template(msgs_p, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    full_str = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False, reasoning_effort=effort)
    if not full_str.startswith(p_str):
        return None
    s_str = full_str[len(p_str) :]
    if "<|content_text|>" not in s_str or not s_str.endswith("<|content_model_end_sampling|>"):
        return None
    p_ids = tok.encode(p_str, add_special_tokens=False)
    s_ids = tok.encode(s_str, add_special_tokens=False)
    if not (min_prefix <= len(p_ids) <= max_prefix):
        return None
    if not (min_suffix <= len(s_ids) <= max_suffix):
        return None
    return p_ids, s_ids


def _split_thought_and_answer(raw: str) -> tuple[str, str] | None:
    """Extract (reasoning, final_answer) from OpenCodeReasoning (<think>...</think>) or OpenThoughts tags."""
    if "</think>" in raw:
        left, right = raw.split("</think>", 1)
        left = left.replace("<think>", "").strip()
        right = right.strip()
        if left and right:
            return left, right
    if "<|end_of_thought|>" in raw:
        left, right = raw.split("<|end_of_thought|>", 1)
        left = left.replace("<|begin_of_thought|>", "").strip()
        right = right.replace("<|begin_of_solution|>", "").replace("<|end_of_solution|>", "").strip()
        if left and right:
            return left, right
    return None


def cmd_sft_verified(args) -> None:
    """Build 100% EOS-complete, 13-gram-decontaminated, native-template SFT trajectories (0 GPU rollout cost)."""
    import pyarrow.parquet as pq
    from inkling.prepare_data_v3 import _has_13gram_overlap, load_eval_13grams

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    forbidden = load_eval_13grams(args.eval_dir, args.eval_raw, args.sec5_raw, tok=tok)
    if args.eval_exact_dir and args.eval_exact_dir.exists():
        forbidden |= load_eval_13grams(args.eval_exact_dir, args.eval_raw, args.sec5_raw, tok=tok)
    print(f"[v4 sft] Loaded {len(forbidden):,} forbidden eval 13-grams", flush=True)

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    leaks_skipped = 0

    # 1. Verified Competitive Coding SFT from NVIDIA OpenCodeReasoning (ocr_0.parquet)
    ocr_path = args.v2_raw / "ocr_0.parquet"
    if ocr_path.exists() and args.n_opencode > 0:
        t = pq.read_table(str(ocr_path))
        indices = list(range(t.num_rows))
        rng.shuffle(indices)
        kept = 0
        for idx in indices:
            if kept >= args.n_opencode:
                break
            raw_out = t["output"][idx].as_py() or ""
            if "</think>" not in raw_out or len(raw_out) > 13500:
                continue
            parsed = _split_thought_and_answer(raw_out)
            if parsed is None:
                continue
            u_txt = t["input"][idx].as_py() or ""
            if _has_13gram_overlap(u_txt, forbidden) or _has_13gram_overlap(parsed[1], forbidden):
                leaks_skipped += 1
                continue
            effort = rng.choice(["high", "high", "medium"])
            pair = _format_native_sft_pair(tok, u_txt, parsed[0], parsed[1], effort, 128, 3072, 16, 4096)
            if pair is None:
                continue
            p_ids, s_ids = pair
            out_rows.append(
                {
                    "id": f"v4_sft_code/{kept:05d}",
                    "pillar": "sft_opencode",
                    "source_mode": "sft_verified",
                    "finish_reason": "stop",
                    "prefix_ids": p_ids,
                    "suffix_ids": s_ids,
                    "effective_tokens": len(s_ids),
                }
            )
            kept += 1
        print(f"[v4 sft] OpenCodeReasoning: kept {kept} complete EOS rows (leaks_skipped={leaks_skipped})", flush=True)

    # 2. Verified Science + Math + Code Reasoning SFT from OpenThoughts-114k shards
    ot_dir = args.v2_raw / "openthoughts_114k"
    ot_files = sorted(ot_dir.glob("*.parquet")) if ot_dir.exists() else []
    if ot_files and args.n_openthoughts > 0:
        # Allocate balanced quota across shards (shard5 science/STEM gets priority!)
        per_shard_target = max(1, (args.n_openthoughts + len(ot_files) - 1) // len(ot_files))
        kept_ot = 0
        for sf in reversed(ot_files):
            t = pq.read_table(str(sf))
            indices = list(range(t.num_rows))
            if "shard5" in sf.name:
                # Prioritize graduate-level Science/STEM rows (tail of shard5, non-boxed) before math rows
                sci_idx, math_idx = [], []
                for idx in indices:
                    convs = t["conversations"][idx].as_py()
                    u0 = (convs[0].get("value", "") or "") if convs else ""
                    if "\\boxed{}" not in u0:
                        sci_idx.append(idx)
                    else:
                        math_idx.append(idx)
                rng.shuffle(sci_idx)
                rng.shuffle(math_idx)
                indices = sci_idx + math_idx
            else:
                rng.shuffle(indices)
            kept_shard = 0
            # Give shard5 (science/STEM + math) up to 60% of quota if available
            shard_cap = int(args.n_openthoughts * 0.60) if "shard5" in sf.name else per_shard_target
            for idx in indices:
                if kept_ot >= args.n_openthoughts or kept_shard >= shard_cap:
                    break
                convs = t["conversations"][idx].as_py()
                if not convs or len(convs) < 2:
                    continue
                u_txt = convs[0].get("value", "") or ""
                a_txt = convs[1].get("value", "") or ""
                if ("<|end_of_thought|>" not in a_txt and "</think>" not in a_txt) or len(a_txt) > 13500:
                    continue
                parsed = _split_thought_and_answer(a_txt)
                if parsed is None:
                    continue
                if _has_13gram_overlap(u_txt, forbidden) or _has_13gram_overlap(parsed[1], forbidden):
                    leaks_skipped += 1
                    continue
                effort = rng.choice(["high", "high", "medium"])
                pair = _format_native_sft_pair(tok, u_txt, parsed[0], parsed[1], effort, 128, 3072, 16, 4096)
                if pair is None:
                    continue
                p_ids, s_ids = pair
                out_rows.append(
                    {
                        "id": f"v4_sft_ot/{sf.stem}_{kept_shard:05d}",
                        "pillar": "sft_openthoughts",
                        "source_mode": "sft_verified",
                        "finish_reason": "stop",
                        "prefix_ids": p_ids,
                        "suffix_ids": s_ids,
                        "effective_tokens": len(s_ids),
                    }
                )
                kept_ot += 1
                kept_shard += 1
            print(f"[v4 sft] OpenThoughts {sf.name}: kept {kept_shard} complete EOS rows", flush=True)
        print(f"[v4 sft] OpenThoughts total: kept {kept_ot} rows (leaks_skipped={leaks_skipped})", flush=True)

    # 3. Untruncated Real Long-Document QA SFT (8K-46K prefixes, above 8,192 LogScale floor!) from longalign_10k
    la_path = args.v2_raw / "longalign_10k.jsonl"
    if la_path.exists() and args.n_longalign_ul > 0:
        la_cands = []
        with la_path.open(encoding="utf-8", errors="replace") as fh:
            for l in fh:
                try:
                    r = json.loads(l)
                except Exception:
                    continue
                msgs = r.get("messages", [])
                if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
                    continue
                # Quick char filter for ~8K-46K prefix
                p_chars = sum(len(str(m.get("content", ""))) for m in msgs[:-1])
                if 28000 <= p_chars <= 200000:
                    la_cands.append(msgs)
        rng.shuffle(la_cands)
        kept_la = 0
        for msgs in la_cands:
            if kept_la >= args.n_longalign_ul:
                break
            u_msgs = [{"role": m["role"], "content": str(m.get("content", ""))} for m in msgs[:-1] if m.get("role") in ("user", "assistant")]
            if not u_msgs or u_msgs[-1]["role"] != "user":
                continue
            ans = str(msgs[-1].get("content", "")).strip()
            if not ans or len(ans) > 10000:
                continue
            last_q = u_msgs[-1]["content"]
            if _has_13gram_overlap(last_q[-4000:], forbidden) or _has_13gram_overlap(ans, forbidden):
                leaks_skipped += 1
                continue
            effort = rng.choice(["medium", "low", "high"])
            p_str = tok.apply_chat_template(u_msgs, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
            full_str = tok.apply_chat_template(
                u_msgs + [{"role": "assistant", "content": ans}],
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort=effort,
            )
            if not full_str.startswith(p_str):
                continue
            s_str = full_str[len(p_str) :]
            if "<|content_text|>" not in s_str or not s_str.endswith("<|content_model_end_sampling|>"):
                continue
            p_ids = tok.encode(p_str, add_special_tokens=False)
            s_ids = tok.encode(s_str, add_special_tokens=False)
            if not (8192 <= len(p_ids) <= 46000) or not (48 <= len(s_ids) <= 1536):
                continue
            out_rows.append(
                {
                    "id": f"v4_sft_longalign/{kept_la:05d}",
                    "pillar": "sft_longalign_ul",
                    "source_mode": "sft_verified",
                    "finish_reason": "stop",
                    "prefix_ids": p_ids,
                    "suffix_ids": s_ids,
                    "effective_tokens": len(s_ids),
                }
            )
            kept_la += 1
        print(f"[v4 sft] LongAlign 8K-46K: kept {kept_la} complete EOS rows (leaks_skipped={leaks_skipped})", flush=True)

    rng.shuffle(out_rows)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    tot_tok = sum(r["effective_tokens"] for r in out_rows)
    print(
        f"[v4 sft] DONE: wrote {len(out_rows)} verified SFT rows ({tot_tok/1e6:.2f}M KL tokens, "
        f"leaks_skipped={leaks_skipped}) -> {args.output}",
        flush=True,
    )


def _post_rollout(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
) -> tuple[str, list[int], str]:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "ignore_eos": False,
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choice = body["choices"][0]
    return choice.get("text", ""), [], choice.get("finish_reason", "stop")


def cmd_rollout_vllm(args) -> None:
    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    prompts = [json.loads(l) for l in args.prompts.open() if l.strip()]
    if args.limit and args.limit < len(prompts):
        prompts = prompts[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[str] = set()
    if args.output.exists():
        for l in args.output.open():
            if l.strip():
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:
                    pass

    todo = [p for p in prompts if p["id"] not in done_ids]
    print(
        f"[v4 rollout] mode={args.rollout_mode} total={len(prompts)} done={len(done_ids)} "
        f"todo={len(todo)} urls={len(args.urls)} conc={args.concurrency}",
        flush=True,
    )
    if not todo:
        return

    urls = args.urls
    in_flight = {u: 0 for u in urls}
    rr = 0
    lock = threading.Lock()

    def acquire() -> str:
        nonlocal rr
        with lock:
            ml = min(in_flight.values())
            for off in range(len(urls)):
                u = urls[(rr + off) % len(urls)]
                if in_flight[u] == ml:
                    in_flight[u] += 1
                    rr = (rr + off + 1) % len(urls)
                    return u
            in_flight[urls[0]] += 1
            return urls[0]

    def release(u: str) -> None:
        with lock:
            in_flight[u] = max(0, in_flight[u] - 1)

    t0 = perf_counter()
    kept = 0
    dropped = 0
    tot_tok = 0

    def run_one(idx_row):
        idx, row = idx_row
        u = acquire()
        try:
            max_tok = int(args.max_tokens_override or row.get("max_tokens", 2048))
            for att in range(3):
                try:
                    txt, _, fr = _post_rollout(
                        u,
                        str(args.model_path),
                        row["prefix_ids"],
                        max_tok,
                        args.temperature,
                        args.top_p,
                        args.seed + idx + att * 997,
                        args.timeout,
                    )
                    if not txt or len(txt) < 16:
                        return None
                    if args.rollout_mode == "teacher":
                        # Teacher trajectories must terminate cleanly with visible text
                        if fr == "length":
                            return None
                        if "<|content_text|>" not in txt and "<|content_invoke_tool_json|>" not in txt:
                            return None
                    # Re-encode exact raw text (preserving special tokens) + append EOS if stopped naturally
                    suffix_ids = tok.encode(txt, add_special_tokens=False)
                    if fr != "length" and tok.eos_token_id is not None:
                        if not suffix_ids or suffix_ids[-1] != tok.eos_token_id:
                            suffix_ids.append(int(tok.eos_token_id))
                    if len(suffix_ids) < 24:
                        return None
                    if len(suffix_ids) > max_tok:
                        suffix_ids = suffix_ids[:max_tok]
                    return {
                        "id": row["id"],
                        "pillar": row.get("pillar", "unknown"),
                        "source_mode": args.rollout_mode,
                        "finish_reason": fr,
                        "prefix_ids": row["prefix_ids"],
                        "suffix_ids": suffix_ids,
                        "effective_tokens": len(suffix_ids),
                    }
                except Exception:
                    if att == 2:
                        return None
        finally:
            release(u)
        return None

    with args.output.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, pair) for pair in enumerate(todo)]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is not None:
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                kept += 1
                tot_tok += rec["effective_tokens"]
            else:
                dropped += 1
            if i % 200 == 0 or i == len(todo):
                dt = perf_counter() - t0
                print(
                    f"  [v4 rollout] {i}/{len(todo)} kept={kept} dropped={dropped} "
                    f"tok={tot_tok/1e6:.2f}M rate={tot_tok/max(dt,1e-6):.0f} tok/s elapsed={dt:.1f}s",
                    flush=True,
                )


def cmd_build(args) -> None:
    from inkling.prepare_data_v3 import _has_13gram_overlap, load_eval_13grams

    args.output.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    forbidden = load_eval_13grams(args.eval_dir, args.eval_raw, args.sec5_raw, tok=tok)
    if args.eval_exact_dir and args.eval_exact_dir.exists():
        forbidden |= load_eval_13grams(args.eval_exact_dir, args.eval_raw, args.sec5_raw, tok=tok)

    rows: list[dict] = []
    seen_ids: set[str] = set()
    leaks = 0
    for spec in args.inputs:
        # Format: "path" or "path:repeat" (e.g. upweight ultralong 2x if desired)
        if ":" in spec and spec.rsplit(":", 1)[1].isdigit():
            p_str, rep_s = spec.rsplit(":", 1)
            rep = int(rep_s)
        else:
            p_str, rep = spec, 1
        src = Path(p_str)
        src_rows = [json.loads(l) for l in src.open() if l.strip()]
        for r in src_rows:
            p_ids = r["prefix_ids"]
            s_ids = r["suffix_ids"]
            if len(p_ids) < 16 or len(s_ids) < 24:
                continue
            # Cap ultra-long suffix at 1536 so 52K-prefix rows stay comfortably <150GB VRAM on B300
            if len(p_ids) > 15000 and len(s_ids) > 1536:
                s_ids = s_ids[:1536]
            elif len(s_ids) > 3072:
                s_ids = s_ids[:3072]
            # Fast 13-gram audit on newly generated ultralong prompts (v3 prompts already audited, check anyway)
            if r.get("pillar", "").startswith("ultralong_"):
                tail_txt = tok.decode(p_ids[-1024:], skip_special_tokens=True)
                if _has_13gram_overlap(tail_txt, forbidden):
                    leaks += 1
                    continue
            for copy_idx in range(rep):
                rid = r["id"] if copy_idx == 0 else f"{r['id']}#c{copy_idx}"
                if rid in seen_ids:
                    continue
                seen_ids.add(rid)
                rows.append(
                    {
                        "id": rid,
                        "pillar": r.get("pillar", "unknown"),
                        "source_mode": r.get("source_mode", "teacher"),
                        "prefix_ids": p_ids,
                        "suffix_ids": s_ids,
                        "effective_tokens": len(s_ids),
                    }
                )

    assert leaks == 0, f"Detected {leaks} 13-gram eval leaks!"
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    out_jsonl = args.output / "train.jsonl"
    out_jsonl.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pillars = Counter(r["pillar"] for r in rows)
    tok_by_pillar = Counter()
    for r in rows:
        tok_by_pillar[r["pillar"]] += r["effective_tokens"]
    p_lens = sorted(len(r["prefix_ids"]) for r in rows)
    s_lens = sorted(len(r["suffix_ids"]) for r in rows)
    n_gt_8k = sum(1 for x in p_lens if x >= 8192)
    n_gt_16k = sum(1 for x in p_lens if x >= 16384)
    manifest = {
        "total_rows": len(rows),
        "total_kl_tokens": sum(r["effective_tokens"] for r in rows),
        "eval_13gram_leaks": leaks,
        "pillars_rows": dict(pillars),
        "pillars_tokens": dict(tok_by_pillar),
        "prefix_quantiles": {
            "min": p_lens[0],
            "p50": p_lens[len(p_lens) // 2],
            "p90": p_lens[int(len(p_lens) * 0.9)],
            "max": p_lens[-1],
            "frac_ge_8k": round(n_gt_8k / len(rows), 4),
            "frac_ge_16k": round(n_gt_16k / len(rows), 4),
        },
        "suffix_quantiles": {
            "min": s_lens[0],
            "p50": s_lens[len(s_lens) // 2],
            "p90": s_lens[int(len(s_lens) * 0.9)],
            "max": s_lens[-1],
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "READY").write_text("ok\n")
    print("[v4 build] DONE:", json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prompts")
    p1.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model_varloop"))
    p1.add_argument("--v2-raw", type=Path, default=Path("/dev/shm/inkling/data/v2_raw"))
    p1.add_argument("--prompts-v3", type=Path, default=Path("/dev/shm/inkling/data/train_inkling_v3/prompts_v3.jsonl"))
    p1.add_argument("--eval-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_sec5_high"))
    p1.add_argument("--eval-exact-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_official_exact"))
    p1.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    p1.add_argument("--sec5-raw", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    p1.add_argument("--output", type=Path, required=True)
    p1.add_argument("--n-ultralong-recall", type=int, default=800)
    p1.add_argument("--n-ultralong-graph", type=int, default=400)
    p1.add_argument("--n-gkd-reason", type=int, default=2000)
    p1.add_argument("--n-gkd-long", type=int, default=400)
    p1.add_argument("--n-vl-reason", type=int, default=1200)
    p1.add_argument("--n-vl-long", type=int, default=500)
    p1.add_argument("--n-vl-proc", type=int, default=450)
    p1.add_argument("--n-vl-tool", type=int, default=250)
    p1.add_argument("--seed", type=int, default=20261123)

    p_sft = sub.add_parser("sft_verified")
    p_sft.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model_varloop"))
    p_sft.add_argument("--v2-raw", type=Path, default=Path("/dev/shm/inkling/data/v2_raw"))
    p_sft.add_argument("--eval-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_sec5_high"))
    p_sft.add_argument("--eval-exact-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_official_exact"))
    p_sft.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    p_sft.add_argument("--sec5-raw", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    p_sft.add_argument("--output", type=Path, required=True)
    p_sft.add_argument("--n-opencode", type=int, default=3000)
    p_sft.add_argument("--n-openthoughts", type=int, default=3000)
    p_sft.add_argument("--n-longalign-ul", type=int, default=1200)
    p_sft.add_argument("--seed", type=int, default=20261123)

    p2 = sub.add_parser("rollout_vllm")
    p2.add_argument("--urls", nargs="+", required=True)
    p2.add_argument("--model-path", type=Path, required=True)
    p2.add_argument("--prompts", type=Path, required=True)
    p2.add_argument("--output", type=Path, required=True)
    p2.add_argument("--rollout-mode", choices=("teacher", "student_gkd"), required=True)
    p2.add_argument("--concurrency", type=int, default=48)
    p2.add_argument("--temperature", type=float, default=1.0)
    p2.add_argument("--top-p", type=float, default=0.95)
    p2.add_argument("--max-tokens-override", type=int, default=0)
    p2.add_argument("--limit", type=int, default=0)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--timeout", type=float, default=900.0)

    p3 = sub.add_parser("build")
    p3.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model_varloop"))
    p3.add_argument("--inputs", nargs="+", required=True)
    p3.add_argument("--eval-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_sec5_high"))
    p3.add_argument("--eval-exact-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_official_exact"))
    p3.add_argument("--eval-raw", type=Path, default=Path("/dev/shm/inkling/data/eval_raw"))
    p3.add_argument("--sec5-raw", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    p3.add_argument("--output", type=Path, required=True)
    p3.add_argument("--seed", type=int, default=20261123)

    args = parser.parse_args()
    if args.cmd == "prompts":
        cmd_prompts(args)
    elif args.cmd == "sft_verified":
        cmd_sft_verified(args)
    elif args.cmd == "rollout_vllm":
        cmd_rollout_vllm(args)
    elif args.cmd == "build":
        cmd_build(args)


if __name__ == "__main__":
    main()
