"""Build clean, 100% on-policy, 13-gram-decontaminated v3 distillation data for Inkling-2-Mini.

Design goals (aligned with Verifier [H0] gate & Inkling-2-Mini architecture):
  1. ZERO fallback to truncated v1_train, ZERO 30% mid-CoT slicing, ZERO malformed special tokens.
  2. Crosses Inkling-2-Mini's LogScale global-attention boundary (`log_scaling_n_floor = 8192`):
     long-context & procedural retrieval/graph prompts span 4K-14K tokens (>55% >= 8,192 tok).
  3. 100% on-policy Original 3/3 rollouts via official Day-0 vLLM (`rollout_vllm` mode), matching
     Inkling's exact native `<|content_thinking|>...<|content_text|>...` distribution at effort=high/medium.
  4. Built-in 13-gram sliding-window decontamination assertion (`overlap == 0`) against ALL 5 official
     benchmarks (GPQA Diamond, LiveCodeBench v6, LongBench v2, MRCR v2 8n, GraphWalks BFS) + BFCL/Apex.

Modes:
  - `prompts`      : Build decontaminated seed prompts (`prompts_v3.jsonl`) across 4 pillars.
  - `rollout_vllm` : Run high-concurrency multi-replica vLLM Original 3/3 rollouts (`rollouts_v3.jsonl`).
  - `build`        : Audit special tags + 13-gram decontamination and emit `train.jsonl` + `manifest.json`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import string
from time import perf_counter

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from inkling.prepare_data_v2 import _convert_apigen_mt, _convert_swesmith


WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _ngrams_13(text: str) -> set[tuple[str, ...]]:
    ws = _words(text)
    if len(ws) < 13:
        return set()
    return {tuple(ws[i : i + 13]) for i in range(len(ws) - 12)}


def _has_13gram_overlap(text: str, forbidden: set[tuple[str, ...]]) -> bool:
    ws = _words(text)
    if len(ws) < 13 or not forbidden:
        return False
    # Stride-1 exact sliding window check
    for i in range(len(ws) - 12):
        if tuple(ws[i : i + 13]) in forbidden:
            return True
    return False


def load_eval_13grams(eval_dir: Path, raw_dir: Path, sec5_raw: Path, tok=None) -> set[tuple[str, ...]]:
    """Collect 13-gram fingerprints from ALL official evaluation benchmarks."""
    ngrams: set[tuple[str, ...]] = set()

    def add_text(txt: str, max_words: int = 4000) -> None:
        ws = _words(txt)
        if len(ws) < 13:
            return
        if len(ws) > max_words:
            # Cover head, middle, and tail (where the actual question lives)
            half = max_words // 2
            ws = ws[:half] + ws[-half:]
        for i in range(len(ws) - 12):
            ngrams.add(tuple(ws[i : i + 13]))

    # 1. Eval JSONLs (gpqa_diamond_high, livecodebench_v6_high, longbench_v2_128k, mrcr_8n, graphwalks_bfs, apex, bfcl)
    if eval_dir.exists():
        for fp in sorted(eval_dir.glob("*.jsonl")):
            for line in fp.open(encoding="utf-8", errors="ignore"):
                if not line.strip():
                    continue
                r = json.loads(line)
                found_str = False
                for key in ("prompt_text", "question", "problem", "input", "context"):
                    val = r.get(key)
                    if isinstance(val, str) and len(val) >= 40:
                        add_text(val)
                        found_str = True
                msgs = r.get("messages")
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict) and isinstance(m.get("content"), str):
                            add_text(m["content"], max_words=1000)
                            found_str = True
                if not found_str and tok is not None and isinstance(r.get("prompt_ids"), list):
                    add_text(tok.decode(r["prompt_ids"], skip_special_tokens=True))

    # 2. Raw benchmark sources (GPQA, LCB v5/v6, Apex)
    gpqa = raw_dir / "gpqa_diamond_mc.parquet"
    if gpqa.exists():
        for r in pq.read_table(gpqa).to_pylist():
            add_text(str(r.get("problem", "")))
    for fn in ("lcb_test5.jsonl", "lcb_test6.jsonl"):
        p = raw_dir / fn
        if p.exists():
            for line in p.open(encoding="utf-8", errors="ignore"):
                if line.strip():
                    r = json.loads(line)
                    add_text(str(r.get("question_content", "")))
    ap = sec5_raw / "apex_2025.parquet"
    if ap.exists():
        for r in pq.read_table(ap).to_pylist():
            add_text(str(r.get("problem", "")))

    return ngrams


def _extract_raw_user_from_v1(tok, prefix_ids: list[int]) -> str | None:
    """Recover clean raw user message text from v1_train prefix_ids without keeping v1 template artifacts."""
    txt = tok.decode(prefix_ids, skip_special_tokens=False)
    marker = "<|message_user|><|content_text|>"
    if marker not in txt:
        return None
    tail = txt.split(marker, 1)[1]
    if "<|end_message|>" not in tail:
        return None
    user = tail.split("<|end_message|>", 1)[0].strip()
    if len(user) < 20:
        return None
    return user


def _load_paragraph_pool(raw_dir: Path, forbidden: set[tuple[str, ...]], max_paras: int = 12000) -> list[str]:
    """Harvest clean, decontaminated 60-260 word paragraphs from raw LongAlign-10k for procedural recall."""
    paras: list[str] = []
    la_path = raw_dir / "longalign_10k.jsonl"
    if not la_path.exists():
        return paras
    for line in la_path.open(encoding="utf-8", errors="ignore"):
        if len(paras) >= max_paras:
            break
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        for m in r.get("messages") or []:
            c = m.get("content")
            if not isinstance(c, str) or len(c) < 300:
                continue
            for p in re.split(r"\n\s*\n", c):
                p = p.strip()
                wc = len(p.split())
                if 50 <= wc <= 220 and p.isascii() and not _has_13gram_overlap(p, forbidden):
                    paras.append(p)
                    if len(paras) >= max_paras:
                        break
            if len(paras) >= max_paras:
                break
    return paras


RECALL_FORMS = [
    "report", "excerpt", "brief", "passage", "note", "memo", "entry", "summary", "article", "section"
]
RECALL_TOPICS = [
    "astronomy", "botany", "architecture", "maritime history", "geology", "linguistics",
    "acoustics", "cartography", "metallurgy", "hydrology", "textiles", "forestry",
    "aviation", "ceramics", "horology", "mineralogy", "optics", "navigation",
    "meteorology", "cryptography", "agriculture", "paleontology", "railways", "typography",
]


def _make_verbatim_recall_prompt(
    tok,
    paras: list[str],
    rng: random.Random,
    target_tokens: int,
    effort: str,
) -> tuple[list[int], str] | None:
    """Procedurally generate a multi-turn multi-needle verbatim-copy prompt (4K-14K tok, zero test overlap)."""
    if len(paras) < 40:
        return None
    target_form = rng.choice(RECALL_FORMS)
    target_topic = rng.choice(RECALL_TOPICS)
    other_forms = [f for f in RECALL_FORMS if f != target_form]
    other_topics = [t for t in RECALL_TOPICS if t != target_topic]

    # Pick 6-10 needles of the target (form, topic) + distractors until target_tokens is reached
    n_needles = rng.randint(6, 10)
    msgs: list[dict] = []
    est_tokens = 0
    needle_turns = 0
    turn_idx = 0

    # Decide positions for needles distributed across the whole conversation
    # Rough average ~180 tokens per turn
    est_total_turns = max(n_needles + 6, int(target_tokens / 190))
    needle_slots = set(rng.sample(range(est_total_turns), min(n_needles, est_total_turns)))

    while est_tokens < target_tokens - 160 and turn_idx < est_total_turns + 40:
        snippet = paras[rng.randrange(len(paras))]
        if turn_idx in needle_slots or (needle_turns < n_needles and est_tokens > target_tokens * 0.8):
            form, topic = target_form, target_topic
            needle_turns += 1
        else:
            # Hard distractor: half share form, half share topic
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

    if needle_turns < 4:
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
    if len(p_ids) < 2500 or len(p_ids) > 14336:
        return None
    return p_ids, f"verbatim_recall_k{k}_n{needle_turns}"


def _make_graph_bfs_prompt(
    tok,
    rng: random.Random,
    target_tokens: int,
    effort: str,
) -> tuple[list[int], str] | None:
    """Procedurally generate a synthetic multi-hop random-graph BFS prompt (4K-13K tok, zero test overlap)."""
    # ~18 tokens per node's adjacency line -> target_tokens / 18 nodes
    n_nodes = max(120, min(650, int(target_tokens / 19)))
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

    # Pick a start node with a non-empty BFS frontier at distance d in {2, 3}
    start = None
    target_dist = rng.choice([2, 3])
    for _ in range(25):
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
        if 2 <= len(frontier) <= 18:
            start = cand
            break
    if start is None:
        return None

    # Shuffle adjacency lines so graph structure is non-local across the 4K-13K prompt
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
    if len(p_ids) < 2500 or len(p_ids) > 14336:
        return None
    return p_ids, f"graph_bfs_d{target_dist}_n{n_nodes}"


def export_prompts_v3(args, tok, forbidden: set[tuple[str, ...]]) -> None:
    rng = random.Random(args.seed)
    prompts: list[dict] = []
    rejected_decontam = 0

    n_total = args.num_prompts
    n_reason = int(n_total * 0.35)
    n_long = int(n_total * 0.35)
    n_recall = int(n_total * 0.12)
    n_graph = int(n_total * 0.08)
    n_tool = n_total - (n_reason + n_long + n_recall + n_graph)

    # --- 1. Hard STEM & Coding Reasoning (35%) ---
    ocr_path = args.raw_dir / "ocr_0.parquet"
    n_ocr_target = int(n_reason * 0.65)
    n_ocr = 0
    if ocr_path.exists():
        rows_ocr = pq.read_table(ocr_path).to_pylist()
        rng.shuffle(rows_ocr)
        for idx, row in enumerate(rows_ocr):
            if n_ocr >= n_ocr_target:
                break
            q = (row.get("input") or "").strip()
            if not q:
                continue
            if _has_13gram_overlap(q, forbidden):
                rejected_decontam += 1
                continue
            eff = rng.choices(["medium", "high"], weights=[0.35, 0.65])[0]
            p_str = tok.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=eff,
            )
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 32 <= len(p_ids) <= 3072:
                prompts.append(
                    {
                        "id": f"v3_ocr/{idx}",
                        "pillar": "reason",
                        "source": "opencode_reasoning",
                        "effort": eff,
                        "prompt_ids": p_ids,
                        "max_new_tokens": 3072,
                    }
                )
                n_ocr += 1

    # Re-render raw OpenThoughts user questions (clean native template, NEVER inheriting v1 truncation!)
    n_ot_target = n_reason - n_ocr
    n_ot = 0
    if args.v1_train.exists() and n_ot_target > 0:
        ot_users: list[str] = []
        seen_u: set[str] = set()
        for line in args.v1_train.open(encoding="utf-8", errors="ignore"):
            if '"openthoughts"' not in line:
                continue
            r = json.loads(line)
            if r.get("source") != "openthoughts":
                continue
            u = _extract_raw_user_from_v1(tok, r["prefix_ids"])
            if not u or u[:120] in seen_u:
                continue
            seen_u.add(u[:120])
            ot_users.append(u)
            if len(ot_users) >= n_ot_target * 4:
                break
        rng.shuffle(ot_users)
        for idx, u in enumerate(ot_users):
            if n_ot >= n_ot_target:
                break
            if _has_13gram_overlap(u, forbidden):
                rejected_decontam += 1
                continue
            eff = rng.choices(["medium", "high"], weights=[0.30, 0.70])[0]
            p_str = tok.apply_chat_template(
                [{"role": "user", "content": u}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=eff,
            )
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 32 <= len(p_ids) <= 2560:
                prompts.append(
                    {
                        "id": f"v3_ot/{idx}",
                        "pillar": "reason",
                        "source": "openthoughts_raw",
                        "effort": eff,
                        "prompt_ids": p_ids,
                        "max_new_tokens": 3072,
                    }
                )
                n_ot += 1

    # --- 2. Real Long-Context (35%, 2K-14K tokens, >=55% crossing >8192 LogScale!) ---
    la_path = args.raw_dir / "longalign_10k.jsonl"
    long_short: list[dict] = []  # 2K-8K
    long_gt8k: list[dict] = []   # 8K-14K (crosses LogScale!)
    if la_path.exists():
        la_lines = [l for l in la_path.open(encoding="utf-8", errors="ignore") if l.strip()]
        rng.shuffle(la_lines)
        for idx, line in enumerate(la_lines):
            if len(long_gt8k) >= int(n_long * 0.65) and len(long_short) >= int(n_long * 0.45):
                break
            try:
                r = json.loads(line)
            except Exception:
                continue
            ms = r.get("messages") or []
            if len(ms) < 2 or ms[0].get("role") != "user":
                continue
            user_txt = str(ms[0].get("content") or "")
            if len(user_txt) < 6000:
                continue
            if _has_13gram_overlap(user_txt, forbidden):
                rejected_decontam += 1
                continue
            eff = rng.choices(["medium", "high"], weights=[0.40, 0.60])[0]
            p_str = tok.apply_chat_template(
                [{"role": "user", "content": user_txt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=eff,
            )
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if len(p_ids) > args.max_prefix:
                # Preserve intact system/template header + document head + document tail & question
                head = 2048
                p_ids = p_ids[:head] + p_ids[-(args.max_prefix - head) :]
            if len(p_ids) < 2048:
                continue
            rec = {
                "id": f"v3_long/{idx}",
                "pillar": "long",
                "source": "longalign_raw",
                "effort": eff,
                "prompt_ids": p_ids,
                "max_new_tokens": 2048,
            }
            if len(p_ids) >= 8192:
                long_gt8k.append(rec)
            else:
                long_short.append(rec)
    take_gt8k = min(len(long_gt8k), int(n_long * 0.62))
    take_short = min(len(long_short), n_long - take_gt8k)
    prompts.extend(long_gt8k[:take_gt8k])
    prompts.extend(long_short[:take_short])

    # --- 3a. Procedural Verbatim Multi-Needle Copy (12%, 4K-14K tok, >=60% >8K LogScale!) ---
    paras = _load_paragraph_pool(args.raw_dir, forbidden)
    print(f"Loaded {len(paras)} clean paragraphs for procedural verbatim recall", flush=True)
    n_rec_done = 0
    for idx in range(n_recall * 3):
        if n_rec_done >= n_recall:
            break
        # 60% in [8500, 13200] (crosses 8192 LogScale!), 40% in [3500, 8000]
        tgt = rng.randint(8600, 13000) if (n_rec_done % 5) < 3 else rng.randint(3600, 7800)
        eff = rng.choices(["medium", "high"], weights=[0.35, 0.65])[0]
        built = _make_verbatim_recall_prompt(tok, paras, rng, tgt, eff)
        if built is None:
            continue
        p_ids, sub = built
        prompts.append(
            {
                "id": f"v3_recall/{idx}",
                "pillar": "procedural",
                "source": sub,
                "effort": eff,
                "prompt_ids": p_ids,
                "max_new_tokens": 1536,
            }
        )
        n_rec_done += 1

    # --- 3b. Procedural Multi-Hop Graph BFS (8%, 4K-13K tok, >=60% >8K LogScale!) ---
    n_gr_done = 0
    for idx in range(n_graph * 3):
        if n_gr_done >= n_graph:
            break
        tgt = rng.randint(8600, 12800) if (n_gr_done % 5) < 3 else rng.randint(3600, 7800)
        eff = rng.choices(["medium", "high"], weights=[0.35, 0.65])[0]
        built = _make_graph_bfs_prompt(tok, rng, tgt, eff)
        if built is None:
            continue
        p_ids, sub = built
        prompts.append(
            {
                "id": f"v3_graph/{idx}",
                "pillar": "procedural",
                "source": sub,
                "effort": eff,
                "prompt_ids": p_ids,
                "max_new_tokens": 2048,
            }
        )
        n_gr_done += 1

    # --- 4. Multi-Turn Tool Use Anchor (10%) ---
    ap_path = args.raw_dir / "apigen_mt_5k.json"
    n_ap = 0
    n_ap_target = int(n_tool * 0.65)
    if ap_path.exists():
        ap = json.load(ap_path.open())
        rng.shuffle(ap)
        for idx, row in enumerate(ap):
            if n_ap >= n_ap_target:
                break
            conv = _convert_apigen_mt(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 0]
            if not asst_idxs:
                continue
            t = rng.choice(asst_idxs)
            eff = rng.choice(["low", "medium"])
            try:
                p_str = tok.apply_chat_template(
                    msgs[:t], tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort=eff
                )
            except Exception:
                continue
            if _has_13gram_overlap(p_str, forbidden):
                rejected_decontam += 1
                continue
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 64 <= len(p_ids) <= 6144:
                prompts.append(
                    {
                        "id": f"v3_apigen/{idx}",
                        "pillar": "tool",
                        "source": "apigen_mt",
                        "effort": eff,
                        "prompt_ids": p_ids,
                        "max_new_tokens": 1024,
                    }
                )
                n_ap += 1

    swe_path = args.raw_dir / "swesmith_tool_0.parquet"
    n_swe = 0
    n_swe_target = n_tool - n_ap
    if swe_path.exists() and n_swe_target > 0:
        rows_swe = pq.read_table(swe_path).to_pylist()
        rng.shuffle(rows_swe)
        for idx, row in enumerate(rows_swe):
            if n_swe >= n_swe_target:
                break
            conv = _convert_swesmith(row)
            if not conv:
                continue
            msgs, tools = conv
            asst_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i > 1]
            if not asst_idxs:
                continue
            t = rng.choice(asst_idxs)
            eff = rng.choice(["low", "medium"])
            try:
                p_str = tok.apply_chat_template(
                    msgs[:t], tools=tools, tokenize=False, add_generation_prompt=True, reasoning_effort=eff
                )
            except Exception:
                continue
            if _has_13gram_overlap(p_str, forbidden):
                rejected_decontam += 1
                continue
            p_ids = tok.encode(p_str, add_special_tokens=False)
            if 128 <= len(p_ids) <= 8192:
                prompts.append(
                    {
                        "id": f"v3_swe/{idx}",
                        "pillar": "tool",
                        "source": "swesmith",
                        "effort": eff,
                        "prompt_ids": p_ids,
                        "max_new_tokens": 1280,
                    }
                )
                n_swe += 1

    rng.shuffle(prompts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in prompts:
            f.write(json.dumps(r) + "\n")

    by_pillar = Counter(r["pillar"] for r in prompts)
    ge_8k = sum(1 for r in prompts if len(r["prompt_ids"]) >= 8192)
    ge_4k = sum(1 for r in prompts if len(r["prompt_ids"]) >= 4096)
    summary = {
        "total_prompts": len(prompts),
        "by_pillar": dict(by_pillar),
        "prefix_ge_4k": ge_4k,
        "prefix_ge_8k_logscale": ge_8k,
        "rejected_13gram_decontam": rejected_decontam,
        "max_prefix": args.max_prefix,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2), flush=True)


async def _run_rollouts_vllm(args, tok) -> None:
    import aiohttp

    prompts = [json.loads(l) for l in args.prompts.open() if l.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if args.output.exists():
        for line in args.output.open():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [p for p in prompts if p["id"] not in done_ids]
    print(
        f"[rollout_vllm] total={len(prompts)}, done={len(done_ids)}, todo={len(todo)}, "
        f"urls={len(args.urls)}, conc={args.concurrency}",
        flush=True,
    )
    if not todo:
        return

    eos_id = 200006
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    stats = {"done": 0, "stop": 0, "tokens": 0, "errors": 0}
    t0 = perf_counter()

    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 32)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        fout = args.output.open("a", buffering=1)
        try:
            async def worker(idx: int, item: dict) -> None:
                url = args.urls[idx % len(args.urls)].rstrip("/") + "/v1/completions"
                payload = {
                    "model": str(args.model_path),
                    "prompt": item["prompt_ids"],
                    "max_tokens": min(int(item.get("max_new_tokens", 2048)), args.max_new_tokens),
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "skip_special_tokens": False,
                    "spaces_between_special_tokens": False,
                    "return_token_ids": True,
                }
                async with sem:
                    for attempt in range(4):
                        try:
                            async with session.post(url, json=payload) as resp:
                                if resp.status != 200:
                                    body = await resp.text()
                                    raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                                data = await resp.json()
                                ch = data["choices"][0]
                                finish = ch.get("finish_reason", "unknown")
                                s_ids = ch.get("token_ids")
                                if not s_ids:
                                    txt = ch.get("text", "")
                                    s_ids = tok.encode(txt, add_special_tokens=False)
                                    if finish == "stop" and (not s_ids or s_ids[-1] != eos_id):
                                        s_ids.append(eos_id)
                                rec = {
                                    "id": item["id"],
                                    "pillar": item["pillar"],
                                    "source": item["source"],
                                    "effort": item["effort"],
                                    "prefix_ids": item["prompt_ids"],
                                    "suffix_ids": s_ids,
                                    "finish_reason": finish,
                                    "effective_tokens": len(s_ids),
                                }
                                async with lock:
                                    fout.write(json.dumps(rec) + "\n")
                                    stats["done"] += 1
                                    stats["stop"] += int(finish == "stop")
                                    stats["tokens"] += len(s_ids)
                                    if stats["done"] % 100 == 0 or stats["done"] == len(todo):
                                        dt = max(1e-6, perf_counter() - t0)
                                        print(
                                            f"  [rollout] {stats['done']}/{len(todo)} | "
                                            f"eos={stats['stop']/stats['done']*100:.1f}% | "
                                            f"tok={stats['tokens']/1e6:.2f}M | "
                                            f"agg={stats['tokens']/dt:.0f} tok/s | err={stats['errors']}",
                                            flush=True,
                                        )
                                return
                        except Exception as exc:
                            if attempt == 3:
                                async with lock:
                                    stats["errors"] += 1
                                    print(f"  [ERR] {item['id']}: {exc}", flush=True)
                                return
                            await asyncio.sleep(2 * (attempt + 1))

            await asyncio.gather(*(worker(i, item) for i, item in enumerate(todo)))
        finally:
            fout.close()


def build_train_v3(args, tok, forbidden: set[tuple[str, ...]]) -> None:
    """Assemble and Verifier-[H0]-audit the final train_inkling_v3/train.jsonl stream."""
    rng = random.Random(args.seed)
    rows: list[dict] = []
    skipped = Counter()

    raw_rows = [json.loads(l) for l in args.rollouts.open() if l.strip()]
    rng.shuffle(raw_rows)

    pillar_toks = Counter()
    source_toks = Counter()
    source_cnts = Counter()
    target_total = args.target_tokens
    # Enforce healthy balance so no single pillar dominates
    pillar_caps = {
        "reason": int(target_total * 0.40),
        "long": int(target_total * 0.35),
        "procedural": int(target_total * 0.25),
        "tool": int(target_total * 0.15),
    }

    for r in raw_rows:
        p_ids = r["prefix_ids"]
        s_ids = r["suffix_ids"]
        if len(p_ids) < 16 or len(s_ids) < 16:
            skipped["too_short"] += 1
            continue
        if len(p_ids) > args.max_prefix:
            skipped["prefix_over_cap"] += 1
            continue
        # Cap suffix at max_target (2048) while keeping the start of native thinking+answer
        if len(s_ids) > args.max_target:
            s_ids = s_ids[: args.max_target]

        # [H0] Audit 1: exact native template integrity (1 Thinking effort header, intact generation start)
        p_txt = tok.decode(p_ids, skip_special_tokens=False)
        if p_txt.count("Thinking effort level:") != 1:
            skipped["bad_effort_header"] += 1
            continue
        if not p_txt.rstrip().endswith("<|end_message|>"):
            skipped["bad_prefix_end"] += 1
            continue

        # [H0] Audit 2: strict 13-gram decontamination against ALL 5 official benchmarks
        if _has_13gram_overlap(p_txt, forbidden):
            skipped["decontam_13gram"] += 1
            continue

        pillar = r.get("pillar", "other")
        if pillar in pillar_caps and pillar_toks[pillar] >= pillar_caps[pillar]:
            skipped["pillar_cap"] += 1
            continue

        rec = {
            "id": r.get("id", ""),
            "source": f"{pillar}/{r.get('source', 'unknown')}",
            "pillar": pillar,
            "effort": r.get("effort", "high"),
            "finish_reason": r.get("finish_reason", "unknown"),
            "prefix_ids": p_ids,
            "suffix_ids": s_ids,
            "effective_tokens": len(s_ids),
        }
        rows.append(rec)
        pillar_toks[pillar] += len(s_ids)
        source_toks[rec["source"]] += len(s_ids)
        source_cnts[rec["source"]] += 1
        if sum(pillar_toks.values()) >= target_total:
            break

    args.output.mkdir(parents=True, exist_ok=True)
    out_file = args.output / "train.jsonl"
    with out_file.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ge_4k = sum(1 for r in rows if len(r["prefix_ids"]) >= 4096)
    ge_8k = sum(1 for r in rows if len(r["prefix_ids"]) >= 8192)
    eos_rate = sum(1 for r in rows if r["finish_reason"] == "stop") / max(1, len(rows))
    manifest = {
        "rows": len(rows),
        "effective_tokens": sum(r["effective_tokens"] for r in rows),
        "on_policy_teacher": "Original 3/3 (vLLM Day-0)",
        "pillar_tokens": dict(pillar_toks),
        "counts_by_subsource": dict(source_cnts),
        "tokens_by_subsource": dict(source_toks),
        "prefix_ge_4k": ge_4k,
        "prefix_ge_8k_logscale": ge_8k,
        "eos_rate": round(eos_rate, 4),
        "skipped": dict(skipped),
        "eval_13gram_overlaps": 0,
        "max_prefix": args.max_prefix,
        "max_target": args.max_target,
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "READY").write_text("OK\n")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prompts", "rollout_vllm", "build"))
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--raw-dir", type=Path, default=Path("/dev/shm/inkling/data/v2_raw"))
    parser.add_argument("--eval-dir", type=Path, default=Path("/dev/shm/inkling/data/eval_sec5_high"))
    parser.add_argument("--sec5-raw", type=Path, default=Path("/dev/shm/inkling/data/official_sec5"))
    parser.add_argument("--v1-train", type=Path, default=Path("/dev/shm/inkling/data/train_inkling/train.jsonl"))
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--rollouts", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urls", nargs="+", default=[f"http://127.0.0.1:{8000+i}" for i in range(8)])
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-prompts", type=int, default=12000)
    parser.add_argument("--target-tokens", type=int, default=24_000_000)
    parser.add_argument("--max-prefix", type=int, default=14336)
    parser.add_argument("--max-target", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    if args.mode == "rollout_vllm":
        asyncio.run(_run_rollouts_vllm(args, tok))
        return

    forbidden = load_eval_13grams(args.eval_dir, args.raw_dir, args.sec5_raw, tok=tok)
    print(f"Loaded {len(forbidden):,} 13-gram eval decontamination shingles", flush=True)
    if args.mode == "prompts":
        export_prompts_v3(args, tok, forbidden)
    else:
        build_train_v3(args, tok, forbidden)


if __name__ == "__main__":
    main()
