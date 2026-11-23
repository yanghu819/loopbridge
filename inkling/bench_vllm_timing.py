"""Rigorous, unconfounded Official Day-0 vLLM Blackwell serving benchmark for Inkling-2-Mini.

Measures TTFT (including prefill + LoopBridge KV-map fold), pure decode throughput (tok/s/user
and aggregate tok/s), and end-to-end wall latency across (L, B, gen_len) regimes using streaming
SSE requests with distinct per-request prompts (eliminating automatic prefix caching confounds).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
from time import perf_counter

import aiohttp
from transformers import AutoTokenizer


def _build_distinct_prompts(tok, source_jsonl: Path | None, L: int, count: int, seed: int) -> list[list[int]]:
    """Build `count` distinct realistic prose token prompts of exact length `L` (no prefix overlap)."""
    rng = random.Random(seed + L * 31)
    pool: list[int] = []
    if source_jsonl and source_jsonl.exists():
        for line in source_jsonl.open(encoding="utf-8", errors="ignore"):
            if not line.strip():
                continue
            r = json.loads(line)
            ids = r.get("prompt_ids") or r.get("prefix_ids")
            if isinstance(ids, list) and len(ids) > 256:
                pool.extend(ids[64:-32])
            if len(pool) >= L * (count + 4):
                break
    if len(pool) < L * 2:
        # Fallback to diverse natural English prose chunks
        base_text = (
            "The structure of distributed inference systems depends on memory bandwidth, attention "
            "cache locality, and layer depth scheduling across prefill and decode phases. "
            "Experimental measurements across diverse reasoning, coding, and retrieval tasks show "
            "distinct compute-to-memory trade-offs as context length scales beyond sliding windows. "
        )
        base_ids = tok.encode(base_text, add_special_tokens=False)
        pool = base_ids * ((L * (count + 2)) // max(1, len(base_ids)) + 2)

    prompts: list[list[int]] = []
    header = tok.encode(
        "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>",
        add_special_tokens=False,
    )
    footer = tok.encode("<|end_message|>", add_special_tokens=False)
    body_len = max(16, L - len(header) - len(footer) - 8)

    for i in range(count):
        # Distinct nonce prefix per request defeats any prefix caching even if enabled
        nonce = tok.encode(f"[REQ-{seed}-{L}-{i}-{rng.randint(100000, 999999)}] ", add_special_tokens=False)
        start = rng.randint(0, max(0, len(pool) - body_len - 1))
        body = pool[start : start + body_len]
        p = (header + nonce + body)[: L - len(footer)] + footer
        prompts.append(p)
    return prompts


async def _stream_one(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt_ids: list[int],
    gen_len: int,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": gen_len,
        "min_tokens": gen_len,
        "ignore_eos": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": True,
    }
    t0 = perf_counter()
    t_first: float | None = None
    chunks = 0
    async with session.post(url, json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:300]}")
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                ev = json.loads(data_str)
            except Exception:
                continue
            choices = ev.get("choices") or []
            if choices and (choices[0].get("text") or choices[0].get("delta")):
                if t_first is None:
                    t_first = perf_counter()
                chunks += 1
    t_end = perf_counter()
    if t_first is None:
        t_first = t_end
    ttft_s = t_first - t0
    dec_s = max(1e-6, t_end - t_first)
    e2e_s = t_end - t0
    return {
        "ttft_ms": ttft_s * 1000.0,
        "t_first_abs": t_first,
        "t_end_abs": t_end,
        "decode_s": dec_s,
        "e2e_s": e2e_s,
        "decode_tok_s": max(1, gen_len - 1) / dec_s,
        "e2e_tok_s": gen_len / max(1e-6, e2e_s),
        "chunks": chunks,
    }


async def run_regime(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompts: list[list[int]],
    gen_len: int,
    warmup: bool = False,
) -> dict:
    t0 = perf_counter()
    results = await asyncio.gather(
        *(_stream_one(session, url, model, p, gen_len) for p in prompts)
    )
    batch_wall_s = perf_counter() - t0
    B = len(prompts)
    ttfts = sorted(r["ttft_ms"] for r in results)
    mean_ttft_ms = sum(ttfts) / B
    p95_ttft_ms = ttfts[min(B - 1, int(0.95 * B))]
    # True uncontaminated batch decode window: starts ONLY after ALL B requests finish prefill (max t_first_abs)
    # so chunked-prefill waves from later requests never bias mixed-scheduler baselines like Original 3/3 or Distilled.
    max_t_first = max(r["t_first_abs"] for r in results)
    max_t_end = max(r["t_end_abs"] for r in results)
    pure_dec_wall_s = max(1e-6, max_t_end - max_t_first)
    agg_pure_dec_tps = B * max(1, gen_len - 1) / pure_dec_wall_s
    return {
        "B": B,
        "gen_len": gen_len,
        "batch_wall_s": round(batch_wall_s, 4),
        "pure_decode_wall_s": round(pure_dec_wall_s, 4),
        "ttft_mean_ms": round(mean_ttft_ms, 2),
        "ttft_p95_ms": round(p95_ttft_ms, 2),
        "per_user_decode_tok_s": round(agg_pure_dec_tps / B, 2),
        "agg_decode_tok_s": round(agg_pure_dec_tps, 2),
        "agg_e2e_tok_s": round(B * gen_len / max(1e-6, batch_wall_s), 2),
        "warmup": warmup,
    }


async def main_async(args) -> None:
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    url = args.url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=256)

    # Parse regimes: list of "L:B:gen_len"
    regimes = []
    for spec in args.regimes.split(","):
        L_s, B_s, G_s = spec.strip().split(":")
        regimes.append((int(L_s), int(B_s), int(G_s)))

    out_rows = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warmup once on shortest regime
        w_prompts = _build_distinct_prompts(tok, args.prompt_source, 512, 2, seed=args.seed + 999)
        await run_regime(session, url, str(args.model), w_prompts, 32, warmup=True)

        for idx, (L, B, gen_len) in enumerate(regimes):
            trials = []
            for rep in range(args.repeats):
                prompts = _build_distinct_prompts(tok, args.prompt_source, L, B, seed=args.seed + idx * 100 + rep)
                res = await run_regime(session, url, str(args.model), prompts, gen_len)
                trials.append(res)
            # Take median by batch_wall_s across repeats
            trials.sort(key=lambda r: r["batch_wall_s"])
            med = dict(trials[len(trials) // 2])
            med.update({"label": args.label, "L": L, "repeats": args.repeats})
            out_rows.append(med)
            print("BENCH_VLLM", json.dumps(med), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"label": args.label, "results": out_rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--prompt-source",
        type=Path,
        default=Path("/dev/shm/inkling/data/eval_sec5_high/longbench_v2_128k.jsonl"),
    )
    parser.add_argument(
        "--regimes",
        default="512:1:256,512:8:256,512:16:256,512:32:256,512:1:1024,512:16:1024,4096:1:256,4096:8:256,4096:16:256,16384:1:256,16384:8:256",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
