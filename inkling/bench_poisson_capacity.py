"""Benchmark real-world staggered Poisson arrivals (P50/P95 TTFT & P50/P95/P99 ITL)
and exact KV-cache memory utilization / max concurrent capacity at 64K and 256K
on an isolated Day-0 vLLM Blackwell server (`inkling.vllm_loopbridge serve`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import re
from time import perf_counter

import aiohttp
from transformers import AutoTokenizer

from inkling.bench_vllm_timing import _build_distinct_prompts


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


async def _stream_req(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt_ids: list[int],
    gen_len: int,
    delay_s: float,
) -> dict:
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": gen_len,
        "min_tokens": gen_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "skip_special_tokens": False,
    }
    t0 = perf_counter()
    t_first: float | None = None
    ts: list[float] = []
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            now = perf_counter()
            if t_first is None:
                t_first = now
            ts.append(now)
    t_end = perf_counter()
    if t_first is None:
        t_first = t_end
    itls_ms = [(ts[i] - ts[i - 1]) * 1000.0 for i in range(1, len(ts))]
    return {
        "ttft_ms": (t_first - t0) * 1000.0,
        "wall_s": t_end - t0,
        "chunks": len(ts),
        "itls_ms": itls_ms,
        "t0": t0,
        "t_end": t_end,
    }


async def _fetch_kv_metrics(session: aiohttp.ClientSession, base_url: str) -> dict:
    try:
        async with session.get(base_url.rstrip("/") + "/metrics") as resp:
            txt = await resp.text()
    except Exception as e:
        return {"error": str(e)}
    out = {}
    for pat, key in [
        (r"vllm:kv_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)", "kv_usage_perc"),
        (r"vllm:gpu_cache_usage_perc\{[^}]*\}\s+([0-9.eE+-]+)", "kv_usage_perc"),
        (r"vllm:num_requests_running\{[^}]*\}\s+([0-9.eE+-]+)", "num_running"),
        (r"vllm:num_requests_waiting\{[^}]*\}\s+([0-9.eE+-]+)", "num_waiting"),
        (r"vllm:num_preemptions_total\{[^}]*\}\s+([0-9.eE+-]+)", "preemptions"),
    ]:
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    return out


async def run_poisson(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    tok,
    prompt_source: Path,
    L: int,
    gen_len: int,
    n_reqs: int,
    rate_rps: float,
    seed: int,
) -> dict:
    url = base_url.rstrip("/") + "/v1/completions"
    prompts = _build_distinct_prompts(tok, prompt_source, L, n_reqs, seed=seed)
    rng = random.Random(seed)
    delays = []
    t_acc = 0.0
    for i in range(n_reqs):
        if i > 0:
            t_acc += rng.expovariate(rate_rps)
        delays.append(t_acc)

    t_start = perf_counter()
    results = await asyncio.gather(
        *[_stream_req(session, url, model, p, gen_len, d) for p, d in zip(prompts, delays)]
    )
    total_wall = perf_counter() - t_start
    ttfts = [r["ttft_ms"] for r in results]
    all_itls = [x for r in results for x in r["itls_ms"]]
    max_itls = [max(r["itls_ms"]) if r["itls_ms"] else 0.0 for r in results]
    return {
        "kind": "poisson",
        "L": L,
        "gen_len": gen_len,
        "n_reqs": n_reqs,
        "rate_rps": rate_rps,
        "total_wall_s": round(total_wall, 3),
        "agg_e2e_tok_s": round(n_reqs * gen_len / max(1e-6, total_wall), 2),
        "ttft_p50_ms": round(_pct(ttfts, 0.50), 2),
        "ttft_p95_ms": round(_pct(ttfts, 0.95), 2),
        "itl_p50_ms": round(_pct(all_itls, 0.50), 2),
        "itl_p95_ms": round(_pct(all_itls, 0.95), 2),
        "itl_p99_ms": round(_pct(all_itls, 0.99), 2),
        "req_max_itl_p50_ms": round(_pct(max_itls, 0.50), 2),
        "req_max_itl_p95_ms": round(_pct(max_itls, 0.95), 2),
    }


async def run_kv_probe(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    tok,
    prompt_source: Path,
    L: int,
    B: int,
    seed: int,
) -> dict:
    """Hold B concurrent sequences of length L active and read exact KV cache usage from /metrics."""
    url = base_url.rstrip("/") + "/v1/completions"
    prompts = _build_distinct_prompts(tok, prompt_source, L, B, seed=seed)
    pre_m = await _fetch_kv_metrics(session, base_url)
    peak_kv = 0.0
    max_running = 0
    done_flag = False

    async def poll():
        nonlocal peak_kv, max_running
        while not done_flag:
            m = await _fetch_kv_metrics(session, base_url)
            peak_kv = max(peak_kv, m.get("kv_usage_perc", 0.0))
            max_running = max(max_running, int(m.get("num_running", 0)))
            await asyncio.sleep(0.25)

    poller = asyncio.create_task(poll())
    try:
        await asyncio.gather(*[_stream_req(session, url, model, p, 96, 0.0) for p in prompts])
    finally:
        done_flag = True
        await poller
    post_m = await _fetch_kv_metrics(session, base_url)
    est_max_B = round(B / peak_kv, 1) if peak_kv > 1e-4 else None
    return {
        "kind": "kv_capacity",
        "L": L,
        "probe_B": B,
        "peak_kv_usage_frac": round(peak_kv, 5),
        "est_max_concurrent_seqs": est_max_B,
        "max_running_observed": max_running,
        "preemptions_delta": round(post_m.get("preemptions", 0.0) - pre_m.get("preemptions", 0.0), 1),
    }


async def main_async(args) -> None:
    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=256)
    out_rows = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warmup
        w_p = _build_distinct_prompts(tok, args.prompt_source, 512, 2, seed=999)
        await asyncio.gather(*[_stream_req(session, args.url.rstrip("/") + "/v1/completions", str(args.model), p, 32, 0.0) for p in w_p])

        specs = []
        for item in args.poisson_specs.split(","):
            item = item.strip()
            if not item:
                continue
            L_s, G_s, N_s, rps_s = item.split(":")
            specs.append((int(L_s), int(G_s), int(N_s), float(rps_s)))

        for L, G, N, rps in specs:
            res = await run_poisson(session, args.url, str(args.model), tok, args.prompt_source, L, G, N, rps, seed=42)
            res["label"] = args.label
            out_rows.append(res)
            print("POISSON", json.dumps(res), flush=True)

        if not args.skip_kv_probe:
            for L, B in [(65536, 4), (262144, 2)]:
                res = await run_kv_probe(session, args.url, str(args.model), tok, args.prompt_source, L, B, seed=77)
                res["label"] = args.label
                out_rows.append(res)
                print("KV_CAP", json.dumps(res), flush=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"label": args.label, "results": out_rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=Path("/dev/shm/inkling/model_varloop"))
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--prompt-source",
        type=Path,
        default=Path("/dev/shm/inkling/data/eval_sec5_high/longbench_v2_128k.jsonl"),
    )
    parser.add_argument(
        "--poisson-specs",
        default="4096:512:24:1.5,65536:512:12:0.25",
        help="Comma-separated L:G:N:rps tuples",
    )
    parser.add_argument("--skip-kv-probe", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
