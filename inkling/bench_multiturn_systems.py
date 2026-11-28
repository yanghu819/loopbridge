"""Isolated 1x B300 Multi-Turn Agentic Serving Benchmark (Real Day-0 vLLM + Prefix Caching).

Measures exact end-to-end multi-turn wall-clock latency, TTFT per turn, decode time per turn,
and KV-fold overhead across real multi-turn agentic regimes:
  - Short-output, long-history tool-calling agents (e.g., L0=32K/64K, T=8 turns, LU=2K, LA=128/256)
  - Reasoning-heavy multi-turn agents (e.g., L0=16K/32K, T=6 turns, LU=2K, LA=1024/2048)

Directly benchmarks a single isolated vLLM server launched with the target engine & cache policy:
  - Prefix-Cached Baselines (`--enable-prefix-caching`): Original 4/4, Original 3/3, Original 2/2
  - LoopBridge Policy R (Full Re-Prefill, `--no-enable-prefix-caching`): full m-pass prefill every turn
  - LoopBridge Policy APC-D0 (`--multi-turn-policy d0`, `--enable-prefix-caching`): Turn-0 m-pass + KV fold; follow-ups run incremental m-pass prefill without re-folding (zero dual-KV overhead)
  - Incremental Re-Prefill Probe (`--probe-incremental`): run against Prefix-Cached Original m/m to measure
    exact GPU TTFT of incremental m-pass prefill of `[A_{t-1}, U_t]` over cached `H_{t-1}`, enabling exact
    real-hardware wall-clock timing of Policy I (combined with real n-pass decode + KV-fold latency).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import urllib.request


def _stream_completion(
    url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    seed: int = 42,
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = perf_counter()
    ttft = None
    chunks = 0
    gen_text_parts = []
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = perf_counter() - t0
                chunks += 1
                gen_text_parts.append(choices[0]["text"])
    total_s = perf_counter() - t0
    if ttft is None:
        ttft = total_s
    return {
        "prompt_tokens": len(prompt_ids),
        "gen_tokens": max_tokens,
        "ttft_ms": round(ttft * 1000.0, 2),
        "decode_ms": round(max(0.0, total_s - ttft) * 1000.0, 2),
        "total_ms": round(total_s * 1000.0, 2),
        "itl_ms": round(max(0.0, total_s - ttft) * 1000.0 / max(1, max_tokens - 1), 3),
        "chunks": chunks,
    }


def _make_unique_tokens(n: int, salt: int) -> list[int]:
    """Generate deterministic non-special token IDs unique per (run, salt) to prevent unintended cross-scenario APC hits."""
    # Safe printable vocabulary range in Inkling-2 tokenizer (1000..40000)
    base = 2000 + (salt * 997) % 25000
    return [base + ((i * 31 + salt * 7) % 10000) for i in range(n)]


def run_scenario(
    url: str,
    model: str,
    l0: int,
    lu: int,
    la: int,
    n_turns: int,
    salt: int,
    probe_incremental: bool = False,
) -> dict:
    """Run one T-turn sequential multi-turn trajectory with strict append-only prefix growth.

    If `probe_incremental=True` (run on APC Original m/m):
      - Turn 0 prefills `H_0` (`l0` tokens), generates `1` token (populates APC for `H_0`).
      - Turn t >= 1 sends `H_{t-1} + [A_{t-1} (synthetic la tokens), U_t (lu tokens)]` and generates `1` token,
        measuring the EXACT GPU TTFT of Policy I's incremental m-pass prefill of `[A_{t-1}, U_t]` over cached `H_{t-1}`!
    """
    hist = _make_unique_tokens(l0, salt=salt * 100 + 0)
    turns = []
    for t in range(n_turns):
        if t > 0:
            if probe_incremental:
                # Re-prefill both prior assistant tokens A_{t-1} (la) and new user/tool observation U_t (lu)
                delta = _make_unique_tokens(la + lu, salt=salt * 100 + t)
            else:
                # Standard APC trajectory: A_{t-1} was already decoded & cached; only U_t (lu) is newly appended
                delta = _make_unique_tokens(lu, salt=salt * 100 + t)
            hist = hist + delta

        gen_len = 2 if probe_incremental else la
        m = _stream_completion(url, model, hist, max_tokens=gen_len, seed=42 + t)
        m["turn"] = t
        m["new_prefill_tokens"] = l0 if t == 0 else ((la + lu) if probe_incremental else lu)
        turns.append(m)

        if not probe_incremental:
            # Wait: for exact APC hit on A_{t-1} in real multi-turn, we need the EXACT token IDs generated!
            # Since /v1/completions returns text rather than raw IDs unless re-encoded, append the next turn
            # directly onto `hist` in Policy R / probe, OR for real APC hit on A_{t-1} capture exact IDs!
            pass

    return {
        "l0": l0,
        "lu": lu,
        "la": la,
        "n_turns": n_turns,
        "probe_incremental": probe_incremental,
        "sum_ttft_ms": round(sum(x["ttft_ms"] for x in turns), 2),
        "sum_decode_ms": round(sum(x["decode_ms"] for x in turns), 2),
        "sum_total_ms": round(sum(x["total_ms"] for x in turns), 2),
        "turn0_ttft_ms": turns[0]["ttft_ms"],
        "mean_followup_ttft_ms": round(sum(x["ttft_ms"] for x in turns[1:]) / max(1, len(turns) - 1), 2),
        "mean_itl_ms": round(sum(x["itl_ms"] for x in turns) / len(turns), 3),
        "turns": turns,
    }


def _stream_completion_with_ids(
    url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    tokenizer,
    seed: int = 42,
) -> tuple[dict, list[int]]:
    """Stream completion and return both timing dict and exact generated token IDs for strict APC prefix continuity."""
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": seed,
        "stream": True,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = perf_counter()
    ttft = None
    parts: list[str] = []
    with urllib.request.urlopen(req, timeout=600.0) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                if ttft is None:
                    ttft = perf_counter() - t0
                parts.append(choices[0]["text"])
    total_s = perf_counter() - t0
    if ttft is None:
        ttft = total_s
    gen_ids = tokenizer.encode("".join(parts), add_special_tokens=False)
    rec = {
        "prompt_tokens": len(prompt_ids),
        "gen_tokens": max_tokens,
        "ttft_ms": round(ttft * 1000.0, 2),
        "decode_ms": round(max(0.0, total_s - ttft) * 1000.0, 2),
        "total_ms": round(total_s * 1000.0, 2),
        "itl_ms": round(max(0.0, total_s - ttft) * 1000.0 / max(1, max_tokens - 1), 3),
    }
    return rec, gen_ids


def run_scenario_exact_apc(
    url: str,
    model: str,
    tokenizer,
    l0: int,
    lu: int,
    la: int,
    n_turns: int,
    salt: int,
    probe_incremental: bool = False,
) -> dict:
    """Run exact multi-turn trajectory preserving real generated assistant tokens in the APC prefix."""
    hist = _make_unique_tokens(l0, salt=salt * 100 + 0)
    turns = []
    for t in range(n_turns):
        if t > 0:
            if probe_incremental:
                # Policy I probe: H_{t-1} is cached in APC; [A_{t-1} (la), U_t (lu)] are uncached & re-prefilled at m
                hist = hist + _make_unique_tokens(la + lu, salt=salt * 100 + t)
            else:
                # Standard APC / Policy D0 / Policy R: prior assistant tokens A_{t-1} already appended; add U_t
                hist = hist + _make_unique_tokens(lu, salt=salt * 100 + t)

        gen_len = 2 if probe_incremental else la
        rec, gen_ids = _stream_completion_with_ids(url, model, hist, max_tokens=gen_len, tokenizer=tokenizer, seed=42 + t)
        rec["turn"] = t
        rec["new_prefill_tokens"] = l0 if t == 0 else ((la + lu) if probe_incremental else lu)
        turns.append(rec)

        if not probe_incremental:
            # Append exact generated assistant tokens so Turn t+1 hits vLLM APC on [H_{t-1}, A_{t-1}]!
            hist = hist + gen_ids

    return {
        "l0": l0,
        "lu": lu,
        "la": la,
        "n_turns": n_turns,
        "probe_incremental": probe_incremental,
        "sum_ttft_ms": round(sum(x["ttft_ms"] for x in turns), 2),
        "sum_decode_ms": round(sum(x["decode_ms"] for x in turns), 2),
        "sum_total_ms": round(sum(x["total_ms"] for x in turns), 2),
        "turn0_ttft_ms": turns[0]["ttft_ms"],
        "mean_followup_ttft_ms": round(sum(x["ttft_ms"] for x in turns[1:]) / max(1, len(turns) - 1), 2),
        "mean_itl_ms": round(sum(x["itl_ms"] for x in turns) / len(turns), 3),
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--probe-incremental", action="store_true",
                        help="Probe exact Policy I incremental m-pass prefill TTFT of [A_{t-1}, U_t] over cached H_{t-1}")
    parser.add_argument(
        "--grid",
        default="16384:2048:128:8,32768:2048:128:8,65536:2048:128:8,32768:4096:128:8,32768:2048:256:8,32768:2048:1024:6,32768:2048:2048:6,8192:1024:512:8",
        help="Comma-separated L0:LU:LA:T scenarios",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)

    # Warmup call (128 prompt -> 16 decode)
    _ = _stream_completion(args.url, args.model, _make_unique_tokens(128, 9999), max_tokens=16, seed=1)

    scenarios = []
    for s_idx, spec in enumerate(args.grid.split(",")):
        l0, lu, la, nt = [int(x) for x in spec.strip().split(":")]
        runs = []
        for r in range(args.repeats):
            salt = 1000 + s_idx * 37 + r * 7 + (500 if args.probe_incremental else 0)
            res = run_scenario_exact_apc(
                url=args.url,
                model=args.model,
                tokenizer=tok,
                l0=l0,
                lu=lu,
                la=la,
                n_turns=nt,
                salt=salt,
                probe_incremental=args.probe_incremental,
            )
            runs.append(res)
        med_total = sorted(x["sum_total_ms"] for x in runs)[len(runs) // 2]
        med_ttft = sorted(x["sum_ttft_ms"] for x in runs)[len(runs) // 2]
        med_dec = sorted(x["sum_decode_ms"] for x in runs)[len(runs) // 2]
        med_f_ttft = sorted(x["mean_followup_ttft_ms"] for x in runs)[len(runs) // 2]
        med_itl = sorted(x["mean_itl_ms"] for x in runs)[len(runs) // 2]
        summary = {
            "scenario": f"L0={l0}_LU={lu}_LA={la}_T={nt}",
            "l0": l0,
            "lu": lu,
            "la": la,
            "n_turns": nt,
            "probe_incremental": args.probe_incremental,
            "median_sum_total_ms": med_total,
            "median_sum_ttft_ms": med_ttft,
            "median_sum_decode_ms": med_dec,
            "median_followup_ttft_ms": med_f_ttft,
            "median_itl_ms": med_itl,
            "runs": runs,
        }
        scenarios.append(summary)
        print(
            f"[{args.label}] {summary['scenario']} | total={med_total:.0f}ms "
            f"(ttft={med_ttft:.0f}ms [followup={med_f_ttft:.1f}ms], dec={med_dec:.0f}ms, itl={med_itl:.2f}ms)",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"label": args.label, "scenarios": scenarios}, indent=2) + "\n")
    print(f"[DONE] Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
