"""High-concurrency multi-replica vLLM evaluator for Inkling-2-Mini official benchmarks.

Queries 1..8 local vLLM OpenAI-compatible servers (e.g. 8x TP=1 replicas on 8x B300) using
pre-tokenized `prompt_ids` with `skip_special_tokens=False` so `<|content_thinking|>` and
`<|content_text|>` match the native template bit-for-bit, scored by `inkling.scoring`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
from time import perf_counter
import urllib.error
import urllib.request

from inkling.scoring import score_row


def _post_completion(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float = 1800.0,
) -> tuple[str, int, str, float]:
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
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_s = perf_counter() - t0
    choice = body["choices"][0]
    raw_text = choice.get("text", "")
    usage = body.get("usage", {})
    gen_tokens = int(usage.get("completion_tokens", 0))
    fr = choice.get("finish_reason", "stop")
    reason = "max_new_tokens" if fr == "length" else "eos"
    return raw_text, gen_tokens, reason, wall_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8000"],
                        help="One or more vLLM server base URLs (round-robined across requests)")
    parser.add_argument("--model", default="/dev/shm/inkling/model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benches", nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Total concurrent in-flight requests across all --urls")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens-map", default="")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Samples per item (e.g. 8 for Apex 2025 avg@8)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    max_tok_overrides: dict[str, int] = {}
    for item in args.max_new_tokens_map.split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            max_tok_overrides[k.strip()] = int(v.strip())

    args.output.mkdir(parents=True, exist_ok=True)
    urls = args.urls

    for bench in args.benches:
        t0 = perf_counter()
        all_rows = [json.loads(line) for line in (args.data / f"{bench}.jsonl").open() if line.strip()]
        if args.limit and args.limit < len(all_rows):
            all_rows = all_rows[: args.limit]
        if bench in max_tok_overrides:
            for r in all_rows:
                r["max_new_tokens"] = max_tok_overrides[bench]

        # Expand for multi-sample avg@K if requested
        jobs: list[tuple[str, dict, int]] = []
        for r in all_rows:
            if args.n_samples == 1:
                jobs.append((r["id"], r, args.seed))
            else:
                for k in range(args.n_samples):
                    jobs.append((f"{r['id']}#s{k}", r, args.seed + k * 1009))

        out_path = args.output / f"{bench}.jsonl"
        done: dict[str, dict] = {}
        if out_path.exists():
            for line in out_path.open():
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()))

        todo = [j for j in jobs if j[0] not in done]
        print(f"[vLLM] {bench}: total={len(jobs)}, done={len(done)}, todo={len(todo)}, urls={len(urls)}, conc={args.concurrency}", flush=True)

        in_flight = {u: 0 for u in urls}
        rr_cursor = 0
        url_lock = threading.Lock()

        def acquire_url() -> str:
            nonlocal rr_cursor
            with url_lock:
                min_load = min(in_flight.values())
                n_urls = len(urls)
                for offset in range(n_urls):
                    cand = urls[(rr_cursor + offset) % n_urls]
                    if in_flight[cand] == min_load:
                        in_flight[cand] += 1
                        rr_cursor = (rr_cursor + offset + 1) % n_urls
                        return cand
                cand = urls[0]
                in_flight[cand] += 1
                return cand

        def release_url(url: str) -> None:
            with url_lock:
                in_flight[url] = max(0, in_flight[url] - 1)

        def run_one(idx_job):
            _idx, (jid, row, seed) = idx_job
            url = acquire_url()
            try:
                last_err = None
                for attempt in range(3):
                    try:
                        raw_txt, n_gen, reason, wall_s = _post_completion(
                            url,
                            args.model,
                            row["prompt_ids"],
                            row["max_new_tokens"],
                            args.temperature,
                            args.top_p,
                            seed + attempt,
                            timeout=args.timeout,
                        )
                        rec = score_row(bench, row, raw_txt, n_gen, reason)
                        rec["id"] = jid
                        rec["wall_s"] = round(wall_s, 4)
                        return rec
                    except Exception as e:
                        last_err = e
                raise RuntimeError(f"Failed {jid} on {url}: {last_err}")
            finally:
                release_url(url)

        completed = 0
        with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(run_one, pair) for pair in enumerate(todo)]
            for fut in as_completed(futs):
                rec = fut.result()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["id"]] = rec
                completed += 1
                if completed % 25 == 0 or completed == len(todo):
                    vals = [float(r["correct"]) if "correct" in r else float(r["score"]) for r in done.values()]
                    eos_vals = [float(r["correct"]) if "correct" in r else float(r["score"]) for r in done.values() if r.get("stop_reason") == "eos"]
                    tr = len(done) - len(eos_vals)
                    print(
                        f"  [{bench}] {len(done)}/{len(jobs)} | score={100*sum(vals)/len(vals):.2f}% "
                        f"| eos({len(eos_vals)})={100*sum(eos_vals)/max(1,len(eos_vals)):.2f}% "
                        f"| trunc={tr}/{len(done)} ({100*tr/len(done):.1f}%) | elapsed={perf_counter()-t0:.1f}s",
                        flush=True,
                    )

        vals = [float(r["correct"]) if "correct" in r else float(r["score"]) for r in done.values()]
        eos_vals = [float(r["correct"]) if "correct" in r else float(r["score"]) for r in done.values() if r.get("stop_reason") == "eos"]
        tr = len(done) - len(eos_vals)
        tot_toks = sum(int(r.get("generated_tokens", 0)) for r in done.values())
        elapsed = perf_counter() - t0
        print(
            f"[DONE] {bench}: score={100*sum(vals)/max(1,len(vals)):.2f}% ({len(done)} rows), "
            f"eos={100*sum(eos_vals)/max(1,len(eos_vals)):.2f}%, trunc={100*tr/max(1,len(done)):.1f}%, "
            f"agg_tok_s={tot_toks/max(elapsed,1e-6):.1f} tok/s, wall={elapsed:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
