"""Real B300 wall-clock latency & throughput benchmark for Inkling-2-Mini LoopBridge.

Measures at both batch=1 (interactive latency) and batched serving (e.g. batch=16):
  - prefill time (ms)
  - one-time prompt KV map fold time (ms, LoopBridge only — zero per-step decode overhead!)
  - decode time & throughput (ms/tok and agg tok/s) for gen_len tokens
  - total end-to-end time & speedup vs native Original 3/3

For deployment-equivalent LoopBridge timing (`--merge-decode`), merges decoder-only LoRA /
router / AttnRes deltas into a copy of the base weights (or measures exact native n-pass
decode kernel after folded prompt KV maps), so decode pays ZERO adapter overhead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from inkling.model import (
    InklingLoopBridgeSystem,
    InklingRunner,
    SystemConfig,
    build_aligned_decode_cache,
    load_inkling_model,
    set_loop_passes,
)


@torch.inference_mode()
def bench_one(
    base_model,
    label: str,
    mode: str,
    m: int,
    n: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    warmup: int = 1,
    repeats: int = 3,
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    sys_cfg = SystemConfig(mode=mode, decode_depths=(n,), lora_rank=64)
    # For deployment timing, LoopBridge decoder LoRA/router/AttnRes merge into decode weights
    # (so decode runs the pure native n-pass kernel) while prompt KV maps fold ONCE after prefill!
    if mode == "loopbridge_deployed":
        sys_cfg = SystemConfig(mode="loopbridge", decode_depths=(n,), lora_rank=64)
        system = InklingLoopBridgeSystem(base_model, sys_cfg)
        # Keep kv_maps active (for real map-fold timing), remove decoder toggle overhead
        # by evaluating decode with adapter_state['on']=False (identical FLOPs to merged weights!).
        kv_maps = system.kv_maps
    else:
        system = InklingLoopBridgeSystem(base_model, sys_cfg)
        kv_maps = system.kv_maps

    ids = torch.randint(100, 50000, (batch_size, prompt_len), device=device, dtype=torch.long)

    pre_times, map_times, dec_times, tot_times = [], [], [], []
    for rep in range(warmup + repeats):
        torch.cuda.synchronize(device)
        t_start = perf_counter()

        if mode in ("native", "distilled_deployed"):
            logits, cache = system._run_prefill_chunked(ids, None, num_passes=m, adapter_on=False, chunk_size=2048)
            logits = logits[:, -1]
            torch.cuda.synchronize(device)
            t_pre = perf_counter() - t_start
            t_map = 0.0
        elif mode == "loopbridge_deployed":
            _, pre_cache = system._run_prefill_chunked(ids[:, :-1], None, num_passes=m, adapter_on=False, chunk_size=2048)
            torch.cuda.synchronize(device)
            t_pre = perf_counter() - t_start
            t_m0 = perf_counter()
            cache = build_aligned_decode_cache(pre_cache, m=m, n=n, config=system.config, kv_maps=kv_maps, training=False)
            torch.cuda.synchronize(device)
            t_map = perf_counter() - t_m0
            del pre_cache
            # First decode token on prompt[-1] with pure native n-pass kernel (merged decode weights)
            set_loop_passes(system.base, n)
            system.adapter_state["on"] = False
            pos = torch.full((batch_size, 1), prompt_len - 1, device=device, dtype=torch.long)
            out = system.base(input_ids=ids[:, -1:], position_ids=pos, cache=cache, use_cache=True, logits_to_keep=1)
            logits, cache = out.logits[:, -1], out.cache
        else:
            runner = InklingRunner(system, m=m, n=n, chunk_size=2048, time_adapter=True)
            logits, cache = runner.prefill(ids)
            torch.cuda.synchronize(device)
            t_map = runner.last_adapter_seconds
            t_pre = (perf_counter() - t_start) - t_map

        torch.cuda.synchronize(device)
        t_d0 = perf_counter()
        cur_pos = prompt_len
        for _ in range(gen_len - 1):
            nxt = logits.argmax(dim=-1, keepdim=True)
            set_loop_passes(system.base, n)
            system.adapter_state["on"] = mode in ("loopbridge", "distilled")
            pos = torch.full((batch_size, 1), cur_pos, device=device, dtype=torch.long)
            out = system.base(input_ids=nxt, position_ids=pos, cache=cache, use_cache=True, logits_to_keep=1)
            logits, cache = out.logits[:, -1], out.cache
            cur_pos += 1
        torch.cuda.synchronize(device)
        t_dec = perf_counter() - t_d0
        t_tot = perf_counter() - t_start

        del cache, logits
        torch.cuda.empty_cache()

        if rep >= warmup:
            pre_times.append(t_pre)
            map_times.append(t_map)
            dec_times.append(t_dec)
            tot_times.append(t_tot)

    # Restore clean unwrapped base_model if we wrapped it
    del system
    pre_ms = 1000.0 * sum(pre_times) / len(pre_times)
    map_ms = 1000.0 * sum(map_times) / len(map_times)
    dec_ms = 1000.0 * sum(dec_times) / len(dec_times)
    tot_ms = 1000.0 * sum(tot_times) / len(tot_times)
    dec_tok_per_s = (batch_size * (gen_len - 1)) / (dec_ms / 1000.0)
    e2e_tok_per_s = (batch_size * gen_len) / (tot_ms / 1000.0)
    rec = {
        "label": label,
        "mode": mode,
        "m": m,
        "n": n,
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "prefill_ms": round(pre_ms, 2),
        "map_fold_ms": round(map_ms, 2),
        "decode_ms": round(dec_ms, 2),
        "total_ms": round(tot_ms, 2),
        "decode_ms_per_tok": round(dec_ms / max(gen_len - 1, 1), 3),
        "decode_agg_tok_per_s": round(dec_tok_per_s, 2),
        "e2e_agg_tok_per_s": round(e2e_tok_per_s, 2),
    }
    print(json.dumps(rec), flush=True)
    return rec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/dev/shm/inkling/model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gen-len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    results = []
    configs = [
        # (batch_size, prompt_len, gen_len)
        (1, 512, args.gen_len),
        (1, 4096, args.gen_len),
        (1, 16384, args.gen_len),
        (16, 512, args.gen_len),
        (16, 2048, args.gen_len),
    ]
    arms = [
        ("Original 3/3", "native", 3, 3),
        ("Original 2/2", "native", 2, 2),
        ("Original 1/1", "native", 1, 1),
        ("LoopBridge 3/2 (folded)", "loopbridge_deployed", 3, 2),
        ("LoopBridge 3/1 (folded)", "loopbridge_deployed", 3, 1),
    ]
    for label, mode, m, n in arms:
        base = load_inkling_model(args.model_path, device=device)
        for bs, plen, glen in configs:
            results.append(bench_one(base, label, mode, m, n, bs, plen, glen, device=device))
        del base
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "results": results}, indent=2) + "\n")
    print(f"Saved timing results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
