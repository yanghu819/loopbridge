"""Real B300 wall-clock latency & throughput benchmark for Inkling-2-Mini LoopBridge.

Measures at both batch=1 (interactive latency) and batched serving (batch=16, 32):
  - prefill time (ms)
  - one-time prompt KV map fold time (ms, LoopBridge only — zero per-step decode overhead)
  - TTFT (ms): prefill + map fold + first-token step
  - decode time & throughput (ms/tok and agg tok/s) for gen_len - 1 subsequent steps
  - total end-to-end time & speedup vs native Original 3/3

Deployed LoopBridge (`loopbridge_deployed`) folds decoder LoRA / MoE router / AttnRes deltas
into the decode weights (exact native n-pass kernel, zero per-step wrapper overhead) while
still executing the real one-time InklingKVMapAdapter prompt KV fold after prefill.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from inkling.model import (
    InklingKVMapAdapter,
    InklingLoopBridgeSystem,
    InklingRunner,
    SystemConfig,
    build_aligned_decode_cache,
    load_inkling_model,
    set_loop_passes,
)


@torch.inference_mode()
def _prefill_chunked_unwrapped(
    base_model,
    input_ids: torch.Tensor,
    num_passes: int,
    chunk_size: int = 2048,
):
    """Chunked prefill on an unwrapped Inkling2LoopForCausalLM (skips lm_head on non-final chunks)."""
    set_loop_passes(base_model, num_passes)
    bsz, seqlen = input_ids.shape
    cache = None
    pos_all = torch.arange(seqlen, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    last_hidden = None
    for s in range(0, seqlen, chunk_size):
        e = min(s + chunk_size, seqlen)
        out = base_model.model(
            input_ids=input_ids[:, s:e],
            position_ids=pos_all[:, s:e],
            cache=cache,
            use_cache=True,
        )
        cache = out.cache
        if e == seqlen:
            last_hidden = out.last_hidden_state[:, -1:]
    logits = base_model.lm_head(last_hidden)[:, -1]
    return logits, cache


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
    kv_maps: InklingKVMapAdapter | None = None,
    system: InklingLoopBridgeSystem | None = None,
    warmup: int = 1,
    repeats: int = 3,
    device: torch.device = torch.device("cuda:0"),
) -> dict:
    gen = torch.Generator(device=device).manual_seed(0)
    ids = torch.randint(100, 50000, (batch_size, prompt_len), generator=gen, device=device, dtype=torch.long)

    pre_times, map_times, ttft_times, dec_times, tot_times = [], [], [], [], []
    for rep in range(warmup + repeats):
        torch.cuda.synchronize(device)
        t_start = perf_counter()

        if mode == "native":
            logits, cache = _prefill_chunked_unwrapped(base_model, ids, num_passes=m, chunk_size=2048)
            torch.cuda.synchronize(device)
            t_pre = perf_counter() - t_start
            t_map = 0.0
            t_ttft = t_pre
        elif mode == "loopbridge_deployed":
            _, pre_cache = _prefill_chunked_unwrapped(base_model, ids[:, :-1], num_passes=m, chunk_size=2048)
            torch.cuda.synchronize(device)
            t_pre = perf_counter() - t_start
            t_m0 = perf_counter()
            cache = build_aligned_decode_cache(
                pre_cache, m=m, n=n, config=base_model.config, kv_maps=kv_maps, training=False
            )
            torch.cuda.synchronize(device)
            t_map = perf_counter() - t_m0
            del pre_cache
            # First generated token on prompt[-1] with pure native n-pass kernel (merged decode weights)
            set_loop_passes(base_model, n)
            pos = torch.full((batch_size, 1), prompt_len - 1, device=device, dtype=torch.long)
            out = base_model(input_ids=ids[:, -1:], position_ids=pos, cache=cache, use_cache=True, logits_to_keep=1)
            logits, cache = out.logits[:, -1], out.cache
            torch.cuda.synchronize(device)
            t_ttft = perf_counter() - t_start
        else:
            assert system is not None
            runner = InklingRunner(system, m=m, n=n, chunk_size=2048, time_adapter=True)
            logits, cache = runner.prefill(ids)
            torch.cuda.synchronize(device)
            t_ttft = perf_counter() - t_start
            t_map = runner.last_adapter_seconds
            t_pre = t_ttft - t_map

        torch.cuda.synchronize(device)
        t_d0 = perf_counter()
        cur_pos = prompt_len
        for _ in range(gen_len - 1):
            nxt = logits.argmax(dim=-1, keepdim=True)
            if mode in ("native", "loopbridge_deployed"):
                set_loop_passes(base_model, n)
                pos = torch.full((batch_size, 1), cur_pos, device=device, dtype=torch.long)
                out = base_model(input_ids=nxt, position_ids=pos, cache=cache, use_cache=True, logits_to_keep=1)
            else:
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
            ttft_times.append(t_ttft)
            dec_times.append(t_dec)
            tot_times.append(t_tot)

    pre_ms = 1000.0 * sum(pre_times) / len(pre_times)
    map_ms = 1000.0 * sum(map_times) / len(map_times)
    ttft_ms = 1000.0 * sum(ttft_times) / len(ttft_times)
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
        "ttft_ms": round(ttft_ms, 2),
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include-unmerged", action="store_true")
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
        (32, 512, args.gen_len),
    ]

    # Single unwrapped base model serves all native + folded-deployment arms with zero mutation!
    base = load_inkling_model(args.model_path, device=device)
    kv_maps_1 = InklingKVMapAdapter(base.config, device=device, decode_depths=(1,))
    kv_maps_2 = InklingKVMapAdapter(base.config, device=device, decode_depths=(2,))
    deployed_arms = [
        ("Original 3/3", "native", 3, 3, None),
        ("Original 2/2", "native", 2, 2, None),
        ("Original 1/1", "native", 1, 1, None),
        ("LoopBridge 3/2 (folded)", "loopbridge_deployed", 3, 2, kv_maps_2),
        ("LoopBridge 3/1 (folded)", "loopbridge_deployed", 3, 1, kv_maps_1),
    ]
    for label, mode, m, n, kvm in deployed_arms:
        for bs, plen, glen in configs:
            results.append(
                bench_one(base, label, mode, m, n, bs, plen, glen, kv_maps=kvm, repeats=args.repeats, device=device)
            )
    del kv_maps_1, kv_maps_2, base
    torch.cuda.empty_cache()

    if args.include_unmerged:
        for label, m, n in [
            ("LoopBridge 3/2 (unmerged)", 3, 2),
            ("LoopBridge 3/1 (unmerged)", 3, 1),
        ]:
            base = load_inkling_model(args.model_path, device=device)
            system = InklingLoopBridgeSystem(base, SystemConfig(mode="loopbridge", decode_depths=(n,), lora_rank=64))
            for bs, plen, glen in configs:
                results.append(
                    bench_one(base, label, "loopbridge", m, n, bs, plen, glen, system=system, repeats=args.repeats, device=device)
                )
            del system, base
            torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "results": results}, indent=2) + "\n")
    print(f"Saved timing results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
