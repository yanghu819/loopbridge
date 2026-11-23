"""8-GPU DDP forward-KL distillation trainer for Inkling-2-Mini LoopBridge & Distilled baselines."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from inkling.model import InklingLoopBridgeSystem, SystemConfig, load_inkling_model


def pad_batch(samples: list[dict], pad_id: int, device: torch.device):
    B = len(samples)
    P = max(len(s["prefix_ids"]) for s in samples)
    S = max(len(s["suffix_ids"]) for s in samples)
    prefix = torch.full((B, P), pad_id, dtype=torch.long)
    pmask = torch.zeros((B, P), dtype=torch.long)
    suffix = torch.full((B, S), pad_id, dtype=torch.long)
    smask = torch.zeros((B, S), dtype=torch.long)
    for i, s in enumerate(samples):
        p, q = s["prefix_ids"], s["suffix_ids"]
        prefix[i, P - len(p) :] = torch.tensor(p, dtype=torch.long)
        pmask[i, P - len(p) :] = 1
        suffix[i, : len(q)] = torch.tensor(q, dtype=torch.long)
        smask[i, : len(q)] = 1
    return prefix.to(device), pmask.to(device), suffix.to(device), smask.to(device)


class DDPWrapper(torch.nn.Module):
    def __init__(self, system: InklingLoopBridgeSystem, pad_id: int):
        super().__init__()
        self.system = system
        self.pad_id = pad_id

    def forward(self, samples: list[dict], m: int, n: int):
        prefix, pmask, suffix, smask = pad_batch(samples, self.pad_id, self.system.device)
        return self.system.forward_kl(prefix, pmask, suffix, smask, m=m, n=n)


def learning_rate(tokens: int, planned_tokens: int, peak_lr: float, min_lr: float) -> float:
    warmup = max(1, int(planned_tokens * 0.03))
    if tokens <= warmup:
        return peak_lr * tokens / warmup
    progress = min(1.0, (tokens - warmup) / max(1, planned_tokens - warmup))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def load_stream(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--data", type=Path, required=True, help="Path to train.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("loopbridge", "distilled"), required=True)
    parser.add_argument("--cells", default="3x1", help="Comma-separated m x n cells, e.g. '3x1' or '3x1,2x1,1x1'")
    parser.add_argument("--cell-weights", default="", help="Optional comma-separated sampling weights matching --cells")
    parser.add_argument("--teacher-passes", type=int, default=3, help="Teacher loop depth (default 3)")
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--freeze-maps", action="store_true")
    parser.add_argument("--kv-map-layers", choices=("all", "global"), default="global",
                        help="Prompt KV map placement: 'global' (L11,L17,L23 only — 6x fewer params/fold) or 'all' (18 layers)")
    parser.add_argument("--no-cross-pass-kv", action="store_true",
                        help="Disable Cross-Pass Global KV Fusion (ablation: map only aligned pass j=m-n+k)")
    parser.add_argument("--adapt-pre-loop", action="store_true",
                        help="Adapt pre-loop layers 0..5 (automatically enabled for mode='distilled' to keep param count fair)")
    parser.add_argument("--no-router", action="store_true")
    parser.add_argument("--no-attnres", action="store_true")
    parser.add_argument("--no-rel-sink", action="store_true")
    parser.add_argument("--freeze-sink", action="store_true",
                        help="Freeze sink_proj while keeping relative_position_bias (protects 1M attention sink calibration)")
    parser.add_argument("--jsd-beta", type=float, default=0.0,
                        help="0.0 = pure Forward KL; (0,1) e.g. 0.5 = Hessian-matched Generalized JSD for on-policy GKD")
    parser.add_argument("--length-bucket", action="store_true",
                        help="Bucket rows into equal-length global_batch blocks then shuffle blocks (zero straggler wait for 16K-56K long rows)")
    parser.add_argument("--micro", type=int, default=4)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--peak-lr", type=float, default=1.5e-4)
    parser.add_argument("--min-lr", type=float, default=1.5e-5)
    parser.add_argument("--stop-tokens", type=int, default=24_000_000)
    parser.add_argument("--planned-tokens", type=int, default=24_000_000)
    parser.add_argument("--save-every-m", default="8,16,24", help="Comma-separated milestone token counts in millions")
    parser.add_argument("--seed", type=int, default=31847)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing on backbone (saves >150GB VRAM for full-seq Distilled runs)")
    args = parser.parse_args()

    cells = [tuple(int(x) for x in c.split("x")) for c in args.cells.split(",")]
    cell_weights = [float(w) for w in args.cell_weights.split(",")] if args.cell_weights else [1.0] * len(cells)
    decode_depths = tuple(sorted({n for _, n in cells}))
    milestones = sorted(int(float(x) * 1_000_000) for x in args.save_every_m.split(",") if x.strip())

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))

    assert args.global_batch % world == 0, "global_batch must divide evenly across ranks"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_stream(args.data)
    if args.length_bucket and len(rows) >= args.global_batch:
        rows.sort(key=lambda r: len(r["prefix_ids"]) + len(r["suffix_ids"]))
        buckets = [rows[i : i + args.global_batch] for i in range(0, len(rows), args.global_batch)]
        random.Random(args.seed + 7919).shuffle(buckets)
        rows = [r for b in buckets for r in b]
    pad_id = 199999  # Inkling-2-Mini pad_token_id

    base = load_inkling_model(args.model_path, device=device)
    if args.gradient_checkpointing and args.mode == "loopbridge":
        raise ValueError(
            "--gradient-checkpointing is incompatible with mode='loopbridge' because recomputation "
            "mutates DynamicCache KV/conv states in-place (supported for mode='distilled' use_cache=False only)."
        )
    adapt_pre = True if args.mode == "distilled" else args.adapt_pre_loop
    cross_pass = (not args.no_cross_pass_kv) and (args.kv_map_layers == "global")
    max_prefill_passes = max(max(m for m, _ in cells), args.teacher_passes, 3)
    sys_cfg = SystemConfig(
        mode=args.mode,
        decode_depths=decode_depths,
        lora_rank=args.lora_rank,
        freeze_maps=args.freeze_maps,
        kv_map_layers=args.kv_map_layers,
        kv_map_cross_pass=cross_pass,
        adapt_router=not args.no_router,
        adapt_attnres=not args.no_attnres,
        adapt_rel_sink=not args.no_rel_sink,
        adapt_sink=not args.freeze_sink,
        adapt_pre_loop=adapt_pre,
        max_prefill_passes=max_prefill_passes,
        teacher_passes=args.teacher_passes,
        jsd_beta=args.jsd_beta,
    )
    system = InklingLoopBridgeSystem(base, sys_cfg)
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in base.model.layers:
            layer.gradient_checkpointing = True
            layer.train()
        base.model.train()  # GradientCheckpointingLayer checks self.training; dropout is 0.0 in Inkling2Loop
    trainable = system.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable, lr=args.peak_lr, betas=(0.9, 0.95), weight_decay=0.0)

    next_sample, consumed, update = 0, 0, 0
    saved_milestones: set[int] = set()
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        system.load_trainable_state(ckpt["trainable_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        next_sample = ckpt["next_sample"]
        consumed = ckpt["kl_tokens"]
        update = ckpt["optimizer_steps"]
        saved_milestones = {m for m in milestones if m <= consumed}

    wrapper = DDPWrapper(system, pad_id=pad_id)
    find_unused = len(decode_depths) > 1
    model = (
        DistributedDataParallel(
            wrapper,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=find_unused,
        )
        if world > 1
        else wrapper
    )

    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        setup = {
            "event": "start",
            "mode": args.mode,
            "cells": cells,
            "cell_weights": cell_weights,
            "system_config": sys_cfg.__dict__,
            "trainable_parameters": sum(p.numel() for p in trainable),
            "world_size": world,
            "global_batch": args.global_batch,
            "micro": args.micro,
            "peak_lr": args.peak_lr,
            "min_lr": args.min_lr,
            "planned_tokens": args.planned_tokens,
            "stop_tokens": args.stop_tokens,
            "milestones": milestones,
            "data_rows": len(rows),
            "seed": args.seed,
        }
        (args.output / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
        print(json.dumps(setup), flush=True)

    def save_checkpoint(tag: str, last_stats: dict):
        if rank != 0:
            return
        ckpt = {
            "mode": args.mode,
            "cells": cells,
            "model_path": str(args.model_path),
            "system_config": sys_cfg.__dict__,
            "trainable_state": system.trainable_state(),
            "optimizer": optimizer.state_dict(),
            "optimizer_steps": update,
            "kl_tokens": consumed,
            "next_sample": next_sample,
            "last_update": last_stats,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        out_pt = args.output / f"checkpoint-{tag}.pt"
        torch.save(ckpt, out_pt)
        print(f"[rank 0] Saved checkpoint: {out_pt} (kl_tokens={consumed})", flush=True)

    started = perf_counter()
    last_stats = {}
    n_rows = len(rows)

    while consumed < args.stop_tokens:
        end = next_sample + args.global_batch
        if end > n_rows:
            # Wrap deterministically with epoch shift if needed
            batch_rows = [rows[(next_sample + i) % n_rows] for i in range(args.global_batch)]
        else:
            batch_rows = rows[next_sample:end]

        # Sort within global batch by length so micro-batches minimize padding waste (identical set of rows!)
        batch_rows.sort(key=lambda r: len(r["prefix_ids"]) + len(r["suffix_ids"]))
        block_tokens = sum(r["effective_tokens"] for r in batch_rows)

        lr = learning_rate(consumed + block_tokens, args.planned_tokens, args.peak_lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad(set_to_none=True)

        stats = torch.zeros(4, device=device, dtype=torch.float64)
        # Partition length-sorted global batch into equal-length blocks before striding across ranks
        # so every microbatch across all ranks has tight length balance and minimal padding waste.
        step_size = world * args.micro
        chunks = [
            batch_rows[b + rank : min(b + step_size, len(batch_rows)) : world]
            for b in range(0, len(batch_rows), step_size)
        ]
        chunks = [c for c in chunks if c]

        # Multi-depth DDP safety: when len(decode_depths) > 1, PyTorch DDP find_unused_parameters
        # only traverses the final synced microbatch and would skip all_reduce for depth-specific
        # parameters touched in earlier no_sync() microbatches. Run all microbatches under no_sync()
        # and explicitly all_reduce(AVG) every trainable parameter so all ranks stay bit-identical.
        manual_sync = world > 1 and len(decode_depths) > 1
        for micro_idx, chunk in enumerate(chunks):
            # Pick cell (m, n) deterministically per (update, micro_idx) — same across ranks
            rng_cell = random.Random(args.seed * 1315423911 + update * 997 + micro_idx)
            m, n = rng_cell.choices(cells, weights=cell_weights, k=1)[0]
            syncing = (not manual_sync) and (micro_idx == len(chunks) - 1)
            ctx = nullcontext() if (syncing or world == 1) else model.no_sync()
            with ctx:
                loss_sum, metrics = model(chunk, m, n)
                (loss_sum * world / block_tokens).backward()
            stats += metrics.to(torch.float64)
            del loss_sum, metrics

        if manual_sync:
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if torch.isfinite(grad_norm):
            optimizer.step()
        else:
            if rank == 0:
                print(f"[WARN] Non-finite grad_norm={grad_norm} at step {update+1}, skipping step", flush=True)
            optimizer.zero_grad(set_to_none=True)

        if world > 1:
            dist.all_reduce(stats)

        vals = stats.tolist()
        toks_actual = max(vals[3], 1.0)
        consumed += int(vals[3])
        update += 1
        next_sample = end % n_rows

        if world > 1 and update == 20:
            max_diff = 0.0
            with torch.no_grad():
                for p in trainable:
                    hi = p.data.detach().clone()
                    lo = p.data.detach().clone()
                    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
                    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
                    d = float((hi - lo).abs().max().item())
                    if d > max_diff:
                        max_diff = d
            if rank == 0:
                print(json.dumps({"event": "ddp_parity_check", "step": 20, "max_rank_diff": max_diff}), flush=True)
            assert max_diff == 0.0, f"DDP rank parameter divergence detected at step 20: max_rank_diff={max_diff}"

        elapsed = perf_counter() - started
        last_stats = {
            "event": "update",
            "mode": args.mode,
            "cells": args.cells,
            "step": update,
            "kl_tokens": consumed,
            "block_tokens": int(vals[3]),
            "kl": vals[0] / toks_actual,
            "student_ce": vals[1] / toks_actual,
            "teacher_ce": vals[2] / toks_actual,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tok_per_sec": consumed / max(elapsed, 1e-6),
            "elapsed_sec": elapsed,
        }
        if rank == 0 and (update == 1 or update % args.log_every == 0 or consumed >= args.stop_tokens):
            print(json.dumps(last_stats), flush=True)
            with (args.output / "train.jsonl").open("a") as fh:
                fh.write(json.dumps(last_stats) + "\n")
            (args.output / "progress.json").write_text(json.dumps(last_stats, indent=2) + "\n")

        for ms in milestones:
            if consumed >= ms and ms not in saved_milestones:
                saved_milestones.add(ms)
                save_checkpoint(f"{ms // 1_000_000}m", last_stats)

    if rank == 0:
        final_tag = f"{args.stop_tokens // 1_000_000}m"
        if not (args.output / f"checkpoint-{final_tag}.pt").exists():
            save_checkpoint(final_tag, last_stats)
        result = {
            "status": "completed",
            "mode": args.mode,
            "cells": args.cells,
            "kl_tokens": consumed,
            "optimizer_steps": update,
            "elapsed_sec": perf_counter() - started,
            "last_update": last_stats,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
