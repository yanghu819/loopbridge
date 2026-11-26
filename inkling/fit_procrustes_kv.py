"""Closed-form 'Plato's Key' Orthogonal Procrustes (and rescaled cross-pass Ridge)
KV-map fitter for LoopBridge (zero gradient descent, zero labels, zero synthetic templates).

Inspired by Cheng et al. (2026), 'Plato's Key' (Hazy Research, Stanford):
collects paired prompt KV states (m-pass prefill -> n-pass prefill) on unlabeled natural
prompts at the three looped/post-loop global NoPE layers (L11, L17, L23), fits a closed-form
head-wise orthogonal rotation + global scale (s * R, via thin SVD Procrustes) or rescaled
cross-pass ridge map, and saves a native LoopBridge KV-Only checkpoint (`lora_rank=0`)
directly loadable by Day-0 vLLM (`inkling.vllm_loopbridge serve`) with zero decode overhead.

Usage:
  python3 -m inkling.fit_procrustes_kv \
    --model-path /dev/shm/inkling/model_varloop \
    --data /dev/shm/inkling/data/train_varloop_v5_nosynth_t4/train.jsonl \
    --m 4 --n 2 --pass-pairing anchored --method procrustes \
    --num-prompts 192 --max-prompt-len 2048 \
    --output /dev/shm/inkling/runs/vl_procrustes_4_2/checkpoint-procrustes.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from time import perf_counter

import torch

from inkling.model import (
    InklingLoopBridgeSystem,
    SystemConfig,
    aligned_pass_indices,
    cache_slot,
    load_inkling_model,
)


def _get_layer_kv_per_pass(cache, layer_id: int, num_passes: int, is_loop: bool, config) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-pass (K, V) tensors [H_kv, L, D] in float32 matching build_aligned_decode_cache."""
    if is_loop:
        out = []
        for p in range(num_passes):
            sl = cache.layers[cache_slot(p, layer_id, config)]
            out.append((sl.keys[0].float(), sl.values[0].float()))
        return out
    sl = cache.layers[layer_id]
    return [(sl.keys[0].float(), sl.values[0].float())]


@torch.no_grad()
def fit_procrustes_map(X_sum_xt_x: torch.Tensor, X_sum_xt_y: torch.Tensor, x_norm2: torch.Tensor, y_norm2: torch.Tensor) -> torch.Tensor:
    """Closed-form Orthogonal Procrustes + global scale per head (Plato's Key).

    Given stacked row vectors X [N, D] (source m-pass) and Y [N, D] (target n-pass),
    we receive sufficient statistics per head:
      C = Y^T X  (passed as X_sum_xt_y.transpose(-1,-2)) -> SVD: C = U S V^T, R = U V^T
      s = sqrt(||Y||_F^2 / ||X||_F^2)
    Returns W = s * R [H, D, D] such that y_hat = x @ W^T = (I + Delta) x (matching einsum 'bhti,hoi->bhto').
    """
    H, D, _ = X_sum_xt_y.shape
    W = torch.zeros((H, D, D), dtype=torch.float32)
    for h in range(H):
        # X_sum_xt_y[h] = sum_t x_t y_t^T = X^T Y -> C = Y^T X = (X^T Y)^T
        C = X_sum_xt_y[h].T.double()
        U, _, Vt = torch.linalg.svd(C, full_matrices=False)
        R = U @ Vt
        s = math.sqrt(max(1e-12, float(y_norm2[h].item())) / max(1e-12, float(x_norm2[h].item())))
        W[h] = (s * R).float()
    return W


@torch.no_grad()
def fit_rescaled_ridge_cross(G_xx: torch.Tensor, G_xy: torch.Tensor, y_norm2: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """Closed-form rescaled multi-pass Ridge mapping [m*D -> D] per head.

    G_xx: [H, m*D, m*D] = X^T X, G_xy: [H, m*D, D] = X^T Y.
    Solves W_row = (X^T X + lam * trace/(m*D) * I)^-1 X^T Y, then rescales per head to match ||Y||_F
    (per Plato's Key Sec. 1 finding that preserving recipient norm prevents least-squares shrinkage!).
    Returns W [H, D, m*D] such that y_hat = concat(x_0..x_{m-1}) @ W^T.
    """
    H, MD, D = G_xy.shape
    W_out = torch.zeros((H, D, MD), dtype=torch.float32)
    eye = torch.eye(MD, dtype=torch.float64)
    for h in range(H):
        A = G_xx[h].double()
        B = G_xy[h].double()
        reg = lam * (A.diagonal().mean().item() + 1e-8)
        W_col = torch.linalg.solve(A + reg * eye, B)  # [MD, D]
        # Variance of predicted y_hat = trace(W_col^T A W_col)
        pred_norm2 = float(torch.sum(W_col * (A @ W_col)).item())
        scale = math.sqrt(max(1e-12, float(y_norm2[h].item())) / max(1e-12, pred_norm2))
        W_out[h] = (scale * W_col.T).float()
    return W_out


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--pass-pairing", choices=("tail", "anchored"), default="anchored")
    parser.add_argument("--method", choices=("procrustes", "cross_ridge"), default="procrustes")
    parser.add_argument("--ridge-lam", type=float, default=5e-2)
    parser.add_argument("--num-prompts", type=int, default=192)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    t0 = perf_counter()
    dev = torch.device(args.device)
    rows = [json.loads(l) for l in args.data.open() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    chosen = [r["prefix_ids"][: args.max_prompt_len] for r in rows if len(r["prefix_ids"]) >= 64][: args.num_prompts]
    print(f"[Procrustes-KV] Loaded {len(chosen)} unlabeled prompts (max_len={args.max_prompt_len})", flush=True)

    base = load_inkling_model(args.model_path, device=dev)
    sys_cfg = SystemConfig(
        mode="loopbridge",
        decode_depths=(args.n,),
        lora_rank=0,
        freeze_maps=False,
        kv_map_layers="global",
        kv_map_cross_pass=True,
        adapt_router=False,
        adapt_attnres=False,
        adapt_rel_sink=False,
        adapt_sink=False,
        adapt_pre_loop=False,
        max_prefill_passes=max(args.m, 4),
        teacher_passes=args.m,
        pass_pairing=args.pass_pairing,
    )
    system = InklingLoopBridgeSystem(base, sys_cfg)
    system.adapter_state["on"] = False

    kv_maps = system.kv_maps
    loop_layers = [kv_maps.loop_start + off for off in kv_maps.loop_offsets]
    post_layers = [kv_maps.loop_end + off for off in kv_maps.post_offsets]
    H = system.config.num_key_value_heads
    D = system.config.head_dim
    m, n = args.m, args.n
    paired = aligned_pass_indices(m, n, args.pass_pairing)
    print(f"[Procrustes-KV] m={m}, n={n}, pairing={args.pass_pairing} ({paired}), loop_layers={loop_layers}, post_layers={post_layers}", flush=True)

    # Accumulate sufficient statistics in float64 on GPU -> exact, O(1) memory regardless of N tokens!
    n_loop = len(loop_layers)
    n_post = len(post_layers)
    if args.method == "procrustes":
        # Pairwise aligned (K_m[j] -> K_n[k], V_m[j] -> V_n[k]) + post
        k_xy_loop = torch.zeros((n, n_loop, H, D, D), dtype=torch.float64, device=dev)
        v_xy_loop = torch.zeros((n, n_loop, H, D, D), dtype=torch.float64, device=dev)
        k_x2_loop = torch.zeros((n, n_loop, H), dtype=torch.float64, device=dev)
        k_y2_loop = torch.zeros((n, n_loop, H), dtype=torch.float64, device=dev)
        v_x2_loop = torch.zeros((n, n_loop, H), dtype=torch.float64, device=dev)
        v_y2_loop = torch.zeros((n, n_loop, H), dtype=torch.float64, device=dev)
    else:
        MD = m * D
        k_xx_loop = torch.zeros((n, n_loop, H, MD, MD), dtype=torch.float64, device=dev)
        k_xy_loop = torch.zeros((n, n_loop, H, MD, D), dtype=torch.float64, device=dev)
        k_y2_loop = torch.zeros((n, n_loop, H), dtype=torch.float64, device=dev)
        v_xx_loop = torch.zeros((n, n_loop, H, MD, MD), dtype=torch.float64, device=dev)
        v_xy_loop = torch.zeros((n, n_loop, H, MD, D), dtype=torch.float64, device=dev)
        v_y2_loop = torch.zeros((n, n_loop, H), dtype=torch.float64, device=dev)

    k_xy_post = torch.zeros((n_post, H, D, D), dtype=torch.float64, device=dev)
    v_xy_post = torch.zeros((n_post, H, D, D), dtype=torch.float64, device=dev)
    k_x2_post = torch.zeros((n_post, H), dtype=torch.float64, device=dev)
    k_y2_post = torch.zeros((n_post, H), dtype=torch.float64, device=dev)
    v_x2_post = torch.zeros((n_post, H), dtype=torch.float64, device=dev)
    v_y2_post = torch.zeros((n_post, H), dtype=torch.float64, device=dev)

    total_toks = 0
    for idx, pids in enumerate(chosen):
        p = torch.tensor([pids], dtype=torch.long, device=dev)
        pm = torch.ones_like(p)
        _, cache_m = system._run_prefill_chunked(p, pm, num_passes=m, adapter_on=False, chunk_size=2048, return_logits=False)
        _, cache_n = system._run_prefill_chunked(p, pm, num_passes=n, adapter_on=False, chunk_size=2048, return_logits=False)
        total_toks += len(pids)

        for li, lid in enumerate(loop_layers):
            kv_m = _get_layer_kv_per_pass(cache_m, lid, m, is_loop=True, config=system.config)
            kv_n = _get_layer_kv_per_pass(cache_n, lid, n, is_loop=True, config=system.config)
            if args.method == "procrustes":
                for kp, j in enumerate(paired):
                    xm_k, xm_v = kv_m[j][0].double(), kv_m[j][1].double()  # [H, L, D]
                    yn_k, yn_v = kv_n[kp][0].double(), kv_n[kp][1].double()
                    k_xy_loop[kp, li] += torch.einsum("hti,htj->hij", xm_k, yn_k)
                    v_xy_loop[kp, li] += torch.einsum("hti,htj->hij", xm_v, yn_v)
                    k_x2_loop[kp, li] += (xm_k * xm_k).sum(dim=(1, 2))
                    k_y2_loop[kp, li] += (yn_k * yn_k).sum(dim=(1, 2))
                    v_x2_loop[kp, li] += (xm_v * xm_v).sum(dim=(1, 2))
                    v_y2_loop[kp, li] += (yn_v * yn_v).sum(dim=(1, 2))
            else:
                xm_k_cat = torch.cat([kv_m[p_i][0].double() for p_i in range(m)], dim=-1)  # [H, L, m*D]
                xm_v_cat = torch.cat([kv_m[p_i][1].double() for p_i in range(m)], dim=-1)
                for kp in range(n):
                    yn_k, yn_v = kv_n[kp][0].double(), kv_n[kp][1].double()
                    k_xx_loop[kp, li] += torch.einsum("hti,htj->hij", xm_k_cat, xm_k_cat)
                    k_xy_loop[kp, li] += torch.einsum("hti,htj->hij", xm_k_cat, yn_k)
                    k_y2_loop[kp, li] += (yn_k * yn_k).sum(dim=(1, 2))
                    v_xx_loop[kp, li] += torch.einsum("hti,htj->hij", xm_v_cat, xm_v_cat)
                    v_xy_loop[kp, li] += torch.einsum("hti,htj->hij", xm_v_cat, yn_v)
                    v_y2_loop[kp, li] += (yn_v * yn_v).sum(dim=(1, 2))

        for li, lid in enumerate(post_layers):
            km, vm = _get_layer_kv_per_pass(cache_m, lid, 1, is_loop=False, config=system.config)[0]
            kn, vn = _get_layer_kv_per_pass(cache_n, lid, 1, is_loop=False, config=system.config)[0]
            km, vm, kn, vn = km.double(), vm.double(), kn.double(), vn.double()
            k_xy_post[li] += torch.einsum("hti,htj->hij", km, kn)
            v_xy_post[li] += torch.einsum("hti,htj->hij", vm, vn)
            k_x2_post[li] += (km * km).sum(dim=(1, 2))
            k_y2_post[li] += (kn * kn).sum(dim=(1, 2))
            v_x2_post[li] += (vm * vm).sum(dim=(1, 2))
            v_y2_post[li] += (vn * vn).sum(dim=(1, 2))

        del cache_m, cache_n
        if (idx + 1) % 32 == 0 or (idx + 1) == len(chosen):
            print(f"[Procrustes-KV] Collected {idx+1}/{len(chosen)} prompts ({total_toks:,} tokens) in {perf_counter()-t0:.1f}s", flush=True)

    # Solve closed-form maps and pack into residual delta ParameterDict (I + Delta = W)
    max_m = sys_cfg.max_prefill_passes
    slot_offset = max_m - m
    eye_d = torch.eye(D, dtype=torch.float32).unsqueeze(0)

    kl = kv_maps.k_loop[str(n)].data.zero_()
    vl = kv_maps.v_loop[str(n)].data.zero_()
    kp_t = kv_maps.k_post[str(n)].data.zero_()
    vp_t = kv_maps.v_post[str(n)].data.zero_()

    for kp, j in enumerate(paired):
        for li in range(n_loop):
            if args.method == "procrustes":
                Wk = fit_procrustes_map(None, k_xy_loop[kp, li].cpu(), k_x2_loop[kp, li].cpu(), k_y2_loop[kp, li].cpu())
                Wv = fit_procrustes_map(None, v_xy_loop[kp, li].cpu(), v_x2_loop[kp, li].cpu(), v_y2_loop[kp, li].cpu())
                ps = slot_offset + j
                kl[kp, li, ps] = (Wk - eye_d).to(kl.device)
                vl[kp, li, ps] = (Wv - eye_d).to(vl.device)
            else:
                Wk_cat = fit_rescaled_ridge_cross(k_xx_loop[kp, li].cpu(), k_xy_loop[kp, li].cpu(), k_y2_loop[kp, li].cpu(), lam=args.ridge_lam)
                Wv_cat = fit_rescaled_ridge_cross(v_xx_loop[kp, li].cpu(), v_xy_loop[kp, li].cpu(), v_y2_loop[kp, li].cpu(), lam=args.ridge_lam)
                for p_i in range(m):
                    ps = slot_offset + p_i
                    wk_sub = Wk_cat[:, :, p_i * D : (p_i + 1) * D]
                    wv_sub = Wv_cat[:, :, p_i * D : (p_i + 1) * D]
                    if p_i == j:
                        wk_sub = wk_sub - eye_d
                        wv_sub = wv_sub - eye_d
                    kl[kp, li, ps] = wk_sub.to(kl.device)
                    vl[kp, li, ps] = wv_sub.to(vl.device)

    for li in range(n_post):
        Wk = fit_procrustes_map(None, k_xy_post[li].cpu(), k_x2_post[li].cpu(), k_y2_post[li].cpu())
        Wv = fit_procrustes_map(None, v_xy_post[li].cpu(), v_x2_post[li].cpu(), v_y2_post[li].cpu())
        kp_t[li] = (Wk - eye_d).to(kp_t.device)
        vp_t[li] = (Wv - eye_d).to(vp_t.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "system_config": sys_cfg.__dict__,
        "trainable_state": system.trainable_state_dict(),
        "meta": {
            "method": args.method,
            "m": m,
            "n": n,
            "pass_pairing": args.pass_pairing,
            "num_prompts": len(chosen),
            "total_tokens": total_toks,
            "wall_s": round(perf_counter() - t0, 2),
        },
    }
    torch.save(payload, args.output)
    print(f"[Procrustes-KV] SAVED {args.output} ({args.output.stat().st_size/1e6:.1f} MB) in {perf_counter()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()
