"""Native LoopBridge & Distilled Serving Plugin for Thinking Machines' Official Day-0 vLLM Engine.

Serves Inkling-2-Mini LoopBridge (m/n, e.g. 3/1 or 3/2), Training-Free (m/n), and param-matched
Distilled (n/n, e.g. 1/1 or 2/2) directly inside official Blackwell vLLM (`vllm.models.inkling2`),
keeping 100% of the 19 GB NVFP4/MXFP8 MoE experts shared zero-copy in VRAM and running full
Blackwell CUDA Graphs (`Inkling2FA4RelAttentionKernel` + `fused_moe_trtllm_sm100` + `INKLING_SCONV_SWA`).

Architecture:
  - Distilled (n/n): folds trained non-expert deltas directly into loaded vLLM weights at startup
    (0 extra VRAM, 0 runtime overhead, native `loop_num_passes=n` speed).
  - LoopBridge (m/n) & Training-Free (m/n):
      * Keeps frozen base non-expert weights for m-pass prefill (bit-exact Original 3/3 prefill).
      * Keeps static merged decode-side non-expert weight buffers (~480 MB total across layers 6..23)
        sharing 100% of the 25B NVFP4/MXFP8 MoE experts zero-copy.
      * Pure Prefill (`num_prefill_tokens > 0, num_decode_tokens == 0`): runs all m passes with frozen
        base weights, then folds Global-3 Cross-Pass KV Fusion maps (L11, L17, L23) ONCE into the
        prefilled PagedAttention KV cache slots (`slot_mapping`) of aligned decode pass(es) j = m-n+k.
      * Pure Decode (`num_prefill_tokens == 0`, including CUDA Graph capture & replay): runs ONLY n
        core passes reading from aligned KV/conv slots `j = m-n+k` using static decode-side buffers.

Activation:
  Set `LOOPBRIDGE_VLLM_CFG=/path/to/lb_vllm_cfg.json` (or run `python -m inkling.vllm_loopbridge serve ...`)
  and ensure `inkling` is importable (with the 2-line site hook active when `LOOPBRIDGE_VLLM_CFG` is set).
  For mathematically exact single-shot prompt KV folding (no chunk-2 seeing chunk-1's folded KV),
  launch LoopBridge servers with `--no-enable-chunked-prefill --no-enable-prefix-caching`.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch import nn


_INSTALLED = False


def _lora_delta(state: dict[str, torch.Tensor], prefix: str, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
    ka = f"{prefix}.lora_a"
    kb = f"{prefix}.lora_b"
    if ka not in state or kb not in state:
        return None
    a = state[ka].to(device=device, dtype=torch.float32)
    b = state[kb].to(device=device, dtype=torch.float32)
    return (b @ a).to(dtype)


class _AttnResShadow(nn.Module):
    """Lightweight nn.Module shadow holding merged decode-side AttnRes queries."""

    def __init__(self, attn_queries: list[torch.Tensor], mlp_queries: list[torch.Tensor], output_query: torch.Tensor):
        super().__init__()
        self.attn_queries = nn.ParameterList(nn.Parameter(t, requires_grad=False) for t in attn_queries)
        self.mlp_queries = nn.ParameterList(nn.Parameter(t, requires_grad=False) for t in mlp_queries)
        self.output_query = nn.Parameter(output_query, requires_grad=False)


def _apply_delta_to_vllm_model(
    model,
    trainable_state: dict[str, torch.Tensor],
    sys_cfg: dict,
    in_place: bool,
) -> dict:
    """Apply HF InklingLoopBridgeSystem trainable_state to a loaded vLLM Inkling2LoopModel.

    If `in_place=True` (Distilled n/n): mutates loaded base weights directly (0 extra VRAM).
    If `in_place=False` (LoopBridge m/n): attaches static `_lb_*_dec` GPU buffers (~480 MB total)
    so prefill reads frozen base weights and decode CUDA graphs read static decode buffers cleanly.
    """
    dev = model.embed_tokens.weight.device
    hidden = model.config.hidden_size
    scale = hidden**-0.5
    core = model.loop_core
    num_layers = len(model.layers)
    extra_bytes = 0

    # 1. AttnRes query deltas (attn_queries, mlp_queries, output_query)
    if not in_place:
        attn_q_dec = [p.detach().clone() for p in model.attn_res.attn_queries]
        mlp_q_dec = [p.detach().clone() for p in model.attn_res.mlp_queries]
        out_q_dec = model.attn_res.output_query.detach().clone()
    else:
        attn_q_dec = list(model.attn_res.attn_queries)
        mlp_q_dec = list(model.attn_res.mlp_queries)
        out_q_dec = model.attn_res.output_query

    for i in range(num_layers):
        k_aq = f"base.model.layers.{i}.attention_input.delta_query"
        if k_aq in trainable_state and i >= 1:
            dq = trainable_state[k_aq].to(device=dev, dtype=torch.float32).view_as(attn_q_dec[i - 1])
            attn_q_dec[i - 1].data.add_(dq * scale)
        k_mq = f"base.model.layers.{i}.mlp_input.delta_query"
        if k_mq in trainable_state:
            dq = trainable_state[k_mq].to(device=dev, dtype=torch.float32).view_as(mlp_q_dec[i])
            mlp_q_dec[i].data.add_(dq * scale)

    k_fq = "base.model.final_input.delta_query"
    if k_fq in trainable_state:
        dq = trainable_state[k_fq].to(device=dev, dtype=torch.float32).view_as(out_q_dec)
        out_q_dec.data.add_(dq * scale)

    if not in_place:
        model._lb_attn_res_dec = _AttnResShadow(attn_q_dec, mlp_q_dec, out_q_dec)
        extra_bytes += sum(t.numel() * t.element_size() for t in attn_q_dec + mlp_q_dec + [out_q_dec])

    # 2. Message queries delta (for n >= 2 decode)
    if hasattr(model, "message_queries"):
        msg_q_dec = model.message_queries if in_place else model.message_queries.detach().clone()
        k_msg = "base.model.message_input.delta_query"
        if k_msg in trainable_state:
            dq = trainable_state[k_msg].to(device=dev, dtype=torch.float32).view_as(msg_q_dec)
            msg_q_dec.data.add_(dq * scale)
        if not in_place:
            model.register_buffer("_lb_message_queries_dec", msg_q_dec, persistent=False)
            extra_bytes += msg_q_dec.numel() * msg_q_dec.element_size()

    # 3. Per-layer attention (qkvr, wo_ud), MoE router gate, and message controllers
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        qkvr_w = attn.qkvr.weight
        wo_w = attn.wo_ud.weight

        # Build qkvr delta across the 6 packed slices: [q, r, gate, sink, k, v]
        slice_prefixes = [
            f"base.model.layers.{i}.self_attn.q_proj.linear",
            f"base.model.layers.{i}.self_attn.relative_position_bias.linear",
            f"base.model.layers.{i}.self_attn.o_proj.gate_proj",
            f"base.model.layers.{i}.self_attn.sink_proj",
            f"base.model.layers.{i}.self_attn.k_proj.linear",
            f"base.model.layers.{i}.self_attn.v_proj.linear",
        ]
        deltas = [_lora_delta(trainable_state, pref, qkvr_w.dtype, dev) for pref in slice_prefixes]
        if any(d is not None for d in deltas):
            target_qkvr = qkvr_w if in_place else qkvr_w.detach().clone()
            row = 0
            for sz, d in zip(attn.split_sizes[:6], deltas):
                if d is not None:
                    assert d.shape == (sz, qkvr_w.shape[1]), (
                        f"Layer {i} qkvr slice shape mismatch: got {d.shape}, expected {(sz, qkvr_w.shape[1])}"
                    )
                    target_qkvr.data[row : row + sz].add_(d)
                row += sz
            if not in_place:
                attn.register_buffer("_lb_qkvr_dec", target_qkvr, persistent=False)
                extra_bytes += target_qkvr.numel() * target_qkvr.element_size()

        d_wo = _lora_delta(trainable_state, f"base.model.layers.{i}.self_attn.o_proj.linear", wo_w.dtype, dev)
        if d_wo is not None:
            target_wo = wo_w if in_place else wo_w.detach().clone()
            assert d_wo.shape == target_wo.shape, f"Layer {i} wo_ud mismatch: {d_wo.shape} vs {target_wo.shape}"
            target_wo.data.add_(d_wo)
            if not in_place:
                attn.register_buffer("_lb_wo_dec", target_wo, persistent=False)
                extra_bytes += target_wo.numel() * target_wo.element_size()

        # MoE router gate delta (fp32)
        k_gate = f"base.model.layers.{i}.mlp.gate.delta_weight"
        if k_gate in trainable_state:
            dg = trainable_state[k_gate].to(device=dev, dtype=layer.mlp.gate.weight.dtype)
            target_gate = layer.mlp.gate.weight if in_place else layer.mlp.gate.weight.detach().clone()
            assert dg.shape == target_gate.shape, f"Layer {i} router gate mismatch: {dg.shape} vs {target_gate.shape}"
            target_gate.data.add_(dg)
            if not in_place:
                layer.mlp._lb_gate_weight_dec = nn.Parameter(target_gate, requires_grad=False)
                extra_bytes += target_gate.numel() * target_gate.element_size()

        # Message controller down_proj LoRAs (core layers only, for n >= 2)
        if i in core and hasattr(layer, "controller_down"):
            mg_names = ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate")
            mg_deltas = [
                _lora_delta(
                    trainable_state,
                    f"base.model.layers.{i}.{gname}.down_proj",
                    layer.controller_down.dtype,
                    dev,
                )
                for gname in mg_names
            ]
            if any(d is not None for d in mg_deltas):
                target_cd = layer.controller_down if in_place else layer.controller_down.detach().clone()
                for s_idx, d in enumerate(mg_deltas):
                    if d is not None:
                        target_cd.data[s_idx].add_(d)
                if not in_place:
                    layer.register_buffer("_lb_controller_down_dec", target_cd, persistent=False)
                    extra_bytes += target_cd.numel() * target_cd.element_size()

    # 4. Prompt KV maps (Cross-Pass or Single-Pass, Global-3 or All-18) for LoopBridge
    kv_bundle = None
    if not in_place and any(k.startswith("kv_maps.") for k in trainable_state):
        kv_map_layers = sys_cfg.get("kv_map_layers", "all")
        cross_pass = bool(sys_cfg.get("kv_map_cross_pass", False))
        local_ids = set(model.config.local_layer_ids)
        if kv_map_layers == "global":
            loop_layers = [l for l in core if l not in local_ids]
            post_layers = [l for l in range(core.stop, num_layers) if l not in local_ids]
        else:
            loop_layers = list(core)
            post_layers = list(range(core.stop, num_layers))

        kv_bundle = {
            "cross_pass": cross_pass,
            "max_prefill_passes": int(model.config.num_loop_passes),
            "loop_layers": loop_layers,
            "post_layers": post_layers,
            "k_loop": {},
            "v_loop": {},
            "k_post": {},
            "v_post": {},
        }
        for key, val in trainable_state.items():
            if not key.startswith("kv_maps."):
                continue
            parts = key.split(".")
            # kv_maps.{k_loop,v_loop,k_post,v_post}.{n}
            kind, n_str = parts[1], parts[2]
            t = val.to(device=dev, dtype=torch.float32)
            kv_bundle[kind][int(n_str)] = t
            extra_bytes += t.numel() * t.element_size()

    return {"extra_mb": round(extra_bytes / (1024 * 1024), 2), "kv_bundle": kv_bundle}


@torch.no_grad()
def _fold_prefill_kv_maps(
    model,
    kv_bundle: dict,
    m: int,
    n: int,
    attn_metadata: dict,
) -> None:
    """Apply LoopBridge prompt KV maps ONCE in-place to prefilled PagedAttention KV slots."""
    if kv_bundle is None or n not in kv_bundle["k_loop"]:
        return

    cross_pass = kv_bundle["cross_pass"]
    max_m = kv_bundle["max_prefill_passes"]
    kd_loop = kv_bundle["k_loop"][n]
    vd_loop = kv_bundle["v_loop"][n]
    kd_post = kv_bundle["k_post"][n]
    vd_post = kv_bundle["v_post"][n]

    # 1. Core loop layers (e.g. L11, L17 for Global-3)
    for idx, layer_id in enumerate(kv_bundle["loop_layers"]):
        attn = model.layers[layer_id].attn
        p0_name, p0_cache = attn._pass_cache(0)
        if p0_cache.numel() == 0 or p0_name not in attn_metadata:
            continue
        md = attn_metadata[p0_name]
        n_dec = getattr(md, "num_decode_tokens", 0)
        n_pre = getattr(md, "num_prefill_tokens", 0)
        if n_pre <= 0:
            continue
        slots = md.slot_mapping[n_dec : n_dec + n_pre]
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            continue

        # Gather prefill KV across needed passes BEFORE mutating any target pass cache
        k_caches = []
        v_caches = []
        block_size = None
        for p in range(m):
            _, pc = attn._pass_cache(p)
            kc, vc = attn._split_kv_cache(pc)  # [num_blocks, block_size, H_kv, D]
            block_size = kc.shape[1]
            k_caches.append(kc)
            v_caches.append(vc)

        blk = torch.div(slots, block_size, rounding_mode="floor")
        off = slots - blk * block_size

        if cross_pass:
            k_src = [k_caches[p][blk, off].float() for p in range(m)]  # each [T, H_kv, D]
            v_src = [v_caches[p][blk, off].float() for p in range(m)]
            slot_offset = max_m - m
            out_pairs = []
            for k_pass in range(n):
                j = min(max(m - n + k_pass, 0), m - 1)
                k_out = k_src[j].clone()
                v_out = v_src[j].clone()
                for p in range(m):
                    ps = max(0, min(max_m - 1, slot_offset + p))
                    k_out.add_(torch.einsum("nhi,hoi->nho", k_src[p], kd_loop[k_pass, idx, ps]))
                    v_out.add_(torch.einsum("nhi,hoi->nho", v_src[p], vd_loop[k_pass, idx, ps]))
                out_pairs.append((j, k_out, v_out))
            for j, k_out, v_out in out_pairs:
                k_caches[j][blk, off] = k_out.to(k_caches[j].dtype)
                v_caches[j][blk, off] = v_out.to(v_caches[j].dtype)
        else:
            for k_pass in range(n):
                j = min(max(m - n + k_pass, 0), m - 1)
                ks = k_caches[j][blk, off].float()
                vs = v_caches[j][blk, off].float()
                k_out = ks + torch.einsum("nhi,hoi->nho", ks, kd_loop[k_pass, idx])
                v_out = vs + torch.einsum("nhi,hoi->nho", vs, vd_loop[k_pass, idx])
                k_caches[j][blk, off] = k_out.to(k_caches[j].dtype)
                v_caches[j][blk, off] = v_out.to(v_caches[j].dtype)

    # 2. Post-loop coda layers (e.g. L23 for Global-3)
    for idx, layer_id in enumerate(kv_bundle["post_layers"]):
        attn = model.layers[layer_id].attn
        p0_name, p0_cache = attn._pass_cache(0)
        if p0_cache.numel() == 0 or p0_name not in attn_metadata:
            continue
        md = attn_metadata[p0_name]
        n_dec = getattr(md, "num_decode_tokens", 0)
        n_pre = getattr(md, "num_prefill_tokens", 0)
        if n_pre <= 0:
            continue
        slots = md.slot_mapping[n_dec : n_dec + n_pre]
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            continue
        kc, vc = attn._split_kv_cache(p0_cache)
        block_size = kc.shape[1]
        blk = torch.div(slots, block_size, rounding_mode="floor")
        off = slots - blk * block_size
        ks = kc[blk, off].float()
        vs = vc[blk, off].float()
        k_out = ks + torch.einsum("nhi,hoi->nho", ks, kd_post[idx])
        v_out = vs + torch.einsum("nhi,hoi->nho", vs, vd_post[idx])
        kc[blk, off] = k_out.to(kc.dtype)
        vc[blk, off] = v_out.to(vc.dtype)


def install_vllm_patch(cfg_path: str | Path | None = None) -> None:
    """Install the LoopBridge / Distilled monkey-patch into vLLM's Inkling2Loop classes."""
    global _INSTALLED
    if _INSTALLED:
        return
    cfg_str = str(cfg_path) if cfg_path is not None else os.environ.get("LOOPBRIDGE_VLLM_CFG", "")
    if not cfg_str:
        return
    cfg = json.loads(Path(cfg_str).read_text())
    mode = cfg["mode"]  # "distilled", "training_free", "loopbridge"
    m = int(cfg.get("m", 3))
    n = int(cfg.get("n", 1))
    ckpt_path = cfg.get("checkpoint")

    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.forward_context import get_forward_context
    from vllm.models.inkling2.nvidia import attention as ink_attn_mod
    from vllm.models.inkling2.nvidia import loop as ink_loop_mod
    from vllm.models.inkling2.nvidia import moe as ink_moe_mod
    from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
    from vllm.models.inkling2.nvidia.ops.linear import linear
    from vllm.models.inkling2.nvidia.ops.ll_gemm_sm120 import (
        qkv_bmm,
        sm12x_bf16_ll_gemm,
        use_qkv_bmm,
        use_sm12x_bf16_ll_gemm,
    )
    from vllm.models.inkling2.nvidia.ops.loop import blend_rms_norm, pair_rms_norm
    from vllm.models.inkling2.nvidia.ops.sconv import conv_metadata

    # --- 1. Patch process_weights_after_loading to load checkpoint deltas ---
    orig_process = ink_loop_mod.Inkling2LoopModel.process_weights_after_loading

    def _patched_process_weights_after_loading(self) -> None:
        orig_process(self)
        self._lb_mode = mode
        self._lb_m = m
        self._lb_n = n
        self._lb_kv_bundle = None
        if ckpt_path:
            saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = saved["trainable_state"]
            sys_cfg = saved.get("system_config", {})
            in_place = mode == "distilled"
            info = _apply_delta_to_vllm_model(self, state, sys_cfg, in_place=in_place)
            self._lb_kv_bundle = info["kv_bundle"]
            print(
                f"[LoopBridge-vLLM] Loaded {mode} ({m}/{n}) from {ckpt_path} "
                f"(in_place={in_place}, extra_vram={info['extra_mb']} MB, "
                f"kv_maps={info['kv_bundle'] is not None})",
                flush=True,
            )
        else:
            print(f"[LoopBridge-vLLM] Active {mode} ({m}/{n}) without checkpoint", flush=True)

    ink_loop_mod.Inkling2LoopModel.process_weights_after_loading = _patched_process_weights_after_loading

    # For Distilled mode, in-place weight folding is all we need!
    if mode == "distilled":
        _INSTALLED = True
        return

    # --- 2. For LoopBridge / Training-Free: patch Attention, MoE, Blend, and LoopModel.forward ---
    # Thread-safe module-level flag read during forward (static during CUDA graph capture!)
    _state = {"decode_mode": False}

    orig_attn_project = ink_attn_mod.Inkling2Attention._project
    orig_attn_forward = ink_attn_mod.Inkling2Attention.forward

    def _patched_attn_project(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor):
        if not _state["decode_mode"] or not hasattr(self, "_lb_qkvr_dec"):
            return orig_attn_project(self, x_q, x_k, x_v)
        weight = self._lb_qkvr_dec
        if x_k is x_q and x_v is x_q:
            projected = linear(x_q, weight)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_side, kv_size = sum(self.split_sizes[:4]), self.split_sizes[4]
        k_end = q_side + kv_size
        if use_qkv_bmm(x_q, x_k, x_v, weight):
            projected = qkv_bmm(x_q, x_k, x_v, weight, q_side, k_end)
            return list(projected.split(self.split_sizes, dim=-1)[:6])
        q_proj_w = weight[: sum(self.q_side_splits)]
        projected = linear(x_q, q_proj_w)
        k = linear(x_k, weight[q_side:k_end])
        v = linear(x_v, weight[k_end : k_end + kv_size])
        return [*projected.split(self.q_side_splits, dim=-1)[:4], k, v]

    def _patched_attn_forward(
        self,
        positions: torch.Tensor,
        x_q: torch.Tensor,
        x_k: torch.Tensor | None = None,
        x_v: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        pass_idx: int = 0,
    ) -> torch.Tensor:
        if not _state["decode_mode"] or not hasattr(self, "_lb_wo_dec"):
            return orig_attn_forward(self, positions, x_q, x_k, x_v, tau, out, pass_idx)
        if self.is_local and x_k is None:
            x_k = x_v = x_q
        assert x_k is not None and x_v is not None
        q, r, gate, sinks, k, v = self._project(x_q, x_k, x_v)
        output = torch.empty(gate.shape, dtype=q.dtype, device=q.device)
        self._attention(
            positions,
            q,
            r,
            gate,
            sinks,
            k,
            v,
            self.k_sconv.dynamic_taps(x_k),
            self.v_sconv.dynamic_taps(x_v),
            tau,
            output,
            pass_idx,
        )
        return linear(output, self._lb_wo_dec, out)

    ink_attn_mod.Inkling2Attention._project = _patched_attn_project
    ink_attn_mod.Inkling2Attention.forward = _patched_attn_forward

    # Patch MoE gate routing weight selection in decode mode (100% static pointer for CUDA graphs!)
    orig_moe_forward = ink_moe_mod.Inkling2MoE.forward

    def _patched_moe_forward(self, x: torch.Tensor, reduce: bool = True, pass_idx: int = 0) -> torch.Tensor:
        if not _state["decode_mode"] or not hasattr(self, "_lb_gate_weight_dec"):
            return orig_moe_forward(self, x, reduce=reduce, pass_idx=pass_idx)
        saved_w = self.gate.weight
        self.gate.weight = self._lb_gate_weight_dec
        try:
            return orig_moe_forward(self, x, reduce=reduce, pass_idx=pass_idx)
        finally:
            self.gate.weight = saved_w

    ink_moe_mod.Inkling2MoE.forward = _patched_moe_forward

    # Patch LoopDecoderLayer.blend for message controller decode weights
    orig_blend = ink_loop_mod.Inkling2LoopDecoderLayer.blend

    def _patched_blend(self, streams, dest, message, norm_weight, norm_eps):
        if not _state["decode_mode"] or not hasattr(self, "_lb_controller_down_dec"):
            return orig_blend(self, streams, dest, message, norm_weight, norm_eps)
        pair = pair_rms_norm(dest, message)
        down = self._lb_controller_down_dec[streams]
        up = self.controller_gain_forget[streams]
        if use_sm12x_bf16_ll_gemm(pair, down):
            hidden = sm12x_bf16_ll_gemm(pair, down)
            gain_forget = sm12x_bf16_ll_gemm(hidden, up, out_dtype=torch.float32)
        else:
            hidden = torch.bmm(pair, down.transpose(1, 2))
            gain_forget = torch.bmm(hidden, up.transpose(1, 2), out_dtype=torch.float32)
        return blend_rms_norm(dest, message, gain_forget, norm_weight, norm_eps)

    ink_loop_mod.Inkling2LoopDecoderLayer.blend = _patched_blend

    # Patch Inkling2LoopModel.forward to dispatch Pure-Decode (n passes + decode weights) vs Prefill (m passes + KV fold)
    orig_loop_forward = ink_loop_mod.Inkling2LoopModel.forward

    def _is_pure_decode_batch(self, attn_metadata) -> tuple[bool, bool]:
        """Return (is_pure_decode, has_prefill)."""
        if not isinstance(attn_metadata, dict):
            return False, False
        layer0_name = self.layers[0].attn.prefix
        md = attn_metadata.get(layer0_name)
        if md is None:
            return False, False
        n_dec = getattr(md, "num_decode_tokens", 0)
        n_pre = getattr(md, "num_prefill_tokens", 0)
        return (n_pre == 0 and n_dec > 0), (n_pre > 0)

    def _patched_loop_forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        attn_metadata = get_forward_context().attn_metadata
        is_pure_decode, has_prefill = _is_pure_decode_batch(self, attn_metadata)

        if not is_pure_decode:
            # Prefill (or dummy warmup with attn_metadata=None): frozen base m-pass forward
            _state["decode_mode"] = False
            out = orig_loop_forward(self, positions, inputs_embeds)
            if has_prefill and self._lb_kv_bundle is not None:
                _fold_prefill_kv_maps(self, self._lb_kv_bundle, self._lb_m, self._lb_n, attn_metadata)
            return out

        # --- PURE DECODE PATH (CUDA Graph captured & replayed at native n-pass speed!) ---
        _state["decode_mode"] = True
        try:
            num_tokens, hidden = inputs_embeds.shape
            core = self.loop_core
            lb_m = self._lb_m
            lb_n = self._lb_n
            stack = inputs_embeds.new_empty((self._max_loop_blocks() + 1, num_tokens, hidden))
            stack[0].copy_(inputs_embeds)
            tau = self._log_scaling_tau(positions)
            workspace = self.allreduce_workspace
            plan = [(False, False)] * 2
            if workspace is not None and conv_metadata(self.layers[0].conv_state) is not None:
                plan = workspace.plan(num_tokens)

            # Aligned pass indices: decode pass k in {0..n-1} reads KV/conv slot j = clip(m - n + k, 0, m - 1)
            aligned_passes = [min(max(lb_m - lb_n + k, 0), lb_m - 1) for k in range(lb_n)]
            # (layer_id, kv_conv_pass_idx, is_looped_pass)
            visits = [(i, 0, False) for i in range(core.start)]
            for k, p_slot in enumerate(aligned_passes):
                visits += [(i, p_slot, k > 0) for i in core]
            visits += [(i, 0, False) for i in range(core.stop, len(self.layers))]

            # Swap AttnRes / message_queries references for decode (static GPU buffers)
            orig_ar = self.attn_res
            orig_msgq = self._buffers["message_queries"]
            if hasattr(self, "_lb_attn_res_dec"):
                self._modules["attn_res"] = self._lb_attn_res_dec
            if "_lb_message_queries_dec" in self._buffers:
                self._buffers["message_queries"] = self._buffers["_lb_message_queries_dec"]

            try:
                aux_hidden_states = []
                num_blocks = 0
                inputs = bank = prelude_blocks = prelude_partial = prelude_dests = None
                for visit, (i, pass_idx, looped) in enumerate(visits):
                    if i == core.start and not looped:
                        prelude_blocks = num_blocks
                        prelude_partial = stack[num_blocks].clone()
                    elif i == core.start:
                        assert prelude_blocks is not None and prelude_partial is not None
                        bank = stack[prelude_blocks + 1 : num_blocks + 1].clone()
                        num_blocks = prelude_blocks
                        stack[num_blocks].copy_(prelude_partial)
                        if prelude_dests is None:
                            query = self.attn_res.attn_queries[i - 1]
                            prelude_dests = attn_res(
                                stack[:num_blocks],
                                prelude_partial,
                                None,
                                query,
                                None,
                                0.0,
                                True,
                            )
                        inputs = prelude_dests
                    if visit in self.aux_hidden_state_layers:
                        aux_hidden_states.append(stack[num_blocks].clone())
                    if visit + 1 == len(visits):
                        next_query, next_norm = self.attn_res.output_query, self.norm
                    else:
                        next_i, _, next_looped = visits[visit + 1]
                        next_query = self.attn_res.attn_queries[next_i - 1]
                        next_norm = None if next_looped else self.layers[next_i].attn_norm
                    num_blocks, inputs = self._run_layer(
                        i,
                        pass_idx,
                        stack,
                        num_blocks,
                        inputs,
                        bank if looped else None,
                        next_query,
                        next_norm,
                        positions,
                        tau,
                        plan,
                    )
                assert inputs is not None
                out = inputs[0]
                if self.aux_hidden_state_layers:
                    return out, [*aux_hidden_states, out]
                return out
            finally:
                self._modules["attn_res"] = orig_ar
                self._buffers["message_queries"] = orig_msgq
        finally:
            _state["decode_mode"] = False

    ink_loop_mod.Inkling2LoopModel.forward = _patched_loop_forward
    _INSTALLED = True


def ensure_pth_hook(venv_site_packages: Path, repo_root: Path) -> Path:
    """Install a zero-overhead .pth hook in vLLM's venv that activates ONLY when LOOPBRIDGE_VLLM_CFG is set."""
    pth = venv_site_packages / "zzz_loopbridge_vllm.pth"
    code = (
        f"import os, sys; "
        f"(sys.path.insert(0, {str(repo_root)!r}) if {str(repo_root)!r} not in sys.path else None); "
        f"(__import__('inkling.vllm_loopbridge', fromlist=['install_vllm_patch']).install_vllm_patch() "
        f"if os.environ.get('LOOPBRIDGE_VLLM_CFG') else None)\n"
    )
    pth.write_text(code)
    return pth


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_hook = sub.add_parser("install-hook", help="Install conditional .pth hook into vLLM venv")
    p_hook.add_argument("--venv", type=Path, default=Path("/dev/shm/vllm-venv"))
    p_hook.add_argument("--repo", type=Path, default=Path("/dev/shm/inkling/code"))

    p_srv = sub.add_parser("serve", help="Launch official vLLM server with LoopBridge/Distilled active")
    p_srv.add_argument("--model", required=True)
    p_srv.add_argument("--mode", choices=("training_free", "loopbridge", "distilled"), required=True)
    p_srv.add_argument("--m", type=int, default=3)
    p_srv.add_argument("--n", type=int, default=1)
    p_srv.add_argument("--checkpoint", default=None)
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--max-model-len", type=int, default=196608)
    p_srv.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p_srv.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args()
    if args.cmd == "install-hook":
        sp = next(args.venv.glob("lib/python*/site-packages"))
        pth = ensure_pth_hook(sp, args.repo)
        print(f"Installed hook: {pth}")
        return

    cfg = {
        "mode": args.mode,
        "m": args.m,
        "n": args.n,
        "checkpoint": args.checkpoint,
    }
    cfg_file = Path(tempfile.gettempdir()) / f"lb_vllm_{args.mode}_{args.m}_{args.n}_{args.port}.json"
    cfg_file.write_text(json.dumps(cfg, indent=2) + "\n")

    env = os.environ.copy()
    env["LOOPBRIDGE_VLLM_CFG"] = str(cfg_file)

    vllm_bin = str(Path(sys.executable).parent / "vllm")
    cmd = [
        vllm_bin,
        "serve",
        args.model,
        "--tensor-parallel-size",
        "1",
        "--port",
        str(args.port),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.mode == "distilled":
        overrides = json.dumps(
            {
                "text_config": {
                    "loop_config": {
                        "num_passes": 3,
                        "core_start": 6,
                        "core_end": 18,
                        "gate_rank": 64,
                        "is_varloop": True,
                    },
                    "loop_num_passes": args.n,
                }
            }
        )
        cmd += ["--hf-overrides", overrides]
    else:
        # Pure prefill (1 shot) + pure decode (full CUDA graphs) -> 100% exact post-prefill KV fold!
        cmd += [
            "--no-enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--max-num-batched-tokens",
            str(min(args.max_model_len, 131072)),
        ]
    if args.extra_args:
        cmd += args.extra_args

    print(f"[LoopBridge-vLLM] Exec: {' '.join(cmd)}", flush=True)
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
