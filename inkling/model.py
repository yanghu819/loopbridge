"""LoopBridge adapter, matched tied-loop distillation control, and fast runner for Inkling-2-Mini.

Inkling-2-Mini architecture (thinkingmachines/Inkling-2-Mini):
  - 24 physical layers: pre-loop 0..5 (6 layers), looped core 6..17 (12 layers x 3 passes), post-loop 18..23 (6 layers)
  - 48 effective layer applications at native 3/3; 36 at 2/2; 24 at 1/1
  - 128-expert Sparse MoE (top-8), hybrid 20 sliding (window 1024) + 4 full attention (layers 5, 11, 17, 23)
  - NoPE (no rotary embeddings) + hidden-conditioned relative position bias + attention sinks
  - 4 short causal convolutions (kernel 4) per layer (k_sconv, v_sconv, post_attn_conv, post_mlp_conv)
  - Block-AttnRes residuals + message-gated inter-pass communication

Because Inkling-2-Mini uses NoPE (no RoPE), per-(pass, layer, KV-head) linear prompt KV maps
M^K = I + Delta^K and M^V = I + Delta^V fold ONCE into the prompt KV cache right after prefill
with zero decode-step overhead, alongside aligned boundary states for all four short convolutions.
Decoder-only LoRA, MoE router deltas, and Block-AttnRes query deltas also fold into base weights
at deployment for zero decode-step parameter overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from time import perf_counter
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Inkling2LoopForCausalLM
from transformers.cache_utils import (
    DynamicCache,
    LinearAttentionAndFullAttentionLayer,
    LinearAttentionAndSlidingWindowAttentionLayer,
)


def load_inkling_model(
    model_path: str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Inkling2LoopForCausalLM:
    """Load text-only Inkling2LoopForCausalLM on a single device."""
    dev = torch.device(device)
    model = Inkling2LoopForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map={"": dev},
        local_files_only=True,
    )
    model.eval()
    return model


def set_loop_passes(model: Inkling2LoopForCausalLM, num_passes: int) -> None:
    model.config.loop_num_passes = num_passes
    model.model.config.loop_num_passes = num_passes


def cache_slot(loop_pass: int, layer_idx: int, config) -> int:
    """Map (loop_pass, physical_layer_idx) to its slot index in Inkling2Loop DynamicCache."""
    loop_start = config.loop_layer_index_start
    loop_end = config.loop_layer_index_end
    if layer_idx < loop_start or layer_idx >= loop_end or loop_pass == 0:
        return layer_idx
    num_loop = loop_end - loop_start
    return config.num_hidden_layers + (loop_pass - 1) * num_loop + (layer_idx - loop_start)


def compute_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """True per-row position IDs for left-padded sequences (exact for LogScale beyond 8192)."""
    pos = attention_mask.long().cumsum(dim=-1) - 1
    return pos.clamp(min=0)


class ToggleLoRALinear(nn.Module):
    """Rank-r, alpha-r (scale 1.0) zero-init LoRA toggled by a shared state['on'] flag."""

    def __init__(self, base: nn.Linear, rank: int, state: dict):
        super().__init__()
        self.base = base
        self.rank = rank
        self.state = state
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.state["on"]:
            return out
        delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b)
        return out + delta.to(out.dtype)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        self.base.weight.data.add_((self.lora_b.float() @ self.lora_a.float()).to(self.base.weight.dtype))
        return self.base


class ToggleRouterDelta(nn.Module):
    """Zero-init fp32 weight delta on Inkling2LoopTopKRouter, active only when state['on']."""

    def __init__(self, router: nn.Module, state: dict):
        super().__init__()
        self.router = router
        self.state = state
        self.delta_weight = nn.Parameter(torch.zeros_like(router.weight, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor):
        if not self.state["on"]:
            return self.router(hidden_states)
        r = self.router
        flat = hidden_states.float().reshape(-1, r.hidden_dim)
        w = r.weight + self.delta_weight
        logits = F.linear(flat, w)
        scores = r.score_fn(logits)
        indices = torch.topk(scores + r.e_score_correction_bias, r.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * r.routed_scaling_factor, indices

    @torch.no_grad()
    def merged_router(self) -> nn.Module:
        self.router.weight.data.add_(self.delta_weight.to(self.router.weight.dtype))
        return self.router


class ToggleQueryDelta(nn.Module):
    """Zero-init fp32 query delta on BlockAttentionResidual or MessageAttentionResidual."""

    def __init__(self, mod: nn.Module, state: dict):
        super().__init__()
        self.mod = mod
        self.state = state
        self.delta_query = nn.Parameter(torch.zeros_like(mod.query, dtype=torch.float32))

    def forward(self, block_states: torch.Tensor, partial_block_states: torch.Tensor, *args):
        if not self.state["on"]:
            return self.mod(block_states, partial_block_states, *args)
        m = self.mod
        stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
        stacked_states = m.norm(stacked_states)
        hidden_size = stacked_states.shape[-1]
        q_eff = m.query.float() + self.delta_query
        if args:
            # Inkling2LoopMessageAttentionResidual(block_states, partial_block_states, loop_layer_idx)
            loop_layer_idx = args[0]
            query_states = q_eff[loop_layer_idx].expand(*stacked_states.shape[:-2], -1, -1)
            scores = torch.matmul(query_states, stacked_states.transpose(-1, -2)) * (hidden_size**-0.5)
            probs = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(probs, stacked_states)
            return attn_output
        # Inkling2LoopBlockAttentionResidual
        query_states = q_eff.view(-1, hidden_size).expand(*stacked_states.shape[:-2], -1, -1)
        scores = torch.matmul(query_states, stacked_states.transpose(-1, -2)) * (hidden_size**-0.5)
        probs = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(probs, stacked_states)
        return attn_output.unbind(-2)

    @torch.no_grad()
    def merged_module(self) -> nn.Module:
        self.mod.query.data.add_(self.delta_query.to(self.mod.query.dtype))
        return self.mod


class InklingKVMapAdapter(nn.Module):
    """Identity-initialized per-(decode_depth, pass, layer, KV-head) linear prompt KV maps.

    Because Inkling-2-Mini is NoPE (additive relative position bias + attention sinks),
    full (head_dim x head_dim) linear key and value maps M^K = I + Delta^K, M^V = I + Delta^V
    fold once into the prompt KV cache right after prefill with zero decode-step overhead.

    Supports:
      - kv_map_layers="global" : ONLY full-attention layers inside/after the loop (L11, L17, L23),
                                 cutting KV-map fold latency by 6x.
      - kv_map_layers="all"    : all 12 looped + 6 post-loop layers (legacy / All-18 ablation).
      - kv_map_cross_pass=True : Cross-Pass Global KV Fusion — for each decode pass k, linearly
                                 fuses ALL m prefill passes' KV caches at the mapped core layers
                                 (e.g., Pass 0 surface-copying keys/values + Pass 1/2 semantic
                                 keys/values) into decode pass k's single KV cache slot:
                                     K_bridge[k] = K_pre[j(k)] + sum_{p=0}^{m-1} Delta^K_{k,p} K_pre[p]
                                 Exact zero-init identity at step 0; zero extra decode memory/FLOPs.
    """

    def __init__(
        self,
        config,
        device: torch.device,
        decode_depths: tuple[int, ...] = (1,),
        kv_map_layers: str = "global",
        cross_pass: bool = False,
        max_prefill_passes: int | None = None,
    ):
        super().__init__()
        assert kv_map_layers in ("all", "global"), f"Unsupported kv_map_layers={kv_map_layers}"
        self.config = config
        self.decode_depths = tuple(sorted(set(decode_depths)))
        self.kv_map_layers = kv_map_layers
        self.cross_pass = cross_pass
        self.max_prefill_passes = (
            int(max_prefill_passes) if max_prefill_passes is not None else int(getattr(config, "loop_num_passes", 3))
        )
        self.loop_start = config.loop_layer_index_start
        self.loop_end = config.loop_layer_index_end
        layer_types = getattr(config, "layer_types", None)

        if kv_map_layers == "global":
            if layer_types is None:
                raise ValueError("kv_map_layers='global' requires config.layer_types")
            # Inkling-2-Mini uses 'hybrid' for full global layers (L5, L11, L17, L23) and 'hybrid_sliding' for SWA.
            self.loop_offsets = tuple(
                l - self.loop_start
                for l in range(self.loop_start, self.loop_end)
                if layer_types[l] in ("hybrid", "full_attention")
            )
            self.post_offsets = tuple(
                l - self.loop_end
                for l in range(self.loop_end, config.num_hidden_layers)
                if layer_types[l] in ("hybrid", "full_attention")
            )
            assert self.loop_offsets and self.post_offsets, (
                f"Expected global layers inside and after loop, got loop={self.loop_offsets}, "
                f"post={self.post_offsets} from layer_types={layer_types}"
            )
        else:
            self.loop_offsets = tuple(range(self.loop_end - self.loop_start))
            self.post_offsets = tuple(range(config.num_hidden_layers - self.loop_end))

        self.loop_offset_to_idx = {off: idx for idx, off in enumerate(self.loop_offsets)}
        self.post_offset_to_idx = {off: idx for idx, off in enumerate(self.post_offsets)}
        self.num_loop = len(self.loop_offsets)
        self.num_post = len(self.post_offsets)
        hkv = config.num_key_value_heads
        hd = config.head_dim

        self.k_loop = nn.ParameterDict()
        self.v_loop = nn.ParameterDict()
        self.k_post = nn.ParameterDict()
        self.v_post = nn.ParameterDict()

        for n in self.decode_depths:
            if cross_pass:
                loop_shape = (n, self.num_loop, self.max_prefill_passes, hkv, hd, hd)
            else:
                loop_shape = (n, self.num_loop, hkv, hd, hd)
            self.k_loop[str(n)] = nn.Parameter(torch.zeros(*loop_shape, device=device, dtype=torch.float32))
            self.v_loop[str(n)] = nn.Parameter(torch.zeros(*loop_shape, device=device, dtype=torch.float32))
            self.k_post[str(n)] = nn.Parameter(
                torch.zeros(self.num_post, hkv, hd, hd, device=device, dtype=torch.float32)
            )
            self.v_post[str(n)] = nn.Parameter(
                torch.zeros(self.num_post, hkv, hd, hd, device=device, dtype=torch.float32)
            )

    @staticmethod
    def _apply_linear(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        # x: [B, H_kv, L, D], delta: [H_kv, D, D] -> (I + delta) x
        res = torch.einsum("bhti,hoi->bhto", x.float(), delta)
        return (x.float() + res).to(x.dtype)

    def map_loop(self, n: int, k: int, loop_layer_offset: int, keys: torch.Tensor, values: torch.Tensor):
        idx = self.loop_offset_to_idx.get(loop_layer_offset)
        if idx is None:
            return keys, values
        kd = self.k_loop[str(n)][k, idx]
        vd = self.v_loop[str(n)][k, idx]
        return self._apply_linear(keys, kd), self._apply_linear(values, vd)

    def map_loop_cross(
        self,
        n: int,
        k: int,
        base_pass: int,
        loop_layer_offset: int,
        keys_all: list[torch.Tensor],
        values_all: list[torch.Tensor],
    ):
        idx = self.loop_offset_to_idx.get(loop_layer_offset)
        if idx is None:
            return keys_all[base_pass], values_all[base_pass]
        m_cur = len(keys_all)
        dtype = keys_all[base_pass].dtype
        k_out = keys_all[base_pass].float()
        v_out = values_all[base_pass].float()
        kd_all = self.k_loop[str(n)][k, idx]  # [max_m, H_kv, D, D]
        vd_all = self.v_loop[str(n)][k, idx]
        slot_offset = self.max_prefill_passes - m_cur
        for p in range(m_cur):
            slot = max(0, min(self.max_prefill_passes - 1, slot_offset + p))
            k_out = k_out + torch.einsum("bhti,hoi->bhto", keys_all[p].float(), kd_all[slot])
            v_out = v_out + torch.einsum("bhti,hoi->bhto", values_all[p].float(), vd_all[slot])
        return k_out.to(dtype), v_out.to(dtype)

    def map_post(self, n: int, post_layer_offset: int, keys: torch.Tensor, values: torch.Tensor):
        idx = self.post_offset_to_idx.get(post_layer_offset)
        if idx is None:
            return keys, values
        kd = self.k_post[str(n)][idx]
        vd = self.v_post[str(n)][idx]
        return self._apply_linear(keys, kd), self._apply_linear(values, vd)


def _train_update_conv_state(self, conv_states: torch.Tensor, state_idx: int = 0, conv_kernel_size: int | None = None, **kwargs):
    """Autograd-safe update_conv_state for single-pass parallel teacher-forcing decode."""
    if not self.is_conv_states_initialized[state_idx]:
        self.lazy_initialization(conv_states=conv_states, state_idx=state_idx, conv_kernel_size=conv_kernel_size)
    if not self.has_previous_state[state_idx]:
        full_conv_states = conv_states
        self.has_previous_state[state_idx] = True
        if not self.record_past and full_conv_states.shape[-1] < self.conv_kernel_size[state_idx]:
            pad_len = self.conv_kernel_size[state_idx] - full_conv_states.shape[-1]
            full_conv_states = F.pad(full_conv_states, (pad_len, 0), value=0)
    else:
        full_conv_states = torch.cat([self.conv_states[state_idx], conv_states], dim=-1)
    # Detach before writing cache state so autograd never sees an in-place mutation of a leaf/grad tensor.
    if not self.record_past:
        self.conv_states[state_idx] = full_conv_states.detach()[..., -self.conv_kernel_size[state_idx] :].clone()
    else:
        self.conv_states[state_idx] = full_conv_states.detach()
    return full_conv_states


def _clone_cache_layer(
    src,
    keys: torch.Tensor | None = None,
    values: torch.Tensor | None = None,
    training: bool = False,
):
    """Clone a Hybrid/Sliding/Full DynamicCache layer preserving KV, cumulative_length, and 4 conv states."""
    if isinstance(src, LinearAttentionAndSlidingWindowAttentionLayer):
        dst = LinearAttentionAndSlidingWindowAttentionLayer(sliding_window=src.sliding_window)
        dst.cumulative_length = src.cumulative_length
    elif isinstance(src, LinearAttentionAndFullAttentionLayer):
        dst = LinearAttentionAndFullAttentionLayer()
    else:
        raise TypeError(f"Unsupported Inkling cache layer type: {type(src)}")

    dst.is_initialized = src.is_initialized
    dst.record_past = src.record_past
    dst.number_of_states = src.number_of_states
    dst.device = src.device
    dst.dtype = src.dtype

    if keys is not None:
        dst.keys = keys
    elif src.keys is not None:
        dst.keys = src.keys.detach().clone()

    if values is not None:
        dst.values = values
    elif src.values is not None:
        dst.values = src.values.detach().clone()

    dst.conv_states = {
        idx: (t.detach().clone() if torch.is_tensor(t) else None)
        for idx, t in src.conv_states.items()
    }
    dst.recurrent_states = {
        idx: (t.detach().clone() if torch.is_tensor(t) else None)
        for idx, t in src.recurrent_states.items()
    }
    dst.is_conv_states_initialized = dict(src.is_conv_states_initialized)
    dst.is_recurrent_states_initialized = dict(src.is_recurrent_states_initialized)
    dst.has_previous_state = dict(src.has_previous_state)
    dst.conv_kernel_size = dict(src.conv_kernel_size)

    if training:
        dst.update_conv_state = types.MethodType(_train_update_conv_state, dst)
    return dst


def aligned_pass_indices(m: int, n: int, pairing: str = "tail") -> list[int]:
    """Map decode pass k in {0..n-1} to prefill pass index j in {0..m-1}.

    - 'tail' (default, legacy v1/v2 compat): j(k) = clip(m - n + k, 0, m - 1).
    - 'anchored' (subspace-exact for Inkling/VarLoop when n >= 2): anchors decode
      pass k=0 (looped=False, no MessageGate) to prefill Pass 0 (looped=False),
      and maps k >= 1 (looped=True) to the deepest prefill passes m - n + k.
      E.g. 4/2 -> [0, 3], 4/3 -> [0, 2, 3], 5/3 -> [0, 3, 4], 5/2 -> [0, 4].
    """
    assert pairing in ("tail", "anchored"), f"Unsupported pass pairing: {pairing}"
    if pairing == "anchored" and n >= 2 and m > n:
        return [0] + [min(max(m - n + k, 1), m - 1) for k in range(1, n)]
    return [min(max(m - n + k, 0), m - 1) for k in range(n)]


def build_aligned_decode_cache(
    prefill_cache: DynamicCache,
    m: int,
    n: int,
    config,
    kv_maps: InklingKVMapAdapter | None = None,
    training: bool = False,
    pass_pairing: str = "tail",
) -> DynamicCache:
    """Align an m-pass prefill cache to an n-pass decode cache and optionally fold prompt KV maps.

    Supports both 'tail' (j = clip(m - n + k, 0, m - 1)) and 'anchored' (k=0 -> Pass 0
    for exact looped=False subspace alignment, k>=1 -> m - n + k) pairing rules.
    """
    loop_start = config.loop_layer_index_start
    loop_end = config.loop_layer_index_end
    num_layers = config.num_hidden_layers
    num_loop = loop_end - loop_start
    total_slots = num_layers + (n - 1) * num_loop

    decode_cache = DynamicCache()
    decode_cache.layers = [None] * total_slots

    # 1. Pre-loop layers 0..5 (unmodified, identical across all m and n)
    for l in range(loop_start):
        decode_cache.layers[l] = _clone_cache_layer(prefill_cache.layers[l], training=training)

    # 2. Looped core layers 6..17 across decode passes k = 0..n-1
    paired_passes = aligned_pass_indices(m, n, pairing=pass_pairing)
    for k, j in enumerate(paired_passes):
        for l in range(loop_start, loop_end):
            src_slot = cache_slot(j, l, config)
            dst_slot = cache_slot(k, l, config)
            src_layer = prefill_cache.layers[src_slot]
            if kv_maps is not None:
                if kv_maps.cross_pass:
                    keys_all = [prefill_cache.layers[cache_slot(p, l, config)].keys.detach() for p in range(m)]
                    vals_all = [prefill_cache.layers[cache_slot(p, l, config)].values.detach() for p in range(m)]
                    mk, mv = kv_maps.map_loop_cross(n, k, j, l - loop_start, keys_all, vals_all)
                else:
                    mk, mv = kv_maps.map_loop(n, k, l - loop_start, src_layer.keys.detach(), src_layer.values.detach())
                decode_cache.layers[dst_slot] = _clone_cache_layer(src_layer, keys=mk, values=mv, training=training)
            else:
                decode_cache.layers[dst_slot] = _clone_cache_layer(src_layer, training=training)

    # 3. Post-loop coda layers 18..23 (from the m-pass prefill coda)
    for l in range(loop_end, num_layers):
        src_layer = prefill_cache.layers[l]
        if kv_maps is not None:
            mk, mv = kv_maps.map_post(n, l - loop_end, src_layer.keys.detach(), src_layer.values.detach())
            decode_cache.layers[l] = _clone_cache_layer(src_layer, keys=mk, values=mv, training=training)
        else:
            decode_cache.layers[l] = _clone_cache_layer(src_layer, training=training)

    return decode_cache


def clone_full_cache(cache: DynamicCache, training: bool = False) -> DynamicCache:
    """Shallow-structure deep-tensor clone of a DynamicCache (prevents mutation)."""
    out = DynamicCache()
    out.layers = [_clone_cache_layer(layer, training=training) for layer in cache.layers]
    return out


@torch.inference_mode()
def select_cache_rows(cache: DynamicCache, keep: torch.Tensor, batch_size: int) -> None:
    """In-place batch compaction for Inkling2Loop DynamicCache (KV + 4 short-conv states)."""
    for layer in cache.layers:
        if layer.keys is not None and layer.keys.shape[0] == batch_size:
            layer.keys = layer.keys.index_select(0, keep)
        if layer.values is not None and layer.values.shape[0] == batch_size:
            layer.values = layer.values.index_select(0, keep)
        for idx, t in layer.conv_states.items():
            if torch.is_tensor(t) and t.shape[0] == batch_size:
                layer.conv_states[idx] = t.index_select(0, keep)
        for idx, t in layer.recurrent_states.items():
            if torch.is_tensor(t) and t.shape[0] == batch_size:
                layer.recurrent_states[idx] = t.index_select(0, keep)


@torch.inference_mode()
def concat_caches(caches: list[DynamicCache]) -> DynamicCache:
    """Concatenate micro-batched Inkling2Loop DynamicCaches (same padded width) along batch dim 0.

    Consumes input caches in-place to avoid doubling peak KV-cache VRAM on 128K batches.
    """
    if len(caches) == 1:
        return caches[0]
    ref = caches[0]
    out = DynamicCache()
    out.layers = []
    for li in range(len(ref.layers)):
        layers_i = [c.layers[li] for c in caches]
        src = layers_i[0]
        keys = torch.cat([l.keys for l in layers_i], dim=0) if src.keys is not None else None
        for l in layers_i:
            l.keys = None
        values = torch.cat([l.values for l in layers_i], dim=0) if src.values is not None else None
        for l in layers_i:
            l.values = None
        dst = _clone_cache_layer(src, keys=keys, values=values, training=False)
        dst.conv_states = {
            idx: (torch.cat([l.conv_states[idx] for l in layers_i], dim=0) if torch.is_tensor(t) else None)
            for idx, t in src.conv_states.items()
        }
        dst.recurrent_states = {
            idx: (torch.cat([l.recurrent_states[idx] for l in layers_i], dim=0) if torch.is_tensor(t) else None)
            for idx, t in src.recurrent_states.items()
        }
        for l in layers_i:
            l.conv_states.clear()
            l.recurrent_states.clear()
        out.layers.append(dst)
    caches.clear()
    return out


@dataclass
class SystemConfig:
    mode: str  # "native", "training_free", "loopbridge", "distilled"
    decode_depths: tuple[int, ...] = (1,)
    lora_rank: int = 64
    freeze_maps: bool = False
    kv_map_layers: str = "all"  # "global" (L11, L17, L23 only) or "all" (18 layers; legacy default for v2 ckpt compat)
    kv_map_cross_pass: bool = False  # Cross-Pass Global KV Fusion across all m prefill passes
    adapt_router: bool = True
    adapt_attnres: bool = True
    adapt_rel_sink: bool = False  # LoRA on relative_position_bias.linear & sink_proj (vLLM qkvr-compatible)
    adapt_sink: bool = True  # Set False with --freeze-sink to protect pretrained 1M attention sink calibration
    adapt_pre_loop: bool = False  # True for param-matched distilled controls
    max_prefill_passes: int = 3  # Max prefill loop depth supported by cross-pass KV maps (e.g. 4 or 5 for varloop)
    teacher_passes: int = 3  # Teacher loop depth for forward-KL distillation
    jsd_beta: float = 0.0  # 0.0 = pure Forward KL; (0,1) e.g. 0.5 = Hessian-matched Generalized JSD for GKD
    pass_pairing: str = "tail"  # "tail" (legacy j=m-n+k) or "anchored" (k=0 -> Pass 0, k>=1 -> m-n+k)


class InklingLoopBridgeSystem(nn.Module):
    """Single-backbone frozen Inkling-2-Mini with toggleable LoopBridge or Distilled adapters."""

    def __init__(self, base_model: Inkling2LoopForCausalLM, cfg: SystemConfig):
        super().__init__()
        self.cfg = cfg
        self.base = base_model.eval().requires_grad_(False)
        self.config = base_model.config
        self.device = next(base_model.parameters()).device
        self.adapter_state = {"on": False}
        self.kv_maps: InklingKVMapAdapter | None = None

        if cfg.mode in ("native", "training_free"):
            return

        loop_start = self.config.loop_layer_index_start
        start_layer = 0 if cfg.adapt_pre_loop else loop_start

        # 1. Attention projection LoRAs (q, k, v, o.linear, o.gate_proj, plus optional r_proj & sink_proj)
        for layer_idx in range(start_layer, self.config.num_hidden_layers):
            layer = self.base.model.layers[layer_idx]
            attn = layer.self_attn
            attn.q_proj.linear = ToggleLoRALinear(attn.q_proj.linear, cfg.lora_rank, self.adapter_state)
            attn.k_proj.linear = ToggleLoRALinear(attn.k_proj.linear, cfg.lora_rank, self.adapter_state)
            attn.v_proj.linear = ToggleLoRALinear(attn.v_proj.linear, cfg.lora_rank, self.adapter_state)
            attn.o_proj.linear = ToggleLoRALinear(attn.o_proj.linear, cfg.lora_rank, self.adapter_state)
            attn.o_proj.gate_proj = ToggleLoRALinear(attn.o_proj.gate_proj, cfg.lora_rank, self.adapter_state)
            if cfg.adapt_rel_sink:
                attn.relative_position_bias.linear = ToggleLoRALinear(
                    attn.relative_position_bias.linear, min(16, cfg.lora_rank), self.adapter_state
                )
                if getattr(cfg, "adapt_sink", True):
                    attn.sink_proj = ToggleLoRALinear(attn.sink_proj, min(8, cfg.lora_rank), self.adapter_state)

            # 2. Sparse-MoE Router delta (lets shortened decode re-route across all 128 frozen experts)
            if cfg.adapt_router:
                layer.mlp.gate = ToggleRouterDelta(layer.mlp.gate, self.adapter_state)

            # 3. Block-AttnRes query deltas
            if cfg.adapt_attnres:
                if layer.attention_input is not None:
                    layer.attention_input = ToggleQueryDelta(layer.attention_input, self.adapter_state)
                layer.mlp_input = ToggleQueryDelta(layer.mlp_input, self.adapter_state)

            # 4. Message gates when decoding with >= 2 passes (only looped core layers 6..17 have them)
            if max(cfg.decode_depths) >= 2 and hasattr(layer, "q_message_gate"):
                mg_rank = min(16, cfg.lora_rank)
                for gname in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                    gate = getattr(layer, gname)
                    gate.down_proj = ToggleLoRALinear(gate.down_proj, mg_rank, self.adapter_state)

        if cfg.adapt_attnres:
            self.base.model.final_input = ToggleQueryDelta(self.base.model.final_input, self.adapter_state)
            if max(cfg.decode_depths) >= 2:
                self.base.model.message_input = ToggleQueryDelta(self.base.model.message_input, self.adapter_state)

        # 5. Prompt KV maps (LoopBridge only)
        if cfg.mode == "loopbridge" and not cfg.freeze_maps:
            self.kv_maps = InklingKVMapAdapter(
                self.config,
                self.device,
                decode_depths=cfg.decode_depths,
                kv_map_layers=cfg.kv_map_layers,
                cross_pass=cfg.kv_map_cross_pass,
                max_prefill_passes=cfg.max_prefill_passes,
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: p.detach().cpu().clone()
            for name, p in self.named_parameters()
            if p.requires_grad
        }

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        params = dict(self.named_parameters())
        expected = {n for n, p in params.items() if p.requires_grad}
        if set(state) != expected:
            missing = expected - set(state)
            extra = set(state) - expected
            raise ValueError(f"Checkpoint trainable keys mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
        with torch.no_grad():
            for name, val in state.items():
                params[name].copy_(val.to(params[name].device, dtype=params[name].dtype))

    @torch.no_grad()
    def merge_for_serving(self) -> None:
        """Fold all LoRAs, RouterDeltas, and QueryDeltas into base weights for zero-overhead serving.
        WARNING: Irreversible — only call for inference/timing, never during training.
        Note: Only valid for mode='distilled' (prefill+decode identical) or when running a separate
        merged decode backbone for pure throughput timing!
        """
        for layer in self.base.model.layers:
            attn = layer.self_attn
            for sub, attr in ((attn.q_proj, "linear"), (attn.k_proj, "linear"), (attn.v_proj, "linear"),
                              (attn.o_proj, "linear"), (attn.o_proj, "gate_proj"),
                              (attn.relative_position_bias, "linear"), (attn, "sink_proj")):
                child = getattr(sub, attr)
                if isinstance(child, ToggleLoRALinear):
                    setattr(sub, attr, child.merged_linear())
            if isinstance(layer.mlp.gate, ToggleRouterDelta):
                layer.mlp.gate = layer.mlp.gate.merged_router()
            if isinstance(layer.attention_input, ToggleQueryDelta):
                layer.attention_input = layer.attention_input.merged_module()
            if isinstance(layer.mlp_input, ToggleQueryDelta):
                layer.mlp_input = layer.mlp_input.merged_module()
            for gname in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate"):
                if hasattr(layer, gname):
                    g = getattr(layer, gname)
                    if isinstance(g.down_proj, ToggleLoRALinear):
                        g.down_proj = g.down_proj.merged_linear()
        if isinstance(self.base.model.final_input, ToggleQueryDelta):
            self.base.model.final_input = self.base.model.final_input.merged_module()
        if isinstance(self.base.model.message_input, ToggleQueryDelta):
            self.base.model.message_input = self.base.model.message_input.merged_module()

    def _lm_head_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.base.lm_head(hidden_states).float()
        return self.base.logit_softcap(logits)

    def _run_prefill_chunked(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_passes: int,
        adapter_on: bool,
        chunk_size: int = 0,
        return_logits: bool = True,
    ):
        """Run prefill with optional chunking, never materializing lm_head logits on non-final tokens."""
        set_loop_passes(self.base, num_passes)
        self.adapter_state["on"] = adapter_on
        seq_len = input_ids.shape[1]
        if attention_mask is not None:
            pos_ids = compute_position_ids(attention_mask)
        else:
            pos_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1)

        if chunk_size <= 0 or seq_len <= chunk_size:
            out = self.base.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=pos_ids,
                use_cache=True,
            )
            logits = self._lm_head_logits(out.last_hidden_state[:, -1:]) if return_logits else None
            return logits, out.cache

        cache = None
        last_hidden = None
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            sub_ids = input_ids[:, start:end]
            sub_pos = pos_ids[:, start:end]
            sub_mask = None if attention_mask is None else attention_mask[:, :end]
            out = self.base.model(
                input_ids=sub_ids,
                attention_mask=sub_mask,
                position_ids=sub_pos,
                cache=cache,
                use_cache=True,
            )
            cache = out.cache
            if end == seq_len:
                last_hidden = out.last_hidden_state[:, -1:]
        logits = self._lm_head_logits(last_hidden) if (return_logits and last_hidden is not None) else None
        return logits, cache

    def forward_kl(
        self,
        prefix: torch.Tensor,
        pmask: torch.Tensor,
        suffix: torch.Tensor,
        smask: torch.Tensor,
        m: int,
        n: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward-KL distillation step from frozen native 3/3 teacher to student (m/n or distilled n/n).

        Supervises EVERY target token suffix[:, 0..S-1] (including the first answer token predicted at
        the last prompt position prefix[:, -1]), matching LoopBridge and Distilled n/n identically.
        """
        dec_in = torch.cat([prefix[:, -1:], suffix[:, :-1]], dim=1)
        full_mask = torch.cat([pmask, smask[:, :-1]], dim=1)
        full_pos = compute_position_ids(full_mask)
        P = prefix.shape[1]
        dec_pos = full_pos[:, P - 1 :]

        # 1. Frozen Original teacher (adapter strictly OFF)
        t_passes = int(getattr(self.cfg, "teacher_passes", 3))
        with torch.no_grad():
            _, teacher_pre_cache = self._run_prefill_chunked(
                prefix[:, :-1], pmask[:, :-1], num_passes=t_passes, adapter_on=False, chunk_size=2048, return_logits=False
            )
            # If LoopBridge with m == t_passes, reuse teacher_pre_cache for student prefill (clone before teacher decode mutates it)
            if self.cfg.mode == "loopbridge" and m == t_passes:
                student_pre_cache = teacher_pre_cache
                t_dec_cache = clone_full_cache(teacher_pre_cache, training=False)
            else:
                student_pre_cache = None
                t_dec_cache = teacher_pre_cache

            set_loop_passes(self.base, t_passes)
            self.adapter_state["on"] = False
            t_out = self.base(
                input_ids=dec_in,
                attention_mask=full_mask,
                position_ids=dec_pos,
                cache=t_dec_cache,
                use_cache=True,
            )
            teacher_logp = t_out.logits.float().log_softmax(dim=-1)
            del t_out, t_dec_cache

        # 2. Student forward
        if self.cfg.mode == "loopbridge":
            if student_pre_cache is None:
                with torch.no_grad():
                    _, student_pre_cache = self._run_prefill_chunked(
                        prefix[:, :-1], pmask[:, :-1], num_passes=m, adapter_on=False, chunk_size=2048, return_logits=False
                    )
            s_dec_cache = build_aligned_decode_cache(
                student_pre_cache,
                m=m,
                n=n,
                config=self.config,
                kv_maps=self.kv_maps,
                training=True,
                pass_pairing=getattr(self.cfg, "pass_pairing", "tail"),
            )
            del student_pre_cache
            set_loop_passes(self.base, n)
            self.adapter_state["on"] = True
            s_out = self.base(
                input_ids=dec_in,
                attention_mask=full_mask,
                position_ids=dec_pos,
                cache=s_dec_cache,
                use_cache=True,
            )
            student_logp = s_out.logits.float().log_softmax(dim=-1)
            del s_out, s_dec_cache
            # Keep adapter_state['on']=True through caller's loss.backward() for gradient checkpointing recomputation.
        elif self.cfg.mode == "distilled":
            # Tied-loop Distilled n/n: adapter active on BOTH prefill and decode (same frozen 3/3 teacher)
            assert m == n, f"Distilled mode requires m == n, got {m}/{n}"
            set_loop_passes(self.base, n)
            self.adapter_state["on"] = True
            full_ids = torch.cat([prefix, suffix[:, :-1]], dim=1)
            s_out = self.base.model(
                input_ids=full_ids,
                attention_mask=full_mask,
                position_ids=full_pos,
                use_cache=False,
            )
            student_logits = self._lm_head_logits(s_out.last_hidden_state[:, P - 1 :])
            student_logp = student_logits.log_softmax(dim=-1)
            del s_out, student_logits
            # Keep adapter_state['on']=True through caller's loss.backward() for gradient checkpointing recomputation.
        else:
            raise ValueError(f"forward_kl unsupported for mode={self.cfg.mode}")

        keep = smask.bool()
        jsd_beta = float(getattr(self.cfg, "jsd_beta", 0.0))
        if jsd_beta > 0.0:
            assert 0.0 < jsd_beta < 1.0, f"jsd_beta must be in (0, 1), got {jsd_beta}"
            t_lp = teacher_logp[keep]
            s_lp = student_logp[keep]
            log_m = torch.logaddexp(t_lp + math.log(jsd_beta), s_lp + math.log(1.0 - jsd_beta))
            kl_tm = F.kl_div(log_m, t_lp, reduction="sum", log_target=True)
            kl_sm = F.kl_div(log_m, s_lp, reduction="sum", log_target=True)
            # Divide by beta*(1-beta) to match Forward-KL curvature near P_S ≈ P_T so LR stays identical
            loss_sum = (jsd_beta * kl_tm + (1.0 - jsd_beta) * kl_sm) / (jsd_beta * (1.0 - jsd_beta))
            with torch.no_grad():
                kl_metric_sum = F.kl_div(s_lp, t_lp, reduction="sum", log_target=True)
        else:
            kl_tok = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True).sum(dim=-1)
            loss_sum = kl_tok[keep].sum()
            kl_metric_sum = loss_sum.detach()
        with torch.no_grad():
            target = suffix.unsqueeze(-1)
            ce_sum = -student_logp.gather(-1, target).squeeze(-1)[keep].sum()
            tce_sum = -teacher_logp.gather(-1, target).squeeze(-1)[keep].sum()
            n_tok = keep.sum().to(ce_sum.dtype)
        return loss_sum, torch.stack((kl_metric_sum, ce_sum, tce_sum, n_tok))


class InklingRunner:
    """Unified inference & timing runner for native, training-free untied, LoopBridge, and Distilled."""

    def __init__(
        self,
        system: InklingLoopBridgeSystem,
        m: int,
        n: int,
        chunk_size: int = 2048,
        time_adapter: bool = False,
    ):
        self.system = system.eval()
        self.m = m
        self.n = n
        self.chunk_size = chunk_size
        self.time_adapter = time_adapter
        self.last_adapter_seconds = 0.0

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        mode = self.system.cfg.mode
        if mode in ("native", "distilled"):
            # Native m/m or Distilled n/n: prefill whole prompt directly
            adapter_on = mode == "distilled"
            logits, cache = self.system._run_prefill_chunked(
                input_ids,
                attention_mask,
                num_passes=self.m,
                adapter_on=adapter_on,
                chunk_size=self.chunk_size,
            )
            self.system.adapter_state["on"] = False
            return logits[:, -1], cache

        # Untied (training_free or loopbridge):
        # Prefill prompt[:-1] with frozen m-pass backbone, fold KV maps once, decode prompt[-1] with n passes.
        assert input_ids.shape[1] >= 2, "Untied prefill requires prompt length >= 2"
        pre_mask = None if attention_mask is None else attention_mask[:, :-1]
        _, pre_cache = self.system._run_prefill_chunked(
            input_ids[:, :-1],
            pre_mask,
            num_passes=self.m,
            adapter_on=False,
            chunk_size=self.chunk_size,
            return_logits=False,
        )
        if self.time_adapter:
            torch.cuda.synchronize(input_ids.device)
            t0 = perf_counter()
        dec_cache = build_aligned_decode_cache(
            pre_cache,
            m=self.m,
            n=self.n,
            config=self.system.config,
            kv_maps=self.system.kv_maps if mode == "loopbridge" else None,
            training=False,
            pass_pairing=getattr(self.system.cfg, "pass_pairing", "tail"),
        )
        if self.time_adapter:
            torch.cuda.synchronize(input_ids.device)
            self.last_adapter_seconds = perf_counter() - t0
        del pre_cache
        return self.decode(input_ids[:, -1:], dec_cache, attention_mask)

    @torch.inference_mode()
    def decode(
        self,
        token_ids: torch.Tensor,
        cache: DynamicCache,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        mode = self.system.cfg.mode
        set_loop_passes(self.system.base, self.n)
        self.system.adapter_state["on"] = mode in ("loopbridge", "distilled")
        if attention_mask is not None:
            pos = (attention_mask.long().sum(dim=-1, keepdim=True) - 1).clamp(min=0)
        else:
            past = cache.get_seq_length()
            pos = torch.full((token_ids.shape[0], 1), past, device=token_ids.device, dtype=torch.long)
        out = self.system.base(
            input_ids=token_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            cache=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        self.system.adapter_state["on"] = False
        return out.logits[:, -1], out.cache


def load_inkling_runner(
    model_path: str | Path,
    mode: str,
    m: int,
    n: int,
    checkpoint: str | Path | None = None,
    device: str | torch.device = "cuda",
    chunk_size: int = 2048,
    time_adapter: bool = False,
    merge_distilled: bool = False,
) -> InklingRunner:
    base = load_inkling_model(model_path, device=device)
    if checkpoint is not None:
        saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        sys_cfg = SystemConfig(**saved["system_config"])
        system = InklingLoopBridgeSystem(base, sys_cfg)
        system.load_trainable_state(saved["trainable_state"])
        if merge_distilled and sys_cfg.mode == "distilled":
            system.merge_for_serving()
            system.cfg.mode = "native"
    else:
        sys_cfg = SystemConfig(mode=mode, decode_depths=(n,))
        system = InklingLoopBridgeSystem(base, sys_cfg)
    return InklingRunner(system, m=m, n=n, chunk_size=chunk_size, time_adapter=time_adapter)
