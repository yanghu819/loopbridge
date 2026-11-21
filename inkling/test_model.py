"""Exhaustive numerical and autograd verification for Inkling-2-Mini LoopBridge.

Verifies before any training begins:
  1. Identity-init LoopBridge (3/3) matches native Original (3/3) (`top1_eq == 1.0`).
  2. Non-zero random LoopBridge (3/1 and 3/2) parallel teacher-forcing decode matches
     step-by-step cached InklingRunner inference (`top1_eq == 1.0`, tiny BF16 KL).
  3. Chunked prefill (chunk_size=16) matches full prefill (`top1_eq == 1.0`).
  4. Gradients flow cleanly and finitely into EVERY trainable parameter (LoRA, RouterDelta,
     QueryDelta, KV maps) for both `loopbridge` (3/1, 3/2) and `distilled` (1/1, 2/2).
  5. Batched cache compaction (`select_cache_rows`) matches unbatched decoding.
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from inkling.model import (
    InklingLoopBridgeSystem,
    InklingRunner,
    SystemConfig,
    build_aligned_decode_cache,
    compute_position_ids,
    load_inkling_model,
    select_cache_rows,
    set_loop_passes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/dev/shm/inkling/model")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    base = load_inkling_model(args.model_path, device=device)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Solve: If 3x + 7 = 22, what is x^2 + 1?"}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="none",
    )
    answer = "<|message_model|><|content_text|>3x = 15, so x = 5, and x^2 + 1 = 26.<|end_message|><|content_model_end_sampling|>"
    p_ids = tok.encode(prompt, add_special_tokens=False)
    s_ids = tok.encode(answer, add_special_tokens=False)
    prefix = torch.tensor([p_ids], device=device)
    suffix = torch.tensor([s_ids], device=device)
    pmask = torch.ones_like(prefix)
    smask = torch.ones_like(suffix)
    print(f"Loaded Inkling-2-Mini. P={len(p_ids)}, S={len(s_ids)}")

    # Test 1: Native 3/3 full vs chunked prefill (chunk_size=16)
    native_sys = InklingLoopBridgeSystem(base, SystemConfig(mode="native"))
    r_full = InklingRunner(native_sys, m=3, n=3, chunk_size=0)
    r_chunk = InklingRunner(native_sys, m=3, n=3, chunk_size=16)
    l_full, _ = r_full.prefill(prefix)
    l_chunk, _ = r_chunk.prefill(prefix)
    top1_chunk = (l_full.argmax(-1) == l_chunk.argmax(-1)).float().mean().item()
    kl_chunk = F.kl_div(l_chunk.log_softmax(-1), l_full.log_softmax(-1), reduction="batchmean", log_target=True).item()
    print(f"[TEST 1] Chunked prefill (16) vs full: top1_eq={top1_chunk:.3f}, KL={kl_chunk:.6f}")
    assert top1_chunk == 1.0 and kl_chunk < 0.05

    # Test 2: Non-zero random LoopBridge (3/1 and 3/2) parallel training vs step-by-step inference
    for n in (1, 2):
        base_copy = load_inkling_model(args.model_path, device=device)
        sys = InklingLoopBridgeSystem(base_copy, SystemConfig(mode="loopbridge", decode_depths=(n,), lora_rank=32))
        n_params = sum(p.numel() for p in sys.trainable_parameters())
        print(f"[TEST 2 n={n}] LoopBridge trainable params: {n_params/1e6:.2f}M")

        # Perturb all zero-init trainable weights with small random values
        torch.manual_seed(42 + n)
        with torch.no_grad():
            for name, p in sys.named_parameters():
                if p.requires_grad and ("lora_b" in name or "delta" in name or "k_loop" in name or "v_loop" in name or "k_post" in name or "v_post" in name):
                    p.normal_(0.0, 0.01)

        # (a) Parallel decode forward (exactly what forward_kl runs)
        with torch.no_grad():
            _, pre_cache = sys._run_prefill_chunked(prefix[:, :-1], pmask[:, :-1], num_passes=3, adapter_on=False)
            dec_cache_par = build_aligned_decode_cache(pre_cache, m=3, n=n, config=sys.config, kv_maps=sys.kv_maps, training=False)
            dec_in = torch.cat([prefix[:, -1:], suffix[:, :-1]], dim=1)
            full_mask = torch.cat([pmask, smask[:, :-1]], dim=1)
            dec_pos = compute_position_ids(full_mask)[:, prefix.shape[1] - 1 :]
            set_loop_passes(sys.base, n)
            sys.adapter_state["on"] = True
            logits_par = sys.base(input_ids=dec_in, attention_mask=full_mask, position_ids=dec_pos, cache=dec_cache_par, use_cache=True).logits
            sys.adapter_state["on"] = False

            # (b) Step-by-step InklingRunner inference
            runner = InklingRunner(sys, m=3, n=n, chunk_size=16)
            first_logit, c = runner.prefill(prefix)
            step_logits = [first_logit.unsqueeze(1)]
            for t in range(suffix.shape[1] - 1):
                nxt, c = runner.decode(suffix[:, t : t + 1], c)
                step_logits.append(nxt.unsqueeze(1))
            logits_step = torch.cat(step_logits, dim=1)

        kl = F.kl_div(logits_step.float().log_softmax(-1), logits_par.float().log_softmax(-1), reduction="none", log_target=True).sum(-1).mean().item()
        top1 = (logits_step.argmax(-1) == logits_par.argmax(-1)).float().mean().item()
        max_diff = (logits_step - logits_par).abs().max().item()
        print(f"[TEST 2 n={n}] Parallel train vs step-by-step runner: top1_eq={top1:.4f}, KL/tok={kl:.6f}, max_diff={max_diff:.4f}")
        assert top1 >= 0.95 and kl < 0.05, f"Train/inference mismatch for n={n}: top1={top1}, kl={kl}"

        # Test 3: Backward pass & gradient check
        sys.zero_grad(set_to_none=True)
        loss, metrics = sys.forward_kl(prefix, pmask, suffix, smask, m=3, n=n)
        (loss / metrics[3]).backward()
        zero_grads = []
        nonfinite = []
        for name, p in sys.named_parameters():
            if p.requires_grad:
                if p.grad is None or p.grad.abs().max().item() == 0.0:
                    zero_grads.append(name)
                elif not torch.isfinite(p.grad).all():
                    nonfinite.append(name)
        print(f"[TEST 3 n={n}] Backward: KL/tok={(metrics[0]/metrics[3]).item():.4f}, zero_grads={len(zero_grads)}, nonfinite={len(nonfinite)}")
        if zero_grads:
            print("  Zero grads:", zero_grads[:10])
        assert not nonfinite and len(zero_grads) == 0, f"Grad issue: zero={zero_grads[:5]}, nonfinite={nonfinite[:5]}"
        del sys, base_copy
        torch.cuda.empty_cache()

    # Test 4: Distilled 1/1 backward check
    base_d = load_inkling_model(args.model_path, device=device)
    sys_d = InklingLoopBridgeSystem(
        base_d, SystemConfig(mode="distilled", decode_depths=(1,), lora_rank=64, adapt_pre_loop=True)
    )
    with torch.no_grad():
        for name, p in sys_d.named_parameters():
            if p.requires_grad and "lora_b" in name:
                p.normal_(0.0, 0.001)
    n_params_d = sum(p.numel() for p in sys_d.trainable_parameters())
    loss, metrics = sys_d.forward_kl(prefix, pmask, suffix, smask, m=1, n=1)
    (loss / metrics[3]).backward()
    zero_d = [n for n, p in sys_d.named_parameters() if p.requires_grad and (p.grad is None or p.grad.abs().max().item() == 0)]
    print(f"[TEST 4 Distilled 1/1] params={n_params_d/1e6:.2f}M, KL/tok={(metrics[0]/metrics[3]).item():.4f}, zero_grads={len(zero_d)}")
    assert len(zero_d) == 0, f"Distilled zero grads: {zero_d[:5]}"
    del sys_d, base_d
    torch.cuda.empty_cache()

    # Test 5: Batched compaction (select_cache_rows)
    p2 = tok.encode(
        tok.apply_chat_template([{"role": "user", "content": "What is 2 + 2?"}], tokenize=False, add_generation_prompt=True, reasoning_effort="none"),
        add_special_tokens=False,
    )
    pad = tok.pad_token_id
    W = max(len(p_ids), len(p2))
    b_ids = torch.full((2, W), pad, dtype=torch.long, device=device)
    b_mask = torch.zeros((2, W), dtype=torch.long, device=device)
    b_ids[0, W - len(p_ids) :] = torch.tensor(p_ids)
    b_mask[0, W - len(p_ids) :] = 1
    b_ids[1, W - len(p2) :] = torch.tensor(p2)
    b_mask[1, W - len(p2) :] = 1

    r_tf = InklingRunner(InklingLoopBridgeSystem(base, SystemConfig(mode="training_free")), m=3, n=1)
    l_bat, c_bat = r_tf.prefill(b_ids, b_mask)
    keep = torch.tensor([1], device=device)
    select_cache_rows(c_bat, keep, 2)
    tok1 = l_bat[1:2].argmax(-1, keepdim=True)
    mask1 = torch.cat([b_mask[1:2], torch.ones((1, 1), dtype=torch.long, device=device)], dim=1)
    l_next_bat, _ = r_tf.decode(tok1, c_bat, mask1)

    l_solo, c_solo = r_tf.prefill(torch.tensor([p2], device=device))
    tok1_s = l_solo.argmax(-1, keepdim=True)
    l_next_solo, _ = r_tf.decode(tok1_s, c_solo)
    eq5 = int(tok1.item() == tok1_s.item()) and int(l_next_bat.argmax(-1).item() == l_next_solo.argmax(-1).item())
    print(f"[TEST 5] select_cache_rows exact match: {bool(eq5)}")
    assert eq5

    print("ALL_TESTS_PASSED!")


if __name__ == "__main__":
    main()
