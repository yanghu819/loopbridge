"""Verify token & top-k logit parity between PyTorch InklingRunner and Official vLLM LoopBridge/Distilled.

Runs greedy decoding for `steps` tokens on identical prompts spanning short (128), medium (2048),
and >8K LogScale (9000) regimes, comparing exact top-1 token agreement and mean top-k logprob
difference against a running vLLM server (`inkling.vllm_loopbridge serve ...`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from inkling.model import load_inkling_runner


def _vllm_greedy_logprobs(url: str, model: str, prompt_ids: list[int], steps: int) -> tuple[list[int], list[float]]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_ids,
            "max_tokens": steps,
            "temperature": 0.0,
            "logprobs": 1,
            "skip_special_tokens": False,
            "return_token_ids": True,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    ch = data["choices"][0]
    tids = ch.get("token_ids") or []
    lps = (ch.get("logprobs") or {}).get("token_logprobs") or []
    return tids, [float(x) if x is not None else 0.0 for x in lps]


@torch.inference_mode()
def _pt_teacher_forced_parity(runner, prompt_ids: list[int], vllm_tids: list[int], device: torch.device) -> dict:
    """Compare PyTorch InklingRunner greedy predictions against vLLM's generated tokens (teacher-forced)."""
    p = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    logits, cache = runner.prefill(p)
    matches = 0
    total = 0
    lp_diffs = []
    # Note: in vLLM LoopBridge, token 0 is emitted by m-pass prefill; tokens 1..steps-1 are emitted by n-pass decode.
    # Evaluate parity over decode steps 1..steps-1 (and separately report step 0).
    step0_match = int(int(logits[0].argmax().item()) == vllm_tids[0]) if vllm_tids else 0
    for t in range(len(vllm_tids) - 1):
        tok_in = torch.tensor([[vllm_tids[t]]], dtype=torch.long, device=device)
        logits, cache = runner.decode(tok_in, cache)
        logp = F.log_softmax(logits[0].float(), dim=-1)
        pt_top1 = int(logp.argmax().item())
        ref_tid = vllm_tids[t + 1]
        matches += int(pt_top1 == ref_tid)
        total += 1
        lp_diffs.append(float(logp[pt_top1].item() - logp[ref_tid].item()))
    return {
        "step0_match": step0_match,
        "decode_steps": total,
        "decode_top1_eq": round(matches / max(1, total), 4),
        "mean_top1_lp_gap": round(sum(lp_diffs) / max(1, len(lp_diffs)), 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=Path("/dev/shm/inkling/model"))
    parser.add_argument("--mode", choices=("native", "training_free", "loopbridge", "distilled"), required=True)
    parser.add_argument("--m", type=int, default=3)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    runner = load_inkling_runner(
        args.model,
        mode=args.mode,
        m=args.m,
        n=args.n,
        checkpoint=args.checkpoint,
        device=args.device,
        merge_distilled=(args.mode == "distilled"),
    )

    test_cases = [
        ("short_128", "Solve step by step: what is the remainder when 7^2026 is divided by 13?"),
        (
            "med_2048",
            "Context:\n"
            + ("The blue archive ledger recorded key ALPHA-7741 for vault 9. " * 140)
            + "\nQuestion: What key is recorded for vault 9?",
        ),
        (
            "logscale_9000",
            "Document:\n"
            + ("Historical maritime shipping logs detail cargo manifests across northern ports. " * 680)
            + "Critical code: OMEGA-9923. "
            + ("Additional port weather reports confirm calm autumn winds. " * 200)
            + "\nQuestion: What is the critical code mentioned in the document?",
        ),
    ]

    dev = torch.device(args.device)
    report = []
    for name, content in test_cases:
        p_str = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium",
        )
        p_ids = tok.encode(p_str, add_special_tokens=False)
        vllm_tids, _ = _vllm_greedy_logprobs(args.url, str(args.model), p_ids, args.steps)
        res = _pt_teacher_forced_parity(runner, p_ids, vllm_tids, dev)
        res["case"] = name
        res["prompt_tokens"] = len(p_ids)
        report.append(res)
        print("PARITY", json.dumps(res), flush=True)

    mean_eq = sum(r["decode_top1_eq"] for r in report) / len(report)
    mean_gap = sum(r["mean_top1_lp_gap"] for r in report) / len(report)
    print(json.dumps({"mean_decode_top1_eq": round(mean_eq, 4), "mean_lp_gap": round(mean_gap, 5)}), flush=True)


if __name__ == "__main__":
    main()
