"""Official BFCL v4 Multi-Turn Stateful Agentic Evaluator for Inkling-2 & TML-d24-varloop.

Executes real multi-turn tool-use episodes against BFCL v4's deterministic stateful Python mock
environments (`GorillaFileSystem`, `TwitterAPI`, `TravelAPI`, `TradingBot`, `VehicleControlAPI`,
`MessageAPI`, `TicketAPI`, `MathAPI`) using Inkling-2's native tool-calling chat template
(`<|message_system|>tool_declare<|content_xml|>`, `<|content_invoke_tool_json|>`, `<|message_tool|>`),
and scores every completed episode with BFCL's official `multi_turn_checker`.

Key design properties:
  1. Strictly append-only token construction across all turns & tool steps -> 100% prefix-cache
     preservation for evaluating vLLM APC baselines (Orig 4/4, 3/3, 2/2, Distilled 2/2) and
     LoopBridge Multi-Turn Policy D0 vs. Policy R/I.
  2. Sticky episode-to-replica routing: all turns of an episode hit the same vLLM GPU server so
     vLLM Automatic Prefix Caching achieves real multi-turn KV reuse.
  3. Thread-safe isolated BFCL mock-environment state (unique per-episode key + cleanup) so high
     concurrency across 8x B300 replicas has zero cross-episode contamination.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import sys
import threading
from time import perf_counter
import urllib.request


_TOOL_CALL_RE = re.compile(
    r"<\|message_model\|>([^\s<]*?)<\|content_invoke_tool_json\|>(.*?)<\|end_message\|>",
    re.DOTALL,
)
_ENV_LOCK = threading.Lock()


def _convert_bfcl_schema(obj):
    """Recursively normalize BFCL function parameter schemas ('dict' -> 'object', 'float' -> 'number')."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str):
                t = v.lower()
                if t == "dict":
                    out[k] = "object"
                elif t == "float":
                    out[k] = "number"
                elif t == "tuple":
                    out[k] = "array"
                else:
                    out[k] = t
            else:
                out[k] = _convert_bfcl_schema(v)
        return out
    if isinstance(obj, list):
        return [_convert_bfcl_schema(x) for x in obj]
    return obj


def _load_tools_for_entry(bfcl_root: Path, involved_classes: list[str], excluded_funcs: list[str]) -> list[dict]:
    from bfcl_eval.constants.executable_backend_config import MULTI_TURN_FUNC_DOC_FILE_MAPPING

    doc_dir = bfcl_root / "bfcl_eval" / "data" / "multi_turn_func_doc"
    excl = set(excluded_funcs or [])
    tools: list[dict] = []
    seen: set[str] = set()
    for cls in involved_classes:
        fname = MULTI_TURN_FUNC_DOC_FILE_MAPPING.get(cls)
        if not fname:
            continue
        p = doc_dir / fname
        if not p.exists():
            continue
        for line in p.open():
            if not line.strip():
                continue
            spec = json.loads(line)
            name = spec["name"]
            if name in excl or name in seen:
                continue
            seen.add(name)
            params = _convert_bfcl_schema(spec.get("parameters", {"type": "object", "properties": {}}))
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": spec.get("description", ""),
                        "parameters": params,
                    },
                }
            )
    return tools


def _parse_inkling_tool_calls(raw_text: str) -> list[tuple[str, dict, str]]:
    """Parse native Inkling-2 `<|content_invoke_tool_json|>` tool calls into (fn_name, args_dict, py_call_str)."""
    calls: list[tuple[str, dict, str]] = []
    for header_name, body in _TOOL_CALL_RE.findall(raw_text):
        body = body.strip()
        try:
            payload = json.loads(body)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        fn_name = str(payload.get("name") or header_name).strip()
        if not fn_name:
            continue
        args = payload.get("args", payload.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        kw_parts = [f"{k}={repr(v)}" for k, v in args.items()]
        py_call = f"{fn_name}({', '.join(kw_parts)})"
        calls.append((fn_name, args, py_call))
    return calls


def _post_completion(
    base_url: str,
    model: str,
    prompt_ids: list[int],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float = 900.0,
) -> tuple[str, int, int, str, float]:
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
    prompt_toks = int(usage.get("prompt_tokens", len(prompt_ids)))
    gen_toks = int(usage.get("completion_tokens", 0))
    fr = choice.get("finish_reason", "stop")
    reason = "max_new_tokens" if fr == "length" else "eos"
    return raw_text, prompt_toks, gen_toks, reason, wall_s


def _cleanup_bfcl_instances(prefix: str) -> None:
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils

    g = multi_turn_utils.__dict__
    to_del = [k for k in list(g.keys()) if k.startswith(prefix) and k.endswith("_instance")]
    for k in to_del:
        g.pop(k, None)


def _run_episode(
    entry: dict,
    gt_entry: dict,
    split_name: str,
    sample_idx: int,
    bfcl_root: Path,
    tokenizer,
    base_url: str,
    model_name: str,
    reasoning_effort: str,
    max_tokens_per_step: int,
    max_steps_per_turn: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    timeout: float,
) -> dict:
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call

    ep_id = entry["id"]
    run_tag = re.sub(r"[^A-Za-z0-9_]", "_", f"run_{split_name}_{ep_id}_s{sample_idx}_{threading.get_ident()}")
    is_long_ctx = "long_context" in split_name
    tools = _load_tools_for_entry(bfcl_root, entry["involved_classes"], entry.get("excluded_function", []))

    eos_id = tokenizer.eos_token_id
    user_turns: list[list[dict]] = entry["question"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]

    # Build Turn-0 prompt via official Inkling-2 chat template (system tools + thinking effort + user turn 0)
    turn0_msgs = list(user_turns[0])
    prompt_str = tokenizer.apply_chat_template(
        turn0_msgs,
        tools=tools,
        reasoning_effort=reasoning_effort,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids: list[int] = tokenizer.encode(prompt_str, add_special_tokens=False)
    turn0_prompt_tokens = len(prompt_ids)

    decoded_turns: list[list[list[str]]] = []
    step_logs: list[dict] = []
    truncated_steps = 0
    step_counter = 0
    t_ep0 = perf_counter()

    try:
        for t_idx, u_turn in enumerate(user_turns):
            if t_idx > 0:
                # Strictly append-only next user message (100% prefix-cache preserving!)
                u_parts = []
                for msg in u_turn:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "user":
                        u_parts.append(f"<|message_user|><|content_text|>{content}<|end_message|>")
                    elif role == "system":
                        u_parts.append(f"<|message_system|><|content_text|>{content}<|end_message|>")
                if u_parts:
                    prompt_ids.extend(tokenizer.encode("".join(u_parts), add_special_tokens=False))

            turn_steps_py: list[list[str]] = []
            for _sub in range(max_steps_per_turn):
                if len(prompt_ids) >= max_context_tokens - 256:
                    truncated_steps += 1
                    break
                eff_max = min(max_tokens_per_step, max_context_tokens - len(prompt_ids) - 64)
                if eff_max <= 16:
                    truncated_steps += 1
                    break

                raw_txt, p_toks, g_toks, reason, wall_s = _post_completion(
                    base_url=base_url,
                    model=model_name,
                    prompt_ids=prompt_ids,
                    max_tokens=eff_max,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed + step_counter * 131,
                    timeout=timeout,
                )
                step_counter += 1
                if reason != "eos":
                    truncated_steps += 1

                # Strictly append assistant tokens (keeping exact generated thinking + tool call / text + EOS)
                gen_ids = tokenizer.encode(raw_txt, add_special_tokens=False)
                prompt_ids.extend(gen_ids)
                if reason == "eos" and eos_id is not None and (not gen_ids or gen_ids[-1] != eos_id):
                    prompt_ids.append(eos_id)

                parsed_calls = _parse_inkling_tool_calls(raw_txt)
                step_logs.append(
                    {
                        "turn": t_idx,
                        "step": _sub,
                        "prompt_tokens": p_toks,
                        "gen_tokens": g_toks,
                        "n_calls": len(parsed_calls),
                        "stop_reason": reason,
                        "wall_s": round(wall_s, 4),
                    }
                )

                if not parsed_calls:
                    break

                py_calls = [c[2] for c in parsed_calls]
                turn_steps_py.append(py_calls)

                # Execute calls in BFCL's official stateful mock environment
                with _ENV_LOCK:
                    exec_results, _ = execute_multi_turn_func_call(
                        func_call_list=py_calls,
                        initial_config=initial_config,
                        involved_classes=involved_classes,
                        model_name=run_tag,
                        test_entry_id=ep_id,
                        long_context=is_long_ctx,
                        is_evaL_run=False,
                    )

                # Append tool responses strictly in native Inkling-2 format
                tool_chunks = []
                for (fn_name, _args, _py), res_str in zip(parsed_calls, exec_results):
                    tool_chunks.append(f"<|message_tool|>{fn_name}<|content_text|>{res_str}<|end_message|>")
                prompt_ids.extend(tokenizer.encode("".join(tool_chunks), add_special_tokens=False))

            decoded_turns.append(turn_steps_py)

        # Official BFCL v4 Multi-Turn state + response checker (fresh eval instance namespace!)
        with _ENV_LOCK:
            check_res = multi_turn_checker(
                multi_turn_model_result_list_decoded=decoded_turns,
                multi_turn_ground_truth_list=gt_entry["ground_truth"],
                test_entry=entry,
                test_category=split_name,
                model_name=run_tag,
            )
    finally:
        with _ENV_LOCK:
            _cleanup_bfcl_instances(run_tag)

    ep_wall_s = perf_counter() - t_ep0
    valid = bool(check_res.get("valid", False))
    tot_prompt = sum(s["prompt_tokens"] for s in step_logs)
    tot_gen = sum(s["gen_tokens"] for s in step_logs)
    max_prompt = max((s["prompt_tokens"] for s in step_logs), default=turn0_prompt_tokens)
    llm_wall = sum(s["wall_s"] for s in step_logs)

    return {
        "id": f"{ep_id}#s{sample_idx}" if sample_idx >= 0 else ep_id,
        "base_id": ep_id,
        "split": split_name,
        "valid": valid,
        "score": 1.0 if valid else 0.0,
        "error_type": None if valid else check_res.get("error_type", "unknown"),
        "num_user_turns": len(user_turns),
        "num_llm_steps": len(step_logs),
        "turn0_prompt_tokens": turn0_prompt_tokens,
        "max_prompt_tokens": max_prompt,
        "total_prompt_tokens": tot_prompt,
        "total_gen_tokens": tot_gen,
        "mean_gen_tokens_per_step": round(tot_gen / max(1, len(step_logs)), 1),
        "truncated_steps": truncated_steps,
        "llm_wall_s": round(llm_wall, 4),
        "wall_s": round(ep_wall_s, 4),
        "steps": step_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8000"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bfcl-root", type=Path, default=Path("/dev/shm/inkling/bfcl_v4"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["multi_turn_base", "multi_turn_long_context"],
        choices=["multi_turn_base", "multi_turn_long_context"],
    )
    parser.add_argument("--reasoning-effort", default="medium",
                        help="Inkling-2 reasoning effort: none, minimal, low, medium, high")
    parser.add_argument("--max-tokens-per-step", type=int, default=8192)
    parser.add_argument("--max-steps-per-turn", type=int, default=12)
    parser.add_argument("--max-context-tokens", type=int, default=131072)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.bfcl_root))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    urls = args.urls

    for split in args.splits:
        t0 = perf_counter()
        q_path = args.bfcl_root / "bfcl_eval" / "data" / f"BFCL_v4_{split}.json"
        gt_path = args.bfcl_root / "bfcl_eval" / "data" / "possible_answer" / f"BFCL_v4_{split}.json"
        entries = [json.loads(l) for l in q_path.open() if l.strip()]
        gt_map = {r["id"]: r for r in (json.loads(l) for l in gt_path.open() if l.strip())}
        if args.limit and args.limit < len(entries):
            entries = entries[: args.limit]

        jobs: list[tuple[str, dict, dict, int, int]] = []
        for idx, e in enumerate(entries):
            for k in range(args.n_samples):
                jid = f"{e['id']}#s{k}"
                jobs.append((jid, e, gt_map[e["id"]], k, args.seed + k * 1009 + idx))

        out_path = args.output / f"{split}.jsonl"
        done: dict[str, dict] = {}
        if out_path.exists():
            for line in out_path.open():
                if line.strip():
                    rec = json.loads(line)
                    done[rec["id"]] = rec
            out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()))

        todo = [j for j in jobs if j[0] not in done]
        print(
            f"[BFCL-MT] {split}: total={len(jobs)}, done={len(done)}, todo={len(todo)}, "
            f"urls={len(urls)}, conc={args.concurrency}, effort={args.reasoning_effort}",
            flush=True,
        )

        # Sticky episode-to-replica load balancer (entire episode stays on one URL for APC reuse!)
        in_flight = {u: 0 for u in urls}
        rr_cursor = 0
        url_lock = threading.Lock()

        def acquire_url() -> str:
            nonlocal rr_cursor
            with url_lock:
                min_load = min(in_flight.values())
                for off in range(len(urls)):
                    cand = urls[(rr_cursor + off) % len(urls)]
                    if in_flight[cand] == min_load:
                        in_flight[cand] += 1
                        rr_cursor = (rr_cursor + off + 1) % len(urls)
                        return cand
                cand = urls[0]
                in_flight[cand] += 1
                return cand

        def release_url(u: str) -> None:
            with url_lock:
                in_flight[u] = max(0, in_flight[u] - 1)

        def run_one(job):
            jid, entry, gt, s_idx, ep_seed = job
            url = acquire_url()
            try:
                last_err = None
                for attempt in range(2):
                    try:
                        return _run_episode(
                            entry=entry,
                            gt_entry=gt,
                            split_name=split,
                            sample_idx=s_idx,
                            bfcl_root=args.bfcl_root,
                            tokenizer=tokenizer,
                            base_url=url,
                            model_name=args.model,
                            reasoning_effort=args.reasoning_effort,
                            max_tokens_per_step=args.max_tokens_per_step,
                            max_steps_per_turn=args.max_steps_per_turn,
                            max_context_tokens=args.max_context_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            seed=ep_seed + attempt,
                            timeout=args.timeout,
                        )
                    except Exception as e:
                        last_err = e
                raise RuntimeError(f"Failed {jid} on {url}: {last_err}")
            finally:
                release_url(url)

        completed = 0
        with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(run_one, j) for j in todo]
            for fut in as_completed(futs):
                rec = fut.result()
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["id"]] = rec
                completed += 1
                if completed % 20 == 0 or completed == len(todo):
                    vals = [float(r["score"]) for r in done.values()]
                    avg_steps = sum(r["num_llm_steps"] for r in done.values()) / len(done)
                    avg_gen = sum(r["mean_gen_tokens_per_step"] for r in done.values()) / len(done)
                    avg_maxp = sum(r["max_prompt_tokens"] for r in done.values()) / len(done)
                    trunc_ep = sum(1 for r in done.values() if r["truncated_steps"] > 0)
                    print(
                        f"  [{split}] {len(done)}/{len(jobs)} | acc={100*sum(vals)/len(vals):.2f}% "
                        f"| steps={avg_steps:.1f} | gen/step={avg_gen:.0f} | max_ctx={avg_maxp:.0f} "
                        f"| trunc_ep={trunc_ep}/{len(done)} | elapsed={perf_counter()-t0:.1f}s",
                        flush=True,
                    )

        vals = [float(r["score"]) for r in done.values()]
        avg_steps = sum(r["num_llm_steps"] for r in done.values()) / max(1, len(done))
        avg_gen = sum(r["mean_gen_tokens_per_step"] for r in done.values()) / max(1, len(done))
        avg_maxp = sum(r["max_prompt_tokens"] for r in done.values()) / max(1, len(done))
        avg_wall = sum(r["llm_wall_s"] for r in done.values()) / max(1, len(done))
        print(
            f"[DONE] {split}: acc={100*sum(vals)/max(1,len(vals)):.2f}% ({len(done)} eps), "
            f"avg_steps={avg_steps:.2f}, gen/step={avg_gen:.1f}, max_ctx={avg_maxp:.0f}, "
            f"mean_ep_llm_s={avg_wall:.2f}s, wall={perf_counter()-t0:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
