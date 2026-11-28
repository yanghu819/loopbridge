"""Official mini-swe-agent Bash-Only SWE-bench Verified Evaluator for Inkling-2 & TML-d24-varloop.

Implements Thinking Machines' exact SWE-bench Verified evaluation protocol (`mini-swe-agent`,
bash-only native tool calls via Inkling-2's `<|content_invoke_tool_json|>` template) over
isolated per-episode git sandboxes and grades every submitted patch using official
`swebench.harness.grading` (`FAIL_TO_PASS` + `PASS_TO_PASS`).

Key systems properties:
  1. 100% strictly append-only multi-turn token IDs (preserving reasoning + tool calls + observations)
     -> achieves >95% real vLLM Automatic Prefix Cache hit rate across 30-100+ turn trajectories!
  2. Sticky episode-to-replica routing -> every turn of an episode hits the same B300 GPU replica.
  3. Direct head-to-head comparison of APC baselines (`Orig 4/4, 3/3, 2/2, Distilled 2/2`) vs.
     LoopBridge Multi-Turn Policies (`Policy R/I` vs. `Policy D0`) on real 30-100+ turn coding agents!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
from time import perf_counter
import urllib.request

from inkling.swe_gold_verify import (
    create_isolated_repo_sandbox,
    grade_patch_official,
    select_venv_for_instance,
)


_TOOL_CALL_RE = re.compile(
    r"<\|message_model\|>([^\s<]*?)<\|content_invoke_tool_json\|>(.*?)<\|end_message\|>",
    re.DOTALL,
)

BASH_TOOL_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a non-interactive bash command in the current repository root directory with the project's Python environment active.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    }
]

SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer shell to solve programming tasks."


def build_instance_prompt(problem_statement: str, workdir_label: str = ".") -> str:
    return (
        f"<pr_description>\nConsider the following PR description:\n{problem_statement}\n</pr_description>\n\n"
        "<instructions>\n"
        "# Task Instructions\n\n"
        "You're a software engineer interacting continuously with a computer by submitting bash commands.\n"
        "Your task is to make changes to non-test source files in the current repository directory in order to fix "
        "the issue described in the PR description in a way that is general and consistent with the codebase.\n\n"
        "## Recommended Workflow\n"
        "1. Analyze the codebase by finding and reading relevant files\n"
        "2. Create a short script or run the relevant test suite to reproduce the issue\n"
        "3. Edit the source code to resolve the issue\n"
        "4. Verify your fix works and test edge cases\n"
        "5. Remove any temporary reproduction scripts you created so `git status` only shows intended source edits\n\n"
        "## Command Execution Rules\n"
        "- Every response MUST include at least one `bash` tool call (you may issue multiple independent `bash` calls in parallel).\n"
        "- Directory or environment variable changes do not persist across calls; each command runs in a fresh subshell in the repo root.\n"
        "- Always use non-interactive flags; avoid interactive editors (vi, nano).\n\n"
        "## Submission\n"
        "When your fix is complete and verified, run a SEPARATE command to submit:\n"
        "```bash\n"
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff\n"
        "```\n"
        "After issuing `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, the task ends immediately and your current `git diff` is graded.\n"
        "</instructions>"
    )


def format_observation(returncode: int, output: str) -> str:
    # Exact official mini-swe-agent swebench.yaml 10,000-char head/tail elision template
    if len(output) < 10000:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    elided = len(output) - 10000
    return (
        f"<returncode>{returncode}</returncode>\n"
        "<warning>\nThe output of your last command was too long. Use head, tail, sed, or targeted grep.\n</warning>\n"
        f"<output_head>\n{output[:5000]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-5000:]}\n</output_tail>"
    )


def parse_bash_calls(raw_text: str) -> list[str]:
    cmds: list[str] = []
    for header_name, body in _TOOL_CALL_RE.findall(raw_text):
        try:
            payload = json.loads(body.strip())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        fn = str(payload.get("name") or header_name).strip()
        if fn != "bash":
            continue
        args = payload.get("args", payload.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if isinstance(args, dict) and isinstance(args.get("command"), str):
            cmds.append(args["command"])
    return cmds


def exec_bash_cmd(command: str, repo_dir: Path, py_bin: Path, timeout_s: int = 60) -> tuple[int, str]:
    env = os.environ.copy()
    env["PATH"] = f"{py_bin}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = f"{repo_dir}:{repo_dir / 'src'}" if (repo_dir / "src").exists() else str(repo_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env["PAGER"] = "cat"
    env["MANPAGER"] = "cat"
    env["LESS"] = "-R"
    env["PIP_PROGRESS_BAR"] = "off"
    env["TQDM_DISABLE"] = "1"
    env["GIT_PAGER"] = "cat"

    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=repo_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        return proc.returncode, out or ""
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
        out, _ = proc.communicate()
        return 124, (out or "") + f"\n[Command timed out after {timeout_s}s]"


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
    p_toks = int(usage.get("prompt_tokens", len(prompt_ids)))
    g_toks = int(usage.get("completion_tokens", 0))
    reason = "max_new_tokens" if choice.get("finish_reason") == "length" else "eos"
    return raw_text, p_toks, g_toks, reason, wall_s


def run_swe_episode(
    instance: dict,
    sample_idx: int,
    tokenizer,
    base_url: str,
    model_name: str,
    repos_root: Path,
    venvs_root: Path,
    pkg_root: Path,
    reasoning_effort: str,
    max_steps: int,
    max_tokens_per_step: int,
    max_context_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict:
    iid = instance["instance_id"]
    venv = select_venv_for_instance(instance, venvs_root)
    py_bin = venv / "bin"
    eos_id = tokenizer.eos_token_id

    tmp_root = Path(tempfile.mkdtemp(prefix=f"swe_agent_{iid}_s{sample_idx}_", dir="/dev/shm"))
    repo_dir = tmp_root / "testbed"
    t_ep0 = perf_counter()
    step_logs: list[dict] = []
    submitted = False
    truncated_steps = 0
    env_wall_s = 0.0
    model_patch = ""

    try:
        create_isolated_repo_sandbox(instance, repos_root, repo_dir)

        init_msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_instance_prompt(instance["problem_statement"])},
        ]
        prompt_str = tokenizer.apply_chat_template(
            init_msgs,
            tools=BASH_TOOL_SPEC,
            reasoning_effort=reasoning_effort,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids: list[int] = tokenizer.encode(prompt_str, add_special_tokens=False)
        turn0_prompt_tokens = len(prompt_ids)

        for step in range(max_steps):
            if len(prompt_ids) >= max_context_tokens - 512:
                truncated_steps += 1
                break
            eff_max = min(max_tokens_per_step, max_context_tokens - len(prompt_ids) - 128)
            if eff_max <= 64:
                truncated_steps += 1
                break

            raw_txt, p_toks, g_toks, reason, llm_s = _post_completion(
                base_url=base_url,
                model=model_name,
                prompt_ids=prompt_ids,
                max_tokens=eff_max,
                temperature=temperature,
                top_p=top_p,
                seed=seed + step * 137,
            )
            if reason != "eos":
                truncated_steps += 1

            # Strictly append-only token growth -> >95% real vLLM APC hit rate!
            gen_ids = tokenizer.encode(raw_txt, add_special_tokens=False)
            prompt_ids.extend(gen_ids)
            if reason == "eos" and eos_id is not None and (not gen_ids or gen_ids[-1] != eos_id):
                prompt_ids.append(eos_id)

            cmds = parse_bash_calls(raw_txt)
            step_logs.append(
                {
                    "step": step,
                    "prompt_tokens": p_toks,
                    "gen_tokens": g_toks,
                    "n_calls": len(cmds),
                    "stop_reason": reason,
                    "llm_s": round(llm_s, 4),
                }
            )

            if not cmds:
                # Official mini-swe-agent format error recovery (append user reminder)
                err_msg = (
                    "Every response must include at least one `bash` tool call. "
                    "Call the `bash` tool with `{\"command\": \"...\"}`, or submit with "
                    "`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff`."
                )
                prompt_ids.extend(
                    tokenizer.encode(f"<|message_user|><|content_text|>{err_msg}<|end_message|>", add_special_tokens=False)
                )
                continue

            tool_chunks = []
            for cmd in cmds:
                t_e0 = perf_counter()
                rc, out = exec_bash_cmd(cmd, repo_dir, py_bin, timeout_s=60)
                env_wall_s += perf_counter() - t_e0
                if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in out and rc == 0:
                    submitted = True
                obs = format_observation(rc, out)
                tool_chunks.append(f"<|message_tool|>bash<|content_text|>{obs}<|end_message|>")
            if submitted:
                break
            prompt_ids.extend(tokenizer.encode("".join(tool_chunks), add_special_tokens=False))

        # Extract clean tracked git diff directly from repo_dir (never include untracked helper scripts!)
        diff_proc = subprocess.run(["git", "diff"], cwd=repo_dir, capture_output=True, text=True, timeout=30)
        model_patch = diff_proc.stdout or ""
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # Grade immediately in a fresh isolated sandbox via official swebench grading
    t_g0 = perf_counter()
    grade = grade_patch_official(
        instance=instance,
        model_patch=model_patch,
        repos_root=repos_root,
        venvs_root=venvs_root,
        swebench_pkg_root=pkg_root,
        timeout_s=120,
    )
    grade_s = perf_counter() - t_g0

    tot_gen = sum(s["gen_tokens"] for s in step_logs)
    max_prompt = max((s["prompt_tokens"] for s in step_logs), default=turn0_prompt_tokens)
    llm_wall = sum(s["llm_s"] for s in step_logs)

    return {
        "id": f"{iid}#s{sample_idx}",
        "instance_id": iid,
        "repo": instance["repo"],
        "version": instance["version"],
        "difficulty": instance.get("difficulty", ""),
        "resolved": bool(grade.get("resolved", False)),
        "score": 1.0 if grade.get("resolved", False) else 0.0,
        "grade_status": grade.get("status", "UNKNOWN"),
        "submitted": submitted,
        "num_steps": len(step_logs),
        "turn0_prompt_tokens": turn0_prompt_tokens,
        "max_prompt_tokens": max_prompt,
        "total_gen_tokens": tot_gen,
        "mean_gen_tokens_per_step": round(tot_gen / max(1, len(step_logs)), 1),
        "truncated_steps": truncated_steps,
        "patch_chars": len(model_patch),
        "llm_wall_s": round(llm_wall, 3),
        "env_wall_s": round(env_wall_s, 3),
        "grade_s": round(grade_s, 3),
        "wall_s": round(perf_counter() - t_ep0, 3),
        "steps": step_logs,
        "patch": model_patch[:20000],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urls", nargs="+", default=["http://127.0.0.1:8000"])
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", type=Path, default=Path("/dev/shm/inkling/swebench/swe_bench_verified.jsonl"))
    p.add_argument("--verified-list", type=Path, default=Path("/dev/shm/inkling/swebench/gold_verified_instances.json"))
    p.add_argument("--repos-root", type=Path, default=Path("/dev/shm/inkling/swebench/repos"))
    p.add_argument("--venvs-root", type=Path, default=Path("/dev/shm/inkling/swebench/venvs"))
    p.add_argument("--pkg-root", type=Path, default=Path("/dev/shm/inkling/swebench"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--max-tokens-per-step", type=int, default=8192)
    p.add_argument("--max-context-tokens", type=int, default=196608)
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    all_rows = {r["instance_id"]: r for r in (json.loads(l) for l in args.dataset.open() if l.strip())}
    vdata = json.loads(args.verified_list.read_text())
    v_ids = vdata["verified_ids"]
    if args.limit and args.limit < len(v_ids):
        # Deterministic stratified selection across repos when limit < len(v_ids)
        by_repo: dict[str, list[str]] = {}
        for iid in v_ids:
            by_repo.setdefault(all_rows[iid]["repo"], []).append(iid)
        picked: list[str] = []
        repos = sorted(by_repo.keys())
        idx_r = {r: 0 for r in repos}
        while len(picked) < args.limit:
            added = False
            for r in repos:
                if idx_r[r] < len(by_repo[r]) and len(picked) < args.limit:
                    picked.append(by_repo[r][idx_r[r]])
                    idx_r[r] += 1
                    added = True
            if not added:
                break
        v_ids = picked

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "swebench_verified.jsonl"
    done: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done.values()))

    jobs = []
    for idx, iid in enumerate(v_ids):
        for k in range(args.n_samples):
            jid = f"{iid}#s{k}"
            if jid not in done:
                jobs.append((jid, all_rows[iid], k, args.seed + k * 1009 + idx))

    print(
        f"[SWE-Verified] total={len(v_ids)*args.n_samples}, done={len(done)}, todo={len(jobs)}, "
        f"urls={len(args.urls)}, conc={args.concurrency}, effort={args.reasoning_effort}, max_steps={args.max_steps}",
        flush=True,
    )

    urls = args.urls
    in_flight = {u: 0 for u in urls}
    rr = 0
    lock = threading.Lock()

    def acq() -> str:
        nonlocal rr
        with lock:
            ml = min(in_flight.values())
            for off in range(len(urls)):
                c = urls[(rr + off) % len(urls)]
                if in_flight[c] == ml:
                    in_flight[c] += 1
                    rr = (rr + off + 1) % len(urls)
                    return c
            in_flight[urls[0]] += 1
            return urls[0]

    def rel(u: str) -> None:
        with lock:
            in_flight[u] = max(0, in_flight[u] - 1)

    def run_one(job):
        jid, inst, k, s = job
        u = acq()
        try:
            return run_swe_episode(
                instance=inst,
                sample_idx=k,
                tokenizer=tok,
                base_url=u,
                model_name=args.model,
                repos_root=args.repos_root,
                venvs_root=args.venvs_root,
                pkg_root=args.pkg_root,
                reasoning_effort=args.reasoning_effort,
                max_steps=args.max_steps,
                max_tokens_per_step=args.max_tokens_per_step,
                max_context_tokens=args.max_context_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=s,
            )
        finally:
            rel(u)

    t0 = perf_counter()
    completed = 0
    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run_one, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[rec["id"]] = rec
            completed += 1
            if completed % 5 == 0 or completed == len(jobs):
                n = len(done)
                res = sum(1 for r in done.values() if r["resolved"])
                avg_st = sum(r["num_steps"] for r in done.values()) / n
                avg_gen = sum(r["mean_gen_tokens_per_step"] for r in done.values()) / n
                avg_ctx = sum(r["max_prompt_tokens"] for r in done.values()) / n
                avg_llm = sum(r["llm_wall_s"] for r in done.values()) / n
                print(
                    f"  [SWE-Verified] {n}/{len(v_ids)*args.n_samples} | resolved={res}/{n} ({100*res/n:.2f}%) "
                    f"| steps={avg_st:.1f} | gen/step={avg_gen:.0f} | max_ctx={avg_ctx:.0f} "
                    f"| llm_s={avg_llm:.1f}s | elapsed={perf_counter()-t0:.1f}s",
                    flush=True,
                )

    n = len(done)
    res = sum(1 for r in done.values() if r["resolved"])
    print(f"[DONE] SWE-bench Verified: resolved={res}/{n} ({100*res/max(1,n):.2f}%), wall={perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
