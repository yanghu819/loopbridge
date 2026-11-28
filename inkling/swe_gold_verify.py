"""Official SWE-bench Verified Environment Builder, Grader & Automated Gold-Patch Verification Gate.

Uses ONLY official `princeton-nlp/SWE-bench` grading logic (`swebench.harness.log_parsers.python`
+ `swebench.harness.grading`: `test_passed`, `test_maintained`, `test_failed`, `get_eval_tests_report`,
`get_resolution_status`) over isolated per-episode git worktrees and pinned Python 3.9 / 3.11 venvs.

Every candidate SWE-bench Verified instance MUST pass the two-sided Gold-Patch Verification Gate
before being admitted to evaluation:
  1. Unmodified `base_commit` + official `test_patch` -> FAILS (`resolved == False`, >=1 FAIL_TO_PASS fails)
  2. Official `gold patch` + official `test_patch`    -> PASSES (`resolved == True`, 100% F2P & P2P pass!)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types


def _install_docker_stub() -> None:
    """Allow importing official `swebench.harness.log_parsers` & `grading` without Docker/Modal/datasets SDK."""
    class _StubMod(types.ModuleType):
        def __getattr__(self, item):
            return object

    for name in (
        "docker", "docker.models", "docker.models.containers", "docker.errors", "docker.types",
        "datasets", "modal", "tenacity",
        "swebench.harness.modal_eval", "swebench.harness.docker_utils", "swebench.harness.reporting",
    ):
        if name not in sys.modules:
            m = _StubMod(name)
            if name.endswith("errors"):
                m.ImageNotFound = Exception
                m.APIError = Exception
                m.NotFound = Exception
            sys.modules[name] = m


def extract_modified_test_files(test_patch: str) -> list[str]:
    """Extract file paths modified in unified diff `test_patch` (exact official SWE-bench logic)."""
    files: list[str] = []
    for line in test_patch.splitlines():
        if line.startswith("diff --git "):
            m = re.search(r" b/(.+)$", line)
            if m:
                files.append(m.group(1).strip())
    return list(dict.fromkeys(files))


def django_test_directives(test_files: list[str]) -> list[str]:
    """Convert Django test file paths (tests/foo/test_bar.py) to dotted Django runtests.py modules."""
    out = []
    for fp in test_files:
        if not fp.startswith("tests/") or not fp.endswith(".py"):
            continue
        mod = fp[len("tests/") : -3].replace("/", ".")
        if mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        out.append(mod)
    return out


def build_official_test_cmd(instance: dict) -> list[str]:
    """Build the exact official per-repo SWE-bench test invocation for an instance."""
    repo = instance["repo"]
    tfiles = extract_modified_test_files(instance["test_patch"])
    if repo == "django/django":
        directives = django_test_directives(tfiles)
        return ["./tests/runtests.py", "--verbosity", "2", "--settings=test_sqlite", "--parallel", "1", *directives]
    if repo == "sympy/sympy":
        py_files = [f for f in tfiles if f.endswith(".py")]
        # Exact official SWE-bench SymPy test runner (required by parse_log_sympy!)
        return ["python", "./bin/test", "-C", "--verbose", *py_files]
    if repo == "pytest-dev/pytest":
        py_files = [f for f in tfiles if f.endswith(".py")]
        return ["python", "-m", "pytest", "-rA", *py_files]
    if repo == "sphinx-doc/sphinx":
        py_files = [f for f in tfiles if f.endswith(".py")]
        return ["python", "-m", "pytest", "-rA", *py_files]
    if repo == "pylint-dev/pylint":
        py_files = [f for f in tfiles if f.endswith(".py")]
        return ["python", "-m", "pytest", "-rA", *py_files]
    if repo == "psf/requests":
        py_files = [f for f in tfiles if f.endswith(".py")]
        return ["python", "-m", "pytest", "-rA", *py_files]
    if repo == "pallets/flask":
        py_files = [f for f in tfiles if f.endswith(".py")]
        return ["python", "-m", "pytest", "-rA", *py_files]
    raise ValueError(f"Unsupported repo: {repo}")


def select_venv_for_instance(instance: dict, venvs_root: Path) -> Path:
    repo = instance["repo"]
    ver = str(instance.get("version", ""))
    if repo == "django/django":
        try:
            vf = float(ver)
        except Exception:
            vf = 4.0
        if vf >= 4.2:
            return venvs_root / "venv_py311"
        return venvs_root / "venv_py39"
    if repo in ("sphinx-doc/sphinx", "psf/requests", "pylint-dev/pylint"):
        compat = venvs_root / "venv_py39_compat"
        if compat.exists():
            return compat
    return venvs_root / "venv_py39"


def _apply_git_patch(workdir: Path, patch_text: str) -> tuple[bool, str]:
    if not patch_text or not patch_text.strip():
        return True, ""
    pfile = workdir / ".__tmp_patch.diff"
    pfile.write_text(patch_text, encoding="utf-8")
    try:
        for cmd in (
            ["git", "apply", "-v", str(pfile)],
            ["git", "apply", "--reject", "--whitespace=fix", str(pfile)],
            ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(pfile)],
        ):
            r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return True, ""
        return False, (r.stderr or r.stdout)[-500:]
    finally:
        pfile.unlink(missing_ok=True)


def create_isolated_repo_sandbox(instance: dict, repos_root: Path, sandbox_dir: Path) -> Path:
    """Checkout exact `base_commit` from local bare mirror into `sandbox_dir` with zero future-commit leakage."""
    safe = instance["repo"].replace("/", "__")
    bare = repos_root / f"{safe}.git"
    if not bare.exists():
        raise FileNotFoundError(f"Missing bare repo: {bare}")
    if sandbox_dir.exists():
        shutil.rmtree(sandbox_dir, ignore_errors=True)
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "safe.directory"
    env["GIT_CONFIG_VALUE_0"] = "*"
    subprocess.run(["git", "init", "-q"], cwd=sandbox_dir, env=env, check=True, capture_output=True, timeout=30)
    # Fetch ONLY the single base_commit (works for both full and shallow bare mirrors, leaks 0 future history!)
    r = subprocess.run(
        ["git", "fetch", "-q", "--depth=1", str(bare), instance["base_commit"]],
        cwd=sandbox_dir, env=env, capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        # Fallback: copy/clone local objects when bare repo doesn't advertise commit directly
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "-q", "--local", "--no-checkout", str(bare), str(sandbox_dir)],
            env=env, check=True, capture_output=True, timeout=60,
        )
        subprocess.run(
            ["git", "checkout", "-f", "-q", instance["base_commit"]],
            cwd=sandbox_dir, env=env, check=True, capture_output=True, timeout=60,
        )
        subprocess.run(["git", "remote", "remove", "origin"], cwd=sandbox_dir, env=env, capture_output=True)
        subprocess.run(
            ["bash", "-c", "git for-each-ref --format='%(refname)' | xargs -r -n1 git update-ref -d"],
            cwd=sandbox_dir, env=env, capture_output=True,
        )
    else:
        subprocess.run(
            ["git", "checkout", "-f", "-q", "FETCH_HEAD"],
            cwd=sandbox_dir, env=env, check=True, capture_output=True, timeout=60,
        )
    # Pytest uses setuptools_scm-generated _pytest/_version.py when imported from source tree
    if instance["repo"] == "pytest-dev/pytest":
        vp = sandbox_dir / "src" / "_pytest" / "_version.py"
        if vp.parent.exists() and not vp.exists():
            ver = str(instance.get("version", "7.0"))
            vp.write_text(f"version = '{ver}.0'\nversion_tuple = (7, 0, 0)\n", encoding="utf-8")
    return sandbox_dir


def grade_patch_official(
    instance: dict,
    model_patch: str,
    repos_root: Path,
    venvs_root: Path,
    swebench_pkg_root: Path,
    timeout_s: int = 120,
    keep_tmp: bool = False,
) -> dict:
    """Grade `model_patch` on a fresh isolated checkout using official SWE-bench parsers & grading rules."""
    _install_docker_stub()
    if str(swebench_pkg_root) not in sys.path:
        sys.path.insert(0, str(swebench_pkg_root))

    from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS, EvalType, ResolvedStatus
    from swebench.harness.grading import get_eval_tests_report, get_resolution_status
    from swebench.harness.log_parsers.python import MAP_REPO_TO_PARSER_PY

    venv = select_venv_for_instance(instance, venvs_root)
    py_bin = venv / "bin"
    tmp_base = Path(tempfile.mkdtemp(prefix=f"swe_grade_{instance['instance_id']}_", dir="/dev/shm"))
    repo_dir = tmp_base / "testbed"

    try:
        create_isolated_repo_sandbox(instance, repos_root, repo_dir)
        # 1. Apply model patch (if non-empty)
        if model_patch and model_patch.strip():
            ok, err = _apply_git_patch(repo_dir, model_patch)
            if not ok:
                return {"resolved": False, "status": "APPLY_PATCH_FAIL", "error": err}

        # 2. Reset any files touched by test_patch back to base_commit, then apply official test_patch
        tfiles = extract_modified_test_files(instance["test_patch"])
        if tfiles:
            subprocess.run(["git", "checkout", instance["base_commit"], "--", *tfiles], cwd=repo_dir, capture_output=True)
        ok_t, err_t = _apply_git_patch(repo_dir, instance["test_patch"])
        if not ok_t:
            return {"resolved": False, "status": "TEST_PATCH_FAIL", "error": err_t}

        # 3. Execute official test command with repo on PYTHONPATH + venv PATH
        cmd = build_official_test_cmd(instance)
        if cmd[0] == "python":
            cmd[0] = str(py_bin / "python")
        elif cmd[0].startswith("./"):
            cmd = [str(py_bin / "python"), cmd[0], *cmd[1:]]

        env = os.environ.copy()
        env["PATH"] = f"{py_bin}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{repo_dir}:{repo_dir / 'src'}" if (repo_dir / "src").exists() else str(repo_dir)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["TQDM_DISABLE"] = "1"

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
            timed_out = False
            rc = proc.returncode
        except subprocess.TimeoutExpired as te:
            combined = ((te.stdout or b"").decode(errors="ignore") if isinstance(te.stdout, bytes) else (te.stdout or ""))
            timed_out = True
            rc = -1
        elapsed = time.perf_counter() - t0

        if timed_out:
            return {"resolved": False, "status": "TESTS_TIMEOUT", "elapsed_s": round(elapsed, 2)}

        # 4. Parse with official SWE-bench repo parser & grade with official get_eval_tests_report
        parser_fn = MAP_REPO_TO_PARSER_PY[instance["repo"]]
        # SymPy uses pytest here so fall back to pytest parser if sympy parser finds 0
        from swebench.harness.log_parsers.python import parse_log_pytest
        dummy_spec = types.SimpleNamespace(
            instance_id=instance["instance_id"],
            repo=instance["repo"],
            version=instance["version"],
        )
        status_map = parser_fn(combined, dummy_spec)
        if not status_map and instance["repo"] == "sympy/sympy":
            status_map = parse_log_pytest(combined, dummy_spec)

        f2p = json.loads(instance["FAIL_TO_PASS"]) if isinstance(instance["FAIL_TO_PASS"], str) else instance["FAIL_TO_PASS"]
        p2p = json.loads(instance["PASS_TO_PASS"]) if isinstance(instance["PASS_TO_PASS"], str) else instance["PASS_TO_PASS"]

        report = get_eval_tests_report(
            eval_status_map=status_map,
            gold_results={FAIL_TO_PASS: f2p, PASS_TO_PASS: p2p},
            eval_type=EvalType.PASS_AND_FAIL,
        )
        res_status = get_resolution_status(report)
        resolved = res_status == ResolvedStatus.FULL.value
        return {
            "resolved": resolved,
            "status": res_status,
            "rc": rc,
            "f2p_pass": len(report[FAIL_TO_PASS]["success"]),
            "f2p_fail": len(report[FAIL_TO_PASS]["failure"]),
            "p2p_pass": len(report[PASS_TO_PASS]["success"]),
            "p2p_fail": len(report[PASS_TO_PASS]["failure"]),
            "parsed_tests": len(status_map),
            "elapsed_s": round(elapsed, 2),
            "log_tail": combined[-600:] if not resolved else "",
        }
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp_base, ignore_errors=True)


def _verify_one(args_tuple):
    inst, repos_root, venvs_root, pkg_root = args_tuple
    iid = inst["instance_id"]
    try:
        # 1. Empty patch MUST NOT resolve
        r_base = grade_patch_official(inst, "", repos_root, venvs_root, pkg_root, timeout_s=90)
        if r_base["resolved"] or r_base.get("parsed_tests", 0) == 0:
            return {"instance_id": iid, "repo": inst["repo"], "version": inst["version"], "verified": False, "reason": f"base_{r_base['status']}", "base": r_base}
        # 2. Official gold patch MUST resolve 100%
        r_gold = grade_patch_official(inst, inst["patch"], repos_root, venvs_root, pkg_root, timeout_s=90)
        ok = bool(r_gold["resolved"])
        return {
            "instance_id": iid,
            "repo": inst["repo"],
            "version": inst["version"],
            "difficulty": inst.get("difficulty", ""),
            "verified": ok,
            "reason": "OK" if ok else f"gold_{r_gold['status']}",
            "base_f2p": f"{r_base.get('f2p_pass',0)}/{r_base.get('f2p_pass',0)+r_base.get('f2p_fail',0)}",
            "gold_f2p": f"{r_gold.get('f2p_pass',0)}/{r_gold.get('f2p_pass',0)+r_gold.get('f2p_fail',0)}",
            "gold_p2p": f"{r_gold.get('p2p_pass',0)}/{r_gold.get('p2p_pass',0)+r_gold.get('p2p_fail',0)}",
            "gold_s": r_gold.get("elapsed_s", 0),
        }
    except Exception as e:
        return {"instance_id": iid, "repo": inst["repo"], "version": inst["version"], "verified": False, "reason": f"exc:{type(e).__name__}:{str(e)[:160]}"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("/dev/shm/inkling/swebench/swe_bench_verified.jsonl"))
    p.add_argument("--repos-root", type=Path, default=Path("/dev/shm/inkling/swebench/repos"))
    p.add_argument("--venvs-root", type=Path, default=Path("/dev/shm/inkling/swebench/venvs"))
    p.add_argument("--pkg-root", type=Path, default=Path("/dev/shm/inkling/swebench"))
    p.add_argument("--output", type=Path, default=Path("/dev/shm/inkling/swebench/gold_verified_instances.json"))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--repos", nargs="+", default=["django/django", "sympy/sympy", "sphinx-doc/sphinx", "pytest-dev/pytest", "pylint-dev/pylint", "psf/requests", "pallets/flask"])
    args = p.parse_args()

    rows = [json.loads(l) for l in args.dataset.open() if l.strip()]
    cand = [r for r in rows if r["repo"] in set(args.repos)]
    print(f"[GoldGate] Verifying {len(cand)} SWE-bench Verified instances across {len(args.repos)} repos with {args.workers} workers...", flush=True)

    t0 = time.perf_counter()
    results = []
    passed = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_verify_one, (r, args.repos_root, args.venvs_root, args.pkg_root)) for r in cand]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            results.append(rec)
            if rec["verified"]:
                passed.append(rec["instance_id"])
            if i % 25 == 0 or i == len(cand):
                print(f"  [{i}/{len(cand)}] verified={len(passed)} ({100*len(passed)/i:.1f}%) elapsed={time.perf_counter()-t0:.1f}s", flush=True)

    by_repo = {}
    for rec in results:
        by_repo.setdefault(rec["repo"], [0, 0])
        by_repo[rec["repo"]][1] += 1
        if rec["verified"]:
            by_repo[rec["repo"]][0] += 1
    print("[GoldGate] Per-repo verified counts:")
    for repo, (ok, tot) in sorted(by_repo.items()):
        print(f"  {repo:24s}: {ok:3d}/{tot:3d} ({100*ok/max(1,tot):.1f}%)")

    args.output.write_text(json.dumps({"verified_ids": sorted(passed), "by_repo": by_repo, "details": results}, indent=2) + "\n")
    print(f"[DONE] Saved {len(passed)} gold-verified official SWE-bench Verified instances to {args.output}", flush=True)


if __name__ == "__main__":
    main()
