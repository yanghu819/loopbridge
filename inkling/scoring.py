"""Self-contained exact benchmark scorers for Inkling-2-Mini evaluations."""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import json
import math
import re
import string
import subprocess
import sys
import tempfile


RULER_TASKS = ("niah_single_1", "niah_multikey_2", "vt", "cwe")


def extract_visible_text(raw: str) -> str:
    """Extract Inkling-2-Mini's final assistant text after <|content_text|>.

    If generation truncated inside <|content_thinking|> without ever emitting <|content_text|>,
    return empty string so unfinished reasoning traces are never falsely parsed as final answers.
    """
    if "<|content_text|>" in raw:
        txt = raw.rsplit("<|content_text|>", 1)[-1]
    elif "<|content_thinking|>" in raw:
        return ""
    else:
        txt = raw
    for tag in ("<|content_model_end_sampling|>", "<|end_message|>", "<|message_model|>", "<|content_thinking|>"):
        txt = txt.replace(tag, "")
    return txt.strip()


def parse_mcq(text: str, valid_letters: str = "ABCD") -> str:
    """Extract final MCQ choice letter with strict priority: \\boxed{} > explicit answer phrase > standalone uppercase."""
    if not text or not text.strip():
        return "[invalid]"
    # 1. Explicit \boxed{\text{X}} or \boxed{(X)} (highest priority — cannot be hijacked by earlier prose)
    ms = re.findall(
        rf"\\boxed\{{\s*(?:\\(?:text|textbf|mathrm)\{{\s*)?\(?\s*([{valid_letters}])\s*\)?(?:\s*\}})?\s*\}}",
        text,
        re.I,
    )
    if ms:
        return ms[-1].upper()
    # 2. Explicit answer header (supports Markdown bold **Answer:** and requires boundary after letter)
    explicit_pat = (
        rf"(?:the\s+correct\s+answer\s+is|final\s+answer\s+is|(?:\*\*)?answer(?:\*\*)?\s*(?:is|:))\s*"
        rf"(?:\*\*)?\s*\(?\s*([{valid_letters}])\s*\)?(?:\*\*)?(?=\s|$|[.,;:\)\]\*])"
    )
    ms = re.findall(explicit_pat, text, re.I)
    if ms:
        return ms[-1].upper()
    # 3. Parenthesized uppercase option (X)
    ms = re.findall(rf"\(\s*([{valid_letters}])\s*\)", text)
    if ms:
        return ms[-1].upper()
    # 4. Standalone uppercase letter on the final non-empty line only (never match prose "I" or lowercase "a")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        ms = re.findall(rf"(?<![A-Za-z0-9_])([{valid_letters}])(?![A-Za-z0-9_])", last_line)
        if "I" in valid_letters and ms:
            ms = [m for m in ms if not (m == "I" and re.search(r"\bI\s+[a-z]", last_line))]
        if ms:
            return ms[-1].upper()
    return "[invalid]"

SUBSTITUTIONS = [
    ("an ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]
REMOVED = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft", "hours", "km", "units",
    "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents", "degrees", "cm", "gm",
    "pounds", "meters", "meals", "edges", "students", "childrentickets", "multiples",
    "\\text{s}", "\\text{.}", "\\text{\ns}", "\\text{}^2", "\\text{}^3", "\\text{\n}",
    "\\text{}", r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]


def minerva_norm(answer: str) -> str:
    answer = answer.split("=")[-1]
    for expr in REMOVED:
        if expr.isalpha():
            answer = re.sub(rf"(?<!\\)\b{expr}\b", "", answer)
    for before, after in SUBSTITUTIONS:
        answer = answer.replace(before, after)
    for expr in REMOVED:
        if expr.isalpha():
            parts = re.split(r"(\\[a-zA-Z]+)", answer)
            for i in range(0, len(parts), 2):
                parts[i] = parts[i].replace(expr, "")
            answer = "".join(parts)
        else:
            answer = answer.replace(expr, "")
    answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", answer)
    answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", answer)
    answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", answer)
    answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", answer)
    answer = re.sub(r"(sqrt)(\[[^\]]*\])(\w)", "\\1\\2{\\3}", answer)
    answer = re.sub(r"(sqrt)(?!\[)([^{])", "sqrt{\\2}", answer)
    answer = answer.replace("$", "")
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", answer):
        answer = answer.replace(",", "")
    return answer


def bbh_answer(text: str) -> str:
    m = re.search(r"the answer is (.*)", text, re.I)
    if not m:
        return "[invalidanswer]"
    answer = m.group(1).strip().split("\n")[0].strip()
    return answer[:-1] if answer.endswith(".") else answer


def helmet_norm(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def helmet_parse(output: str, prefix: str) -> str:
    for pat in (re.compile(f"(?:{prefix})(.*)(?:\n|$)", flags=re.I), re.compile(r"(?:^)(.*)(?:\n|$)")):
        m = pat.search(output)
        if m is not None:
            return re.sub(f"^{re.escape(prefix)}", "", m[1].strip(), flags=re.I).strip()
    return ""


def helmet_rankings(output: str) -> dict[str, int]:
    output = re.sub(r"[\[\]:]", "", output).lower().replace("id", "")
    longest = max((m.group(0) for m in re.finditer(r"(\d+)(?:\s*>\s*(\d+))*", output)), key=len, default="")
    ranks = [n.strip() for n in longest.split(">") if n.strip().isdigit()]
    out: dict[str, int] = {}
    for i, r in enumerate(ranks):
        out.setdefault(r, len(ranks) - i)
    return out


def helmet_score(row: dict, text: str) -> float:
    if row["metric"] == "subem":
        return float(any(helmet_norm(a) in helmet_norm(text) for a in row["answers"]))
    if row["metric"] == "icl_em":
        pred = helmet_norm(helmet_parse(text, "label:"))
        return float(any(pred == helmet_norm(a) for a in row["answers"]))
    run = sorted(helmet_rankings(text).items(), key=lambda kv: -kv[1])[:10]
    dcg = sum(row["qrel"].get(d, 0) / math.log2(i + 2) for i, (d, _) in enumerate(run))
    ideal = sorted(row["qrel"].values(), reverse=True)[:10]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def qa_f1_score(prediction: str, reference: str) -> float:
    pred = helmet_norm(prediction).split()
    ref = helmet_norm(reference).split()
    common = sum((Counter(pred) & Counter(ref)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


def ruler_score(task: str, prediction: str, answers: list[str]) -> float:
    if task in ("qa_1", "qa_2"):
        prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
        return float(any(answer.lower() in prediction.lower() for answer in answers))
    if task in RULER_TASKS or task.startswith("niah_") or task == "fwe":
        prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip()
        return sum(answer.lower() in prediction.lower() for answer in answers) / len(answers)
    if task in ("qasper", "hotpotqa", "2wikimqa"):
        return max(qa_f1_score(prediction, answer) for answer in answers)
    raise ValueError(f"Unsupported generation task: {task}")


def graphwalks_f1(response: str, answer_nodes: list[str]) -> float:
    """Official OpenAI GraphWalks node-set F1 scorer (handles both same-line and newline Final Answer lists)."""
    if not response or not response.strip():
        return 0.0
    m = None
    for cand in re.finditer(r"Final\s+Answer\s*:\s*(\[[^\]]*\])", response, re.I | re.S):
        m = cand
    if m is None:
        return 0.0
    sampled = [item.strip().strip("'\"") for item in m.group(1).strip("[]").split(",") if item.strip()]
    sampled_set = set(sampled)
    truth_set = set(answer_nodes)
    n_golden = len(truth_set)
    n_sampled = len(sampled_set)
    if n_golden == 0 and n_sampled == 0:
        return 1.0
    n_overlap = len(sampled_set & truth_set)
    recall = n_overlap / n_golden if n_golden > 0 else 0.0
    precision = n_overlap / n_sampled if n_sampled > 0 else 0.0
    return 2.0 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0


def mrcr_score(response: str, answer: str, random_string_to_prepend: str) -> float:
    """Official OpenAI MRCR v2 SequenceMatcher grader (openai/mrcr README)."""
    resp = response.strip()
    if not resp.startswith(random_string_to_prepend):
        return 0.0
    resp = resp.removeprefix(random_string_to_prepend)
    ans = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, resp, ans).ratio())


def extract_inkling_tool_calls(raw_text: str) -> list[tuple[str, dict]]:
    """Parse native Inkling-2-Mini tool invocations from raw model output (with special tokens)."""
    calls: list[tuple[str, dict]] = []
    # Pattern: <|message_model|>tool_name<|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>
    pat = re.compile(
        r"<\|content_invoke_tool_json\|>\s*(\{.*?\})\s*(?:<\|end_message\|>|<\|content_model_end_sampling\|>|$)",
        re.S,
    )
    for m in pat.finditer(raw_text):
        blob = m.group(1).strip()
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("args", obj.get("arguments", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if isinstance(args, dict):
                calls.append((obj["name"], args))
    return calls


def _bfcl_val_match(pred, allowed_list: list) -> bool:
    for a in allowed_list:
        if pred == a:
            return True
        if isinstance(pred, (int, float)) and isinstance(a, (int, float)) and not isinstance(pred, bool) and not isinstance(a, bool):
            if abs(float(pred) - float(a)) <= 1e-6:
                return True
        if isinstance(pred, str) and isinstance(a, str):
            if pred.strip().lower() == a.strip().lower():
                return True
        if isinstance(pred, list) and isinstance(a, list) and len(pred) == len(a):
            if all(_bfcl_val_match(x, [y]) for x, y in zip(pred, a)):
                return True
        if isinstance(pred, dict) and isinstance(a, dict) and set(pred.keys()) == set(a.keys()):
            if all(_bfcl_val_match(pred[k], [a[k]]) for k in a):
                return True
    return False


def _bfcl_call_matches(call: tuple[str, dict], gt_item: dict) -> bool:
    if len(gt_item) != 1:
        return False
    gt_name, gt_params = next(iter(gt_item.items()))
    pred_name, pred_args = call
    # Accept exact or dotted-to-underscore name match (BFCL normalizes dots in some tools)
    if pred_name != gt_name and pred_name.replace(".", "_") != gt_name.replace(".", "_"):
        return False
    if not isinstance(gt_params, dict):
        return False
    # No unexpected non-null parameters outside ground truth schema
    for k, v in pred_args.items():
        if k not in gt_params and v is not None and v != "":
            return False
    # Check every ground-truth parameter
    for k, allowed in gt_params.items():
        if not isinstance(allowed, list):
            allowed = [allowed]
        if k not in pred_args or pred_args[k] is None:
            # Allowed to omit iff "" or None is in allowed list
            if "" in allowed or None in allowed:
                continue
            return False
        if not _bfcl_val_match(pred_args[k], allowed):
            return False
    return True


def bfcl_score(raw_text: str, ground_truth: list[dict] | None, category: str) -> tuple[list[tuple[str, dict]], float]:
    """Exact AST/JSON grader for BFCL v3 Live categories using native Inkling tool invocations."""
    calls = extract_inkling_tool_calls(raw_text)
    if "irrelevance" in category:
        # Never reward truncated thinking traces that never emitted a visible <|content_text|> response
        if "<|content_thinking|>" in raw_text and "<|content_text|>" not in raw_text and not calls:
            return calls, 0.0
        return calls, (1.0 if len(calls) == 0 else 0.0)
    if not ground_truth or len(calls) != len(ground_truth):
        return calls, 0.0
    # Order-insensitive bipartite matching (handles parallel & parallel_multiple cleanly)
    used = [False] * len(ground_truth)

    def backtrack(idx: int) -> bool:
        if idx == len(calls):
            return True
        for j, gt in enumerate(ground_truth):
            if not used[j] and _bfcl_call_matches(calls[idx], gt):
                used[j] = True
                if backtrack(idx + 1):
                    return True
                used[j] = False
        return False

    return calls, (1.0 if backtrack(0) else 0.0)


def extract_python_code(text: str) -> str:
    blocks = re.findall(r"```(?:python3?|py)?\s*\n(.*?)```", text, re.S | re.I)
    if blocks:
        return blocks[-1].strip()
    return ""


_LCB_HARNESS = r'''
import bisect, collections, functools, heapq, io, itertools, json, math, operator, re, signal, string, sys, traceback, typing
from bisect import bisect_left, bisect_right, insort
from collections import Counter, defaultdict, deque, OrderedDict
from contextlib import redirect_stdout
from functools import cache, cmp_to_key, lru_cache, reduce
from heapq import heapify, heappop, heappush, heappushpop, heapreplace, nlargest, nsmallest
from itertools import accumulate, combinations, groupby, permutations, product
from math import ceil, comb, floor, gcd, inf, isqrt, lcm, log, log2, perm, sqrt
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

_STD_PRELUDE = {
    "bisect": bisect, "bisect_left": bisect_left, "bisect_right": bisect_right, "insort": insort,
    "collections": collections, "Counter": Counter, "defaultdict": defaultdict, "deque": deque, "OrderedDict": OrderedDict,
    "functools": functools, "cache": cache, "cmp_to_key": cmp_to_key, "lru_cache": lru_cache, "reduce": reduce,
    "heapq": heapq, "heapify": heapify, "heappop": heappop, "heappush": heappush, "heappushpop": heappushpop,
    "heapreplace": heapreplace, "nlargest": nlargest, "nsmallest": nsmallest,
    "itertools": itertools, "accumulate": accumulate, "combinations": combinations, "groupby": groupby,
    "permutations": permutations, "product": product,
    "math": math, "ceil": ceil, "comb": comb, "floor": floor, "gcd": gcd, "inf": inf, "isqrt": isqrt,
    "lcm": lcm, "log": log, "log2": log2, "perm": perm, "sqrt": sqrt,
    "operator": operator, "re": re, "string": string, "sys": sys, "typing": typing,
    "Any": Any, "Callable": Callable, "Dict": Dict, "Iterable": Iterable, "List": List,
    "Optional": Optional, "Sequence": Sequence, "Set": Set, "Tuple": Tuple, "Union": Union,
}

def _alarm(signum, frame):
    raise TimeoutError("timeout")

signal.signal(signal.SIGALRM, _alarm)

payload = json.loads(sys.stdin.read())
code = payload["code"]
tests = payload["tests"]
fn_name = payload.get("fn_name") or ""

def norm_lines(s):
    return "\n".join(line.rstrip() for line in s.strip().splitlines())

passed = 0
for tc in tests:
    ttype = tc.get("testtype", "stdin")
    inp = tc.get("input", "")
    exp = tc.get("output", "")
    signal.alarm(2)
    try:
        if ttype == "functional" and fn_name:
            ns = dict(_STD_PRELUDE)
            exec(compile(code, "<sol>", "exec"), ns)
            fn = None
            if "Solution" in ns and hasattr(ns["Solution"], fn_name):
                fn = getattr(ns["Solution"](), fn_name)
            elif fn_name in ns:
                fn = ns[fn_name]
            if fn is None:
                print(json.dumps({"ok": False, "passed": passed, "total": len(tests), "err": "no_fn"}))
                sys.exit(0)
            args = [json.loads(line) for line in inp.strip().splitlines() if line.strip()]
            res = fn(*args)
            expected = json.loads(exp.strip())
            ok = (res == expected)
        else:
            ns = {"__name__": "__main__"}
            old_stdin = sys.stdin
            buf = io.StringIO()
            sys.stdin = io.StringIO(inp)
            try:
                with redirect_stdout(buf):
                    exec(compile(code, "<sol>", "exec"), ns)
            finally:
                sys.stdin = old_stdin
            got = norm_lines(buf.getvalue())
            want = norm_lines(exp)
            ok = (got == want)
            if not ok:
                # Floating-point tolerance fallback for single-number outputs
                try:
                    ok = abs(float(got) - float(want)) <= 1e-4 * max(1.0, abs(float(want)))
                except Exception:
                    ok = False
        signal.alarm(0)
        if not ok:
            print(json.dumps({"ok": False, "passed": passed, "total": len(tests)}))
            sys.exit(0)
        passed += 1
    except Exception:
        signal.alarm(0)
        print(json.dumps({"ok": False, "passed": passed, "total": len(tests)}))
        sys.exit(0)

print(json.dumps({"ok": True, "passed": passed, "total": len(tests)}))
'''


def lcb_score(vis_text: str, test_cases: list[dict], fn_name: str = "") -> tuple[bool, int, int]:
    """Execute extracted Python code against LiveCodeBench test cases in an isolated subprocess."""
    code = extract_python_code(vis_text)
    if not code or not test_cases:
        return False, 0, len(test_cases)
    payload = json.dumps({"code": code, "tests": test_cases, "fn_name": fn_name})
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _LCB_HARNESS],
            input=payload,
            capture_output=True,
            text=True,
            timeout=25,
        )
        out = proc.stdout.strip().splitlines()
        if out:
            res = json.loads(out[-1])
            return bool(res.get("ok")), int(res.get("passed", 0)), int(res.get("total", len(test_cases)))
    except Exception:
        pass
    return False, 0, len(test_cases)


def extract_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth > 0:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip()


def parse_gsm8k(text: str) -> str:
    b = extract_boxed(text)
    if b is not None:
        m = re.findall(r"-?\d[\d,]*(?:\.\d+)?", b)
        if m:
            return m[-1].replace(",", "").rstrip(".")
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else "[invalid]"


def score_gsm8k(pred: str, gold: str) -> bool:
    if pred == gold:
        return True
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def score_math500(text: str, gold_norm: str) -> tuple[str, bool]:
    b = extract_boxed(text)
    if b is None:
        m = re.search(r"(?:final answer is|answer is)\s*[:$]?\s*([^\n.]+)", text, re.I)
        b = m.group(1).strip() if m else ""
    if not b:
        return "[invalid]", False
    pred_norm = minerva_norm(b)
    if pred_norm == gold_norm:
        return pred_norm, True
    try:
        from math_verify import parse, verify
        if verify(parse(f"${gold_norm}$"), parse(f"${b}$")):
            return pred_norm, True
    except Exception:
        pass
    return pred_norm, False


def score_bbh(text: str, gold: str) -> tuple[str, bool]:
    pred = bbh_answer(text)
    if pred == "[invalidanswer]":
        b = extract_boxed(text)
        if b:
            pred = b.strip()
        else:
            m = re.search(r"answer\s*(?:is|:)\s*([^\n]+)", text, re.I)
            if m:
                pred = m.group(1).strip().rstrip(".")
    g = gold.strip()
    ok = pred.strip().lower() == g.lower() or pred.strip("() ").lower() == g.strip("() ").lower()
    return pred, ok


def score_row(bench: str, row: dict, raw_text: str, gen_tokens: int, reason: str) -> dict:
    vis = extract_visible_text(raw_text)
    base = {
        "id": row["id"],
        "task": row["task"],
        "generated_tokens": gen_tokens,
        "stop_reason": reason,
        "text": vis[:800],
    }
    if bench == "gsm8k":
        pred = parse_gsm8k(vis)
        return {**base, "pred": pred, "gold": row["answer"], "correct": score_gsm8k(pred, row["answer"])}
    if bench.startswith("math500"):
        pred, ok = score_math500(vis, row["answer"])
        return {**base, "pred": pred, "gold": row["answer"], "correct": ok}
    if bench == "mmlu_pro":
        pred = parse_mcq(vis, "ABCDEFGHIJ")
        return {**base, "pred": pred, "gold": row["answer"], "correct": pred == row["answer"]}
    if bench == "arc_c":
        pred = parse_mcq(vis, "ABCDE")
        return {**base, "pred": pred, "gold": row["answer"], "correct": pred == row["answer"]}
    if bench.startswith("longbench_v2"):
        pred = parse_mcq(vis, "ABCD")
        return {
            **base,
            "length_bucket": row.get("length_bucket"),
            "difficulty": row.get("difficulty"),
            "pred": pred,
            "gold": row["answer"],
            "correct": pred == row["answer"],
        }
    if bench == "bbh":
        pred, ok = score_bbh(vis, row["answer"])
        return {**base, "pred": pred, "gold": row["answer"], "correct": ok}
    if bench.startswith("ruler"):
        sc = float(ruler_score(row["task"], vis, row["answers"]))
        return {**base, "score": sc}
    if bench == "helmet":
        sc = float(helmet_score(row, vis))
        return {**base, "length": row["length"], "score": sc}
    if bench.startswith("graphwalks"):
        sc = float(graphwalks_f1(vis, row["answer_nodes"]))
        return {**base, "prompt_tokens": len(row["prompt_ids"]), "score": sc}
    if bench.startswith("mrcr"):
        sc = float(mrcr_score(vis, row["answer"], row["random_string_to_prepend"]))
        return {**base, "prompt_tokens": len(row["prompt_ids"]), "score": sc}
    if bench.startswith("apex"):
        pred, ok = score_math500(vis, row["answer"])
        return {**base, "pred": pred, "gold": row["answer"], "correct": ok}
    if bench.startswith("gpqa"):
        pred = parse_mcq(vis, "ABCD")
        return {**base, "pred": pred, "gold": row["answer"], "correct": pred == row["answer"]}
    if bench.startswith("bfcl"):
        calls, sc = bfcl_score(raw_text, row.get("ground_truth"), row.get("task", ""))
        return {**base, "pred_calls": calls, "score": sc, "correct": sc >= 1.0}
    if bench.startswith("livecodebench") or bench.startswith("lcb"):
        ok, passed, total = lcb_score(vis, row["test_cases"], row.get("fn_name", ""))
        return {**base, "difficulty": row.get("difficulty", ""), "passed": passed, "total_tests": total, "correct": ok}
    raise ValueError(f"Unknown bench: {bench}")
