"""Self-contained exact benchmark scorers (identical to ouro/score.py & ouro/evaluate_bench.py)."""

from __future__ import annotations

from collections import Counter
import math
import re
import string


RULER_TASKS = ("niah_single_1", "niah_multikey_2", "vt", "cwe")

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
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
