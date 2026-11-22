"""Adversarial unit tests for Inkling-2-Mini scorers and answer parsers (H0 verification gate)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkling.scoring import bfcl_score, extract_visible_text, graphwalks_f1, lcb_score, minerva_norm, parse_mcq


def test_extract_visible_text():
    # Truncated inside thinking without <|content_text|> MUST return empty string
    raw_trunc = "<|content_thinking|>Let us consider option A or maybe B because a value is..."
    assert extract_visible_text(raw_trunc) == "", f"Leaked thinking text: {extract_visible_text(raw_trunc)!r}"

    # Complete output with <|content_text|> extracts only after <|content_text|>
    raw_ok = "<|content_thinking|>Maybe A? No, B.<|content_text|>The correct answer is (B).<|end_message|>"
    assert extract_visible_text(raw_ok) == "The correct answer is (B)."


def test_parse_mcq_adversarial():
    # Verifier #6 Bug 1: prose "answer is clear" MUST NOT hijack \boxed{B}
    assert parse_mcq("The answer is clear from symmetry, so \\boxed{B}.", "ABCD") == "B"
    assert parse_mcq("The answer is derived from Gauss's law: \\boxed{A}.", "ABCD") == "A"
    # Markdown bold + pronoun "I" in A..J
    assert parse_mcq("**Answer:** E\nI verified this directly.", "ABCDEFGHIJ") == "E"
    # \boxed{\text{C}} after earlier parenthesized choice
    assert parse_mcq("Choice (A) fails. Thus \\boxed{\\text{C}}.", "ABCD") == "C"
    # Lowercase article "a" must NEVER match option A
    assert parse_mcq("Let us consider a function over a domain.", "ABCD") == "[invalid]"
    # Empty truncated text -> [invalid]
    assert parse_mcq("", "ABCD") == "[invalid]"


def test_graphwalks_f1_multiline():
    # Verifier #6 Bug 4: newline between Final Answer: and list
    assert graphwalks_f1("Steps...\nFinal Answer:\n['node_a', 'node_b']", ["node_a", "node_b"]) == 1.0
    assert graphwalks_f1("Final Answer: [node_a, node_c]", ["node_a", "node_b"]) == 0.5
    assert graphwalks_f1("", ["node_a"]) == 0.0


def test_bfcl_irrelevance_truncation():
    # Verifier #6 Bug 2: truncated thinking MUST NOT get 1.0 on live_irrelevance
    raw_trunc = "<|content_thinking|>Checking whether any tool applies..."
    _, sc = bfcl_score(raw_trunc, [], "live_irrelevance")
    assert sc == 0.0, f"Truncated thinking falsely rewarded on live_irrelevance: {sc}"

    raw_valid = "<|content_thinking|>No tool fits.<|content_text|>I cannot answer this with the given tools.<|end_message|>"
    _, sc_ok = bfcl_score(raw_valid, [], "live_irrelevance")
    assert sc_ok == 1.0


def test_lcb_python312_prelude():
    # Verifier #6 Bug 3: functional LeetCode starter_code with List[int], Optional, Counter, inf
    vis = """Here is the solution:
```python
class Solution:
    def twoSumCount(self, nums: List[int], target: Optional[int]) -> int:
        c = Counter(nums)
        best = inf
        return sum(1 for x in nums if target is not None and x < best and c[x] > 0)
```
"""
    ok, passed, total = lcb_score(
        vis,
        [{"testtype": "functional", "input": "[1, 2, 3]\n5", "output": "3"}],
        fn_name="twoSumCount",
    )
    assert ok and passed == 1 and total == 1, f"LCB functional prelude failed: ok={ok}, passed={passed}/{total}"


def test_minerva_norm_preserves_variable_a():
    # Verifier #6 Bug 4: "a + b" must not strip variable "a"
    assert minerva_norm("a + b") == "a+b", f"Got {minerva_norm('a + b')!r}"


if __name__ == "__main__":
    test_extract_visible_text()
    test_parse_mcq_adversarial()
    test_graphwalks_f1_multiline()
    test_bfcl_irrelevance_truncation()
    test_lcb_python312_prelude()
    test_minerva_norm_preserves_variable_a()
    print("ALL 6 ADVERSARIAL SCORER TESTS PASSED!")
