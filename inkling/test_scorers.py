"""Adversarial unit tests for inkling.scoring (locking down all Verifier-audited edge cases)."""

from __future__ import annotations

from inkling.scoring import extract_python_code, extract_visible_text, graphwalks_f1, lcb_score, mrcr_score, parse_mcq


def test_parse_mcq():
    # 1. Exact \boxed{} & \boxed{\text{X}}
    assert parse_mcq(r"Reasoning... \boxed{\text{C}}") == "C"
    assert parse_mcq(r"The answer is clear... \boxed{B}") == "B"
    # 2. \boxed{B: 3.5 \text{ C}} with physical unit 'C' inside choice text
    assert parse_mcq(r"Final: \boxed{B: 3.5 \text{ C}}") == "B"
    # 3. Prose "answer is a positive trend" must NOT hijack '(B)'
    assert parse_mcq("First check whether the answer is a positive trend.\nTherefore, the correct choice is (B).") == "B"
    # 4. Optional colon after 'is' ("is: (C)" and "is: B.")
    assert parse_mcq("The answer is a bit subtle...\n**The correct answer is: (C)**") == "C"
    assert parse_mcq("The correct answer is: B.\nThis matches paragraph 4.") == "B"
    # 5. Truncated thinking without <|content_text|> must return [invalid]
    assert parse_mcq(extract_visible_text("<|content_thinking|>maybe A or B")) == "[invalid]"


def test_graphwalks_f1():
    assert graphwalks_f1("Final Answer: [n1, n2]", ["n1", "n2"]) == 1.0
    assert graphwalks_f1("Final Answer:\n['n1', 'n2']", ["n1", "n2"]) == 1.0
    assert graphwalks_f1("**Final Answer:** ['n1', 'n2']", ["n1", "n2"]) == 1.0
    assert graphwalks_f1("Final Answer: `['n1', 'n2']`", ["n1", "n2"]) == 1.0
    assert graphwalks_f1("Final Answer: [`n1`, `n2`]", ["n1", "n2"]) == 1.0


def test_lcb_score():
    # 1. Code block priority: explicit python tag beats trailing untagged sample output block
    vis = "```python\nprint(42)\n```\nSample Output:\n```\n42\n```\n"
    assert extract_python_code(vis) == "print(42)"

    # 2. sys.stdin.buffer + sys.stdout.buffer + SystemExit(0)
    code_buf = (
        "```python\nimport sys\n"
        "data = sys.stdin.buffer.read().split()\n"
        "sys.stdout.buffer.write(f'{int(data[0])+int(data[1])}\\n'.encode())\n"
        "sys.exit(0)\n```"
    )
    ok, p, t = lcb_score(code_buf, [{"testtype": "stdin", "input": "12 30\n", "output": "42\n"}])
    assert ok and p == 1 and t == 1

    # 3. Low-level os.read(0, ...) / os.write(1, ...) FastIO template
    code_fd = (
        "```python\nimport os\n"
        "data = os.read(0, 1 << 20).split()\n"
        "os.write(1, f'{int(data[0])*int(data[1])}\\n'.encode())\n```"
    )
    ok, p, t = lcb_score(code_fd, [{"testtype": "stdin", "input": "6 7\n", "output": "42\n"}])
    assert ok and p == 1 and t == 1

    # 4. sys.exit(1) must fail
    ok, _, _ = lcb_score("```python\nimport sys\nprint(42)\nsys.exit(1)\n```", [{"testtype": "stdin", "input": "", "output": "42\n"}])
    assert not ok

    # 5. Deep recursion + >4300-digit integer
    code_rec = "```python\nimport sys\ndef f(n): return 0 if n == 0 else f(n-1)\nf(int(input()))\nprint(len(str(10**5000)))\n```"
    ok, p, _ = lcb_score(code_rec, [{"testtype": "stdin", "input": "4000\n", "output": "5001\n"}])
    assert ok and p == 1

    # 6. Functional mode: tuple -> list normalization + float tolerance + std typing prelude
    code_fn = "```python\nclass Solution:\n    def pair(self, a: int, b: int) -> Tuple[int, float]:\n        return (a + b, 1.0 / 3.0)\n```"
    ok, p, _ = lcb_score(
        code_fn,
        [{"testtype": "functional", "input": "2\n3\n", "output": "[5, 0.333333]\n"}],
        fn_name="pair",
    )
    assert ok and p == 1


def test_mrcr():
    assert mrcr_score("XYZhello world", "XYZhello world", "XYZ") == 1.0
    assert mrcr_score("hello world", "XYZhello world", "XYZ") == 0.0


if __name__ == "__main__":
    test_parse_mcq()
    test_graphwalks_f1()
    test_lcb_score()
    test_mrcr()
    print("ALL VERIFIER ADVERSARIAL SCORER TESTS PASSED!")
