# ARR Responsible NLP checklist — draft answers

Target: ACL 2027 via ARR, submission deadline 2027-01-04 (AoE). Fill these into the OpenReview form; sections are named as
in `paper/latex/main.tex` (appendix letters shift, so names are used). Items marked **TODO** need the author's decision.

## A. For every submission
- **A1 Limitations.** Yes: the unnumbered "Limitations" section after the Conclusion.
- **A2 Potential risks.** Yes, briefly: the method changes inference compute of existing open checkpoints and adds no
  new capabilities or data; risks are those of the underlying models. **TODO:** add one sentence to Limitations if
  the form requires it in the paper.

## B. Did you use or create scientific artifacts?
Yes. Models: Ouro-1.4B/2.6B and their Thinking versions, Lev decision models (Lev-1.4B/2.6B/3B).
Benchmarks: MMLU, MMLU-Pro, BBH, ARC-Challenge, HellaSwag, Winogrande, GSM8K, MATH-500, HumanEval+, MBPP+, RULER, HELMET,
LevBench (development set only). Training data: OpenThoughts-114k, Tulu 3 SFT mixture, LongAlpaca-12k, FineWeb-Edu,
Lev's S31 subset. Tools: lm-evaluation-harness, evalplus, math-verify, FlashAttention.
- **B1 Cite creators.** Yes: models and benchmarks in Section 4 (Experimental Setup), training data in the Training
  Details appendix, evaluation tools and the Minerva prompt in the Evaluation Details appendix (all cited since 11-16).
- **B2 Licenses.** **TODO:** list each artifact's license in the Evaluation Details appendix (Ouro code is Apache-2.0, see
  `third_party/ouro/LICENSE`; our code is MIT). Check the model cards for the others.
- **B3 Intended use.** Yes: all artifacts are used for research evaluation, consistent with their terms; benchmark
  questions are not redistributed (only per-item scores keyed by index), and LevBench's locked test split is never
  evaluated.
- **B4 Personal information / offensive content.** N/A: public benchmarks and corpora; no new data collected.
- **B5 Documentation of artifacts.** Yes: README, REPRODUCE.md, per-item scores.
- **B6 Statistics.** Yes: item counts per benchmark (Evaluation Details appendix; table and figure captions).

## C. Did you run computational experiments?
- **C1 Model size and compute.** Parameters: Section 3 (Method) and the Training Details appendix (63,307,776 trainable on Ouro-2.6B, 2.4%). GPU type:
  NVIDIA B300. Training time: 6.75 h on 4 GPUs for the Ouro-2.6B adapter (Training Details appendix). **TODO:** total GPU hours of all
  runs (sum the training and evaluation logs; state an estimate in the Training Details appendix).
- **C2 Hyperparameters.** Yes: Training Details appendix (optimiser, schedule, batch, tokens). The learning rate was
  checked at 0.3x and 3x for the distilled controls and for Ours (Method Details appendix, "Baseline learning rate");
  both are reported at the default.
- **C3 Descriptive statistics.** Yes: paired bootstrap 95% intervals (5,000 resamples, seed 0) everywhere; three
  training seeds (mean and s.d.) in the Table 1 caption and Sections 5.2, 5.4, 5.6 and 5.7; four seeds for the
  decision models; single runs otherwise, stated.
- **C4 Existing packages.** Yes: lm-evaluation-harness and evalplus (cited, versions in `environments/ouro`),
  math-verify, FlashAttention (cited), Hugging Face transformers.

## D. Human annotators / participants
- No (D1–D5 N/A).

## E. AI assistants in research or writing
- **E1.** Yes. **TODO (author's wording):** e.g. "We used AI assistants (Claude, Codex) for writing and running
  experiment code, for drafting and editing text, and for internal review of claims. All results are recomputed from
  released per-item scores by the released script; the authors checked every claim."

## Anonymity for review
- The review PDF says "Anonymous ACL submission"; the arXiv build adds the author and repository link.
- **TODO:** the code link for review must be anonymous (e.g. an anonymized mirror of this repository); the public
  repository name and the README author line would de-anonymize. Do not post the arXiv version during the ARR
  anonymity period if the target cycle forbids it (check the cycle's preprint policy).
