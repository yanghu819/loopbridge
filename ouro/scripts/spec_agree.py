"""Speculative decoding with a one-loop drafter for Original 4/4 (greedy): how many drafted tokens the target accepts.

For each timed request (the same requests as time_requests.py: prompts from SAMPLES_DIR, chosen by a hash of the item
id) the drafter is teacher-forced on Original 4/4's greedy answer (the text in TARGET.jsonl, re-tokenised): at every
answer position it predicts the next token from the target's own prefix. In greedy speculative decoding a drafted token
is accepted exactly when it equals the target's next token given the accepted prefix, which is this teacher-forced
prefix, so the per-position agreement determines every block: paper_numbers.spec_decoding() simulates blocks of k
drafts from it. The drafter's prompt is prefilled as in its own setting (Ours 4/1: four loops, as the target does).

usage: spec_agree.py MODEL_DIR COND CKPT|- SAMPLES_DIR N TARGET.jsonl OUT.jsonl
  COND: a drafter condition of bench_throughput.make (native1st, D1st, a4x1pm, ...); N: requests per task file group
  OUT rows: id, cond, answer_tokens, agree (a string of 0/1, one per answer position)
"""
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

from ouro.bench_throughput import make

model_dir, cond, ckpt, samples_dir, n, target, out = sys.argv[1:8]
n, out = int(n), Path(out)
MAX_LENGTH = 16384   # as the evaluation and time_requests.py

newest = {}
for f in glob.glob(os.path.join(samples_dir, "*", "samples_*.jsonl")):
    task, stamp = os.path.basename(f)[:-len(".jsonl")].split("samples_")[1].rsplit("_", 1)
    newest[task] = max(newest.get(task, (stamp, f)), (stamp, f))
requests = {}
for task, (_, f) in newest.items():
    for r in map(json.loads, open(f)):
        requests.setdefault(f"{task}/{r['doc_id']}", r)
key = lambda i: hashlib.md5(i.encode()).hexdigest()
chosen = sorted(requests, key=key)[:n]
answers = {r["id"]: r["text"] for r in map(json.loads, open(target))}

tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
runner = make(cond, model_dir, None if ckpt == "-" else ckpt, merge=ckpt != "-")


def prompt(r):
    context, kw = r["arguments"]["gen_args_0"]["arg_0"], r["arguments"]["gen_args_0"]["arg_1"]
    max_new = int(kw.get("max_gen_toks", 256))
    ids, limit = tok.encode(context, add_special_tokens=False), MAX_LENGTH - max_new
    if len(ids) > limit:
        ids = ids[:limit // 2] + ids[len(ids) - (limit - limit // 2):]
    return ids


with torch.inference_mode(), open(out, "w") as f:
    for i in chosen:
        if i not in answers:
            continue
        ids = prompt(requests[i])
        ans = tok.encode(answers[i], add_special_tokens=False)
        logits, cache = runner.prefill(torch.tensor([ids], device="cuda"))
        agree = []
        for t in ans:
            agree.append("1" if int(logits.argmax(-1)) == t else "0")
            logits, cache = runner.decode(torch.tensor([[t]], device="cuda"), cache)
        del cache
        f.write(json.dumps(dict(id=i, cond=cond, answer_tokens=len(ans), agree="".join(agree))) + "\n")
        f.flush()
