"""Archive the Lev decision-model work (/dev/shm/levwork) to the private ModelScope repo, once, at project close.

usage: python3 backup_lev.py BOX        (BOX = a1 | a2: the folder in the repo; token in /root/.ms.token)
backup_ms.py covers /dev/shm/big-small-ouro only; the Lev runs of Appendix M live beside it in /dev/shm/levwork, which
is also a tmpfs. Uploaded to BOX/lev/ in the repo:
  train/<run>/          one folder per Lev training run: config, metrics, log and the trained adapter
  out/, rows/           LevBench development outputs and the scored rows behind the Lev figure
  logs/                 the per-run training logs kept beside the folders
  *.jsonl               the development split and the sampled rows
Not uploaded: hf/ (base models, re-downloadable) and the released Lev adapters, LevBench and Lev-train-v2 (third-party
downloads with their own licences).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BOX = sys.argv[1]
L = Path("/dev/shm/levwork")
STAGE = Path("/dev/shm/lev-backup") / BOX
REPO = "torushy/untying-prefill-decode-backup"
STAGE.mkdir(parents=True, exist_ok=True)


def pack(src, dst):
    if dst.exists() and dst.stat().st_mtime >= max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()),
                                                   default=0):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    subprocess.run(["tar", "czf", str(tmp), "-C", str(src.parent), src.name], check=True)
    tmp.rename(dst)


def copy(src, dst):
    if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


if (L / "train").exists():
    for d in sorted((L / "train").iterdir()):
        if d.is_dir():
            pack(d, STAGE / "train" / f"{d.name}.tar.gz")
for name in ("out", "rows"):
    if (L / name).exists():
        pack(L / name, STAGE / f"{name}.tar.gz")
for f in sorted(L.glob("*.log")):
    copy(f, STAGE / "logs" / f.name)
for f in sorted(L.glob("*.jsonl")):
    copy(f, STAGE / f.name)

manifest_path = STAGE.parent / f".uploaded-lev-{BOX}.json"
done = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
todo = [p for p in STAGE.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
        and done.get(str(p.relative_to(STAGE))) != [p.stat().st_size, int(p.stat().st_mtime)]]
print(f"{BOX} lev: {len(todo)} new or changed files, {sum(p.stat().st_size for p in todo) / 1e9:.2f} GB", flush=True)
if todo:
    out = STAGE.parent / f"outbox-lev-{BOX}"
    shutil.rmtree(out, ignore_errors=True)
    for p in todo:
        q = out / p.relative_to(STAGE); q.parent.mkdir(parents=True, exist_ok=True); os.link(p, q)
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(open("/root/.ms.token").read().strip())
    for attempt in range(3):
        try:
            api.upload_folder(REPO, folder_path=str(out), path_in_repo=f"{BOX}/lev",
                              commit_message=f"lev backup {BOX}: {len(todo)} files")
            break
        except Exception as e:
            print(f"{BOX} lev: upload attempt {attempt + 1} failed: {str(e)[:200]}", flush=True)
    else:
        sys.exit(1)
    for p in todo:
        done[str(p.relative_to(STAGE))] = [p.stat().st_size, int(p.stat().st_mtime)]
    manifest_path.write_text(json.dumps(done))
    shutil.rmtree(out, ignore_errors=True)
    print(f"{BOX} lev: uploaded", flush=True)
