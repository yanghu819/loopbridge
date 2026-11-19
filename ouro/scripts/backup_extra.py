"""Archive, once at project close, the working files that backup_ms.py does not cover, to BOX/extra/ in the private
ModelScope repo.

usage: python3 backup_extra.py BOX        (BOX = a1 | a2; token in /root/.ms.token)
What goes up:
  data/<name>.tar.gz    prepared data this box holds and the hourly backup never sent: the distillation corpus
                        (adapter-distill-200m), the Thinking mixture, the older mix-data, the local benchmark copies
                        (hf-lmeval, gpqa-data, ifeval-data) and the exported per-item scores (items-export*)
  pn.tar.gz             /root/pn: the paper-number and packaging logs, and the Huginn training-free check
  code-on-box.tar.gz    /root/big-small as the runs left it (without its virtual environment): the public repository
                        is a pruned copy of this tree, so this is what actually ran
  env/                  pip freeze, python and driver versions, and the base models' config, tokenizer and
                        modeling_*.py files with the SHA256 of each weight file (the weights themselves stay upstream)
Already covered elsewhere, so not sent: mix-data-48m, bench and bench-mini (in a1/data), the SenseNova and Nanbeige
mixtures and weights (that line is not in the paper), lmeval-pkgs and fwcache (package and download caches), and
msbak (an October restore copy of what is already in this repo).
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BOX = sys.argv[1]
M = Path("/dev/shm/big-small-ouro")
STAGE = Path("/dev/shm/extra-backup") / BOX
REPO = "torushy/untying-prefill-decode-backup"
DATA = ["adapter-distill-200m", "hf-lmeval", "gpqa-data", "ifeval-data", "items-export"]
if BOX == "a1":                      # identical on both boxes (file list and sizes compared), so sent once
    DATA += ["mix-think-48m", "mix-data"]
MODELS = ["model-2.6b", "model-1.4b", "model-2.6b-think", "model-1.4b-think"]
STAGE.mkdir(parents=True, exist_ok=True)


def pack(src, dst):
    if dst.exists() and dst.stat().st_mtime >= max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()),
                                                   default=0):
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    subprocess.run(["tar", "czf", str(tmp), "-C", str(src.parent), src.name], check=True)
    tmp.rename(dst)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


for name in DATA:
    if (M / name).exists():
        pack(M / name, STAGE / "data" / f"{name}.tar.gz")
for p in (Path("/dev/shm/items-export-new"), Path("/root/pn")):
    if p.exists():
        pack(p, STAGE / ("data" if p.parent == Path("/dev/shm") else ".") / f"{p.name}.tar.gz")

code, dst = Path("/root/big-small"), STAGE / "code-on-box.tar.gz"
if code.exists() and not dst.exists():
    tmp = dst.with_suffix(".tmp")
    subprocess.run(["tar", "czf", str(tmp), "--exclude=.venv", "--exclude=__pycache__",
                    "-C", str(code.parent), code.name], check=True)
    tmp.rename(dst)

env = STAGE / "env"
env.mkdir(parents=True, exist_ok=True)
(env / "pip-freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                                   text=True).stdout)
(env / "versions.txt").write_text(
    f"python {sys.version}\n"
    + subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout
    + subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                     capture_output=True, text=True).stdout)
models = {}
for name in MODELS:
    d = M / name
    if not d.exists():
        continue
    out = env / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        if f.suffix in (".safetensors", ".bin", ".pt"):
            weights[f.name] = [f.stat().st_size, sha256(f)]
        elif f.stat().st_size < 50 << 20:
            shutil.copy2(f, out / f.name)
    models[name] = weights
(env / "model-weight-sha256.json").write_text(json.dumps(models, indent=1))

manifest_path = STAGE.parent / f".uploaded-extra-{BOX}.json"
done = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
todo = [p for p in STAGE.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
        and done.get(str(p.relative_to(STAGE))) != [p.stat().st_size, int(p.stat().st_mtime)]]
print(f"{BOX} extra: {len(todo)} new or changed files, {sum(p.stat().st_size for p in todo) / 1e9:.2f} GB", flush=True)
if todo:
    out = STAGE.parent / f"outbox-extra-{BOX}"
    shutil.rmtree(out, ignore_errors=True)
    for p in todo:
        q = out / p.relative_to(STAGE); q.parent.mkdir(parents=True, exist_ok=True); os.link(p, q)
    from modelscope.hub.api import HubApi
    api = HubApi(); api.login(open("/root/.ms.token").read().strip())
    for attempt in range(3):
        try:
            api.upload_folder(REPO, folder_path=str(out), path_in_repo=f"{BOX}/extra",
                              commit_message=f"extra backup {BOX}: {len(todo)} files")
            break
        except Exception as e:
            print(f"{BOX} extra: upload attempt {attempt + 1} failed: {str(e)[:200]}", flush=True)
    else:
        sys.exit(1)
    for p in todo:
        done[str(p.relative_to(STAGE))] = [p.stat().st_size, int(p.stat().st_mtime)]
    manifest_path.write_text(json.dumps(done))
    shutil.rmtree(out, ignore_errors=True)
    print(f"{BOX} extra: uploaded", flush=True)
