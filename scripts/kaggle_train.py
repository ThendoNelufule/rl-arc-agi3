"""Run training jobs on Kaggle (internet OFF) and pull the results back.

Builds kaggle_runs/<name>/ = a self-contained script kernel that embeds rl/*.py + splits.json,
installs arc-agi (competition wheels) + SB3 (arc3-rl-assets) offline, points ENVIRONMENTS_DIR at
the competition's game files, runs the given training commands IN PARALLEL, and leaves
experiments/train/<tag>/ in /kaggle/working as kernel output.

  push : .venv/bin/python scripts/kaggle_train.py push NAME --accel t4 \
             --job "train_dqn.py --tag X --steps 400000 --device cuda" [--job ...]
  pull : .venv/bin/python scripts/kaggle_train.py pull NAME   # -> experiments/train/<tag>/
  status: .venv/bin/python scripts/kaggle_train.py status NAME
"""
import argparse, base64, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "thendonelufule"
SHIP = ["rl/arc_gym.py", "rl/nets.py", "rl/replay.py", "rl/sb3_nets.py", "rl/policies.py",
        "rl/train_dqn.py", "rl/train_ppo.py", "rl/rnd.py", "splits.json"]
ACCEL = {"cpu": dict(enable_gpu=False), "t4": dict(enable_gpu=True, machine_shape="NvidiaTeslaT4"),
         "p100": dict(enable_gpu=True, machine_shape="NvidiaTeslaP100")}

RUNNER = r'''
import base64, glob, os, subprocess, sys, time
W = "/kaggle/working"
for rel, b in FILES.items():
    p = os.path.join(W, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "wb").write(base64.b64decode(b))
def find(name):
    for pat in ("/kaggle/input/" + name, "/kaggle/input/*/" + name, "/kaggle/input/*/*/" + name,
                "/kaggle/input/*/*/*/" + name):
        h = sorted(glob.glob(pat))
        if h:
            return h[0]
    raise FileNotFoundError(name)
assets, comp = find("arc3-rl-assets"), find("arc_agi_3_wheels")
envdir = find("environment_files")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", comp,
                    "--find-links", assets + "/wheels", "arc-agi", "stable-baselines3", "sb3-contrib"],
                   capture_output=True, text=True)
print("PIP", r.returncode, r.stderr[-500:])
os.environ["ENVIRONMENTS_DIR"] = envdir
os.environ["ARC3_ARCADE_MODE"] = "offline"   # no internet: never ask the API for a key
print("ENVDIR", envdir, len(os.listdir(envdir)), "games | cpus", os.cpu_count())
subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
t0 = time.time()
procs = []
for i, job in enumerate(JOBS):
    log = open(f"{W}/job{i}.log", "w")
    procs.append((job, subprocess.Popen([sys.executable, "-u", f"{W}/rl/" + job.split()[0]] + job.split()[1:],
                                        cwd=W, stdout=log, stderr=subprocess.STDOUT)))
for job, p in procs:
    p.wait()
    print("JOB", p.returncode, job)
for i in range(len(JOBS)):
    print(f"--- job{i} tail ---"); print("".join(open(f"{W}/job{i}.log").readlines()[-6:]))
print("WALL_S", round(time.time() - t0, 1))
import shutil
shutil.rmtree(f"{W}/rl", ignore_errors=True); os.remove(f"{W}/splits.json")
'''


def kaggle(*a):
    env = dict(os.environ, KAGGLE_API_TOKEN=(ROOT / ".kaggle" / "access_token").read_text().strip())
    return subprocess.run([str(ROOT / ".venv" / "Scripts" / "kaggle"), *a], env=env,
                          capture_output=True, text=True)


def push(name, accel, jobs):
    d = ROOT / "kaggle_runs" / name
    d.mkdir(parents=True, exist_ok=True)
    files = {rel: base64.b64encode((ROOT / rel).read_bytes()).decode() for rel in SHIP}
    (d / "run.py").write_text(f"FILES = {files!r}\nJOBS = {jobs!r}\n" + RUNNER)
    meta = dict(id=f"{USER}/arc3-train-{name}", title=f"arc3-train-{name}", code_file="run.py",
                language="python", kernel_type="script", is_private=True, enable_tpu=False,
                enable_internet=False, dataset_sources=[f"{USER}/arc3-rl-assets"], kernel_sources=[],
                competition_sources=["arc-prize-2026-arc-agi-3"], model_sources=[], **ACCEL[accel])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    (d / "jobs.json").write_text(json.dumps(dict(accel=accel, jobs=jobs), indent=2))
    r = kaggle("kernels", "push", "-p", str(d))
    print(r.stdout.strip(), r.stderr.strip()[-500:])


def status(name):
    r = kaggle("kernels", "status", f"{USER}/arc3-train-{name}")
    print(r.stdout.strip() or r.stderr.strip())


def pull(name):
    d = ROOT / "kaggle_runs" / name / "out"
    shutil.rmtree(d, ignore_errors=True)
    r = kaggle("kernels", "output", f"{USER}/arc3-train-{name}", "-p", str(d))
    print(r.stdout.strip()[-300:], r.stderr.strip()[-300:])
    src = d / "experiments" / "train"
    for run in sorted(src.glob("*")) if src.exists() else []:
        dst = ROOT / "experiments" / "train" / run.name
        if dst.exists():
            print("SKIP (exists locally)", dst); continue
        shutil.copytree(run, dst)
        (dst / "trained_on.txt").write_text(f"kaggle kernel {USER}/arc3-train-{name}\n")
        print("pulled", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull", "status"])
    ap.add_argument("name")
    ap.add_argument("--accel", default="t4", choices=sorted(ACCEL))
    ap.add_argument("--job", action="append", default=[])
    a = ap.parse_args()
    if a.cmd == "push":
        if not a.job:
            sys.exit("need at least one --job")
        push(a.name, a.accel, a.job)
    else:
        {"pull": pull, "status": status}[a.cmd](a.name)
