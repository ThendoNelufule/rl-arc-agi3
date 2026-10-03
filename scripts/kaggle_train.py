"""Run training AND evaluation jobs on Kaggle (internet OFF) and pull the results back.

Builds kaggle_runs/<name>/ = a self-contained script kernel that embeds rl/*.py + splits.json,
installs arc-agi (competition wheels) + SB3 (arc3-rl-assets) offline, points ENVIRONMENTS_DIR at
the competition's game files (arcade in offline mode), runs the given jobs (at most --maxpar at
once, OMP threads per job = --threads), and leaves experiments/train/<tag>/ and
experiments/runs/<tag>/ in /kaggle/working as kernel output. "{ASSETS}" in a job is replaced by
the dataset mount path (checkpoints live in arc3-rl-assets/weights/ckpt/).

  push : .venv/bin/python scripts/kaggle_train.py push NAME --accel t4 \
             --job "train_dqn.py --tag X --steps 400000 --device cuda" [--job ...]
         .venv/bin/python scripts/kaggle_train.py push NAME --accel cpu --maxpar 4 --threads 1 \
             --job "evaluate.py --policy dqn-online --ckpt {ASSETS}/weights/ckpt/X.pt --split dev --tag X-dev --record"
  pull : .venv/bin/python scripts/kaggle_train.py pull NAME [--overwrite]
  status: .venv/bin/python scripts/kaggle_train.py status NAME
"""
import argparse, base64, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "thendonelufule"
SHIP = ["rl/arc_gym.py", "rl/nets.py", "rl/replay.py", "rl/sb3_nets.py", "rl/recurrent.py", "rl/policies.py",
        "rl/train_dqn.py", "rl/train_ppo.py", "rl/rnd.py", "rl/effect.py", "rl/evaluate.py", "splits.json"]
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
subprocess.run("grep -m1 'model name' /proc/cpuinfo; free -g | head -2", shell=True)
env = dict(os.environ)
if THREADS:
    env["OMP_NUM_THREADS"] = str(THREADS); env["MKL_NUM_THREADS"] = str(THREADS)
t0 = time.time()
queue, running = list(enumerate(JOBS)), []
while queue or running:
    while queue and len(running) < MAXPAR:
        i, job = queue.pop(0)
        job = job.replace("{ASSETS}", assets)
        log = open(f"{W}/job{i}.log", "w")
        running.append((job, time.time(), subprocess.Popen(
            [sys.executable, "-u", f"{W}/rl/" + job.split()[0]] + job.split()[1:],
            cwd=W, stdout=log, stderr=subprocess.STDOUT, env=env)))
    for item in list(running):
        if item[2].poll() is not None:
            running.remove(item)
            print("JOB", item[2].returncode, round(time.time() - item[1], 1), "s", item[0], flush=True)
    time.sleep(2)
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


def push(name, accel, jobs, maxpar=99, threads=0):
    d = ROOT / "kaggle_runs" / name
    d.mkdir(parents=True, exist_ok=True)
    files = {rel: base64.b64encode((ROOT / rel).read_bytes()).decode() for rel in SHIP}
    header = f"FILES = {files!r}\nJOBS = {jobs!r}\nMAXPAR = {maxpar}\nTHREADS = {threads}\n"
    (d / "run.py").write_text(header + RUNNER)
    meta = dict(id=f"{USER}/arc3-train-{name}", title=f"arc3-train-{name}", code_file="run.py",
                language="python", kernel_type="script", is_private=True, enable_tpu=False,
                enable_internet=False, dataset_sources=[f"{USER}/arc3-rl-assets"], kernel_sources=[],
                competition_sources=["arc-prize-2026-arc-agi-3"], model_sources=[], **ACCEL[accel])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    (d / "jobs.json").write_text(json.dumps(dict(accel=accel, jobs=jobs, maxpar=maxpar,
                                                 threads=threads), indent=2))
    r = kaggle("kernels", "push", "-p", str(d))
    print(r.stdout.strip(), r.stderr.strip()[-500:])


def status(name):
    r = kaggle("kernels", "status", f"{USER}/arc3-train-{name}")
    print(r.stdout.strip() or r.stderr.strip())


def pull(name, overwrite=False):
    d = ROOT / "kaggle_runs" / name / "out"
    shutil.rmtree(d, ignore_errors=True)
    r = kaggle("kernels", "output", f"{USER}/arc3-train-{name}", "-p", str(d))
    print(r.stdout.strip()[-300:], r.stderr.strip()[-300:])
    for kind, note in (("train", "trained_on.txt"), ("runs", "evaluated_on.txt")):
        src = d / "experiments" / kind
        for run in sorted(src.glob("*")) if src.exists() else []:
            dst = ROOT / "experiments" / kind / run.name
            if dst.exists():
                if not overwrite:
                    print("SKIP (exists locally)", dst)
                    continue
                shutil.rmtree(dst)
            shutil.copytree(run, dst)
            (dst / note).write_text(f"kaggle kernel {USER}/arc3-train-{name}\n")
            print("pulled", dst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull", "status"])
    ap.add_argument("name")
    ap.add_argument("--accel", default="t4", choices=sorted(ACCEL))
    ap.add_argument("--job", action="append", default=[])
    ap.add_argument("--maxpar", type=int, default=99, help="max jobs running at once")
    ap.add_argument("--threads", type=int, default=0, help="OMP threads per job (0 = default)")
    ap.add_argument("--overwrite", action="store_true", help="pull: replace existing local runs")
    a = ap.parse_args()
    if a.cmd == "push":
        if not a.job:
            sys.exit("need at least one --job")
        push(a.name, a.accel, a.job, a.maxpar, a.threads)
    elif a.cmd == "pull":
        pull(a.name, a.overwrite)
    else:
        status(a.name)
