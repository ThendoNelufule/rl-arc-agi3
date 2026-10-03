"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "cpu"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"

# Private Kaggle dataset with offline wheels + trained weights (kaggle_dataset/).
ASSETS_SLUG = "thendonelufule/arc3-rl-assets"
# rl/ modules the agent imports; written to /tmp/rl on Kaggle (same code as local eval).
# Image of the original starter kernel (py3.12.13, torch 2.10.0+cpu, numpy 2.0.2), verified in P1.4.
# Pinned on every per-version submission kernel so all versions run in the same environment.
DOCKER_IMAGE = ("gcr.io/kaggle-images/python@sha256:"
                "dafd4ce5668bbf1ad422e4c109e0f18c9623c3a7c7f48b0235f13142755c40b9")
RL_MODULES = ["arc_gym.py", "nets.py", "replay.py", "sb3_nets.py", "policies.py"]


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def pin_agent(body: str, policy: str, weights: str | None) -> str:
    """Freeze the agent's POLICY/WEIGHTS constants for this submission."""
    out = []
    for line in body.splitlines(keepends=True):
        if line.startswith("POLICY = "):
            line = f"POLICY = {policy!r}\n"
        elif line.startswith("WEIGHTS = "):
            line = f"WEIGHTS = {weights!r}\n"
        out.append(line)
    return "".join(out)


def build(policy: str = "random", weights: str | None = None) -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = pin_agent(AGENT_SRC.read_text(), policy, weights)
    rl_cells = [code_cell(f"%%writefile /tmp/rl/{m}\n" + (ROOT / "rl" / m).read_text())
                for m in RL_MODULES]
    smoke_cell = code_cell(dedent(
        f"""\
        # Commit-time smoke test: the submitted policy + weights load and act offline.
        import sys, numpy as np
        sys.path.insert(0, '/tmp/rl')
        from policies import make_policy
        _w = {weights!r}
        _ck = (os.environ['ARC3_ASSETS'] + '/weights/' + _w + '.pt') if _w else None
        _p = make_policy({policy!r}, ckpt=_ck); _p.reset('smoke', 0)
        _o = np.zeros((1, 64, 64), np.uint8); _m = np.ones(70, bool)
        for _ in range(40):
            _a = _p.act(_o, _m, {{}}); _p.observe(_o, _a, 0.0, _o, False, {{}})
        print('SMOKE OK policy={policy} weights={weights} last_action', _a)
        """
    ))

    # Offline install: competition wheels (arc-agi) + our assets dataset
    # (SB3 wheels, weights). Verified internet-OFF in P1.4 (arc3-offline-verify).
    # Mount paths differ between Kaggle images (/kaggle/input/<slug> on older
    # pinned images, /kaggle/input/datasets/<user>/<slug> on newer), so glob.
    install_cell = code_cell(dedent(
        f"""\
        import glob, os, subprocess, sys, platform
        def _find(name):
            for pat in ('/kaggle/input/' + name, '/kaggle/input/*/' + name,
                        '/kaggle/input/*/*/' + name, '/kaggle/input/*/*/*/' + name):
                hits = sorted(glob.glob(pat))
                if hits:
                    return hits[0]
            raise FileNotFoundError(name)
        ASSETS = _find('{ASSETS_SLUG.split('/')[1]}')
        COMP_WHEELS = _find('arc_agi_3_wheels')
        os.environ['ARC3_ASSETS'] = ASSETS   # read by the agent to locate weights
        os.environ['ARC3_RL_DIR'] = '/tmp/rl'  # rl/*.py written by the next cells
        os.makedirs('/tmp/rl', exist_ok=True)
        print('ASSETS', ASSETS, '| COMP_WHEELS', COMP_WHEELS)
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
                            '--find-links', COMP_WHEELS, '--find-links', ASSETS + '/wheels',
                            'arc-agi', 'python-dotenv', 'stable-baselines3', 'sb3-contrib'],
                           capture_output=True, text=True)
        print(r.stdout[-800:], r.stderr[-800:])
        assert r.returncode == 0, 'offline pip install failed'
        import torch, numpy, gymnasium, stable_baselines3
        print('ENV python', platform.python_version(), '| torch', torch.__version__,
              '| numpy', numpy.__version__, '| gymnasium', gymnasium.__version__,
              '| sb3', stable_baselines3.__version__, '| threads', torch.get_num_threads())
        """
    ))

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            *rl_cells,
            smoke_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="random")
    ap.add_argument("--weights", default=None, help="run tag; needs weights/<tag>.pt in the dataset")
    ap.add_argument("--slug", default=None, help="per-version kernel, e.g. team-10-dqn-base -> arc3-team-10-dqn-base")
    a = ap.parse_args()
    if a.weights and not (ROOT / "kaggle_dataset" / "weights" / f"{a.weights}.pt").exists():
        raise SystemExit(f"kaggle_dataset/weights/{a.weights}.pt missing: copy it and `make dataset-push` first")
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(a.policy, a.weights), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR}, policy: {a.policy}, weights: {a.weights})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        changed = False
        if a.slug:
            meta["id"] = f"thendonelufule/arc3-{a.slug}"
            meta["title"] = f"arc3-{a.slug}"
            meta["docker_image"] = DOCKER_IMAGE
            changed = True
            print(f"[build_notebook] Kernel {meta['id']} (image pinned)")
        if ASSETS_SLUG not in meta.get("dataset_sources", []):
            meta["dataset_sources"] = meta.get("dataset_sources", []) + [ASSETS_SLUG]
            changed = True
            print(f"[build_notebook] Attached dataset {ASSETS_SLUG}")
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            changed = True
            print(f"[build_notebook] Synced enable_gpu={wanted}")
        if changed:
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
