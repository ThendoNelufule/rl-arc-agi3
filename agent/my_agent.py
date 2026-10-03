"""Kaggle / framework wrapper around any rl/policies.py Policy.

Same code path as rl/evaluate.py: frame -> (1,64,64) grid (arc_gym.frame_to_grid), legality
mask (arc_gym.mask_from_raw), Policy.act -> Discrete(70) index -> GameAction (arc_gym.decode_action).
Online policies get observe() with reward = level delta, exactly as in the eval harness.
Only the observable frame is used (no game-specific code, no hidden state).

Pick the agent with the three constants below; scripts/build_notebook.py ships rl/*.py
to /tmp/rl on Kaggle and the weights come from the arc3-rl-assets dataset.
"""
from __future__ import annotations

import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any

POLICY = os.environ.get("ARC3_POLICY", "random")   # name in rl/policies.py POLICIES
WEIGHTS = os.environ.get("ARC3_WEIGHTS") or None   # run tag, e.g. "TEAM-10-dqn-base"
BUDGET = 500        # actions per game, same as the eval protocol
# Kaggle wall-clock guard (notebook limit 9 h for 110 parallel games), measured from import:
LEARN_DEADLINE_S = 6.5 * 3600   # after this, no more test-time learning (act only)
STOP_DEADLINE_S = 8.0 * 3600    # after this, every game stops (is_done -> True)

_RL_DIR = os.environ.get("ARC3_RL_DIR") or str(Path(__file__).resolve().parents[1] / "rl")
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agents.agent import Agent  # noqa: E402
from arc_gym import decode_action, frame_to_grid, mask_from_raw  # noqa: E402
from policies import make_policy  # noqa: E402

_T0 = time.time()


def _elapsed() -> float:
    return time.time() - _T0


def _weights_path():
    if WEIGHTS is None:
        return None
    assets = os.environ.get("ARC3_ASSETS")
    if assets:                                   # Kaggle: dataset copy
        return str(Path(assets) / "weights" / f"{WEIGHTS}.pt")
    return str(Path(_RL_DIR).parent / "experiments" / "train" / WEIGHTS / "model.pt")


class MyAgent(Agent):
    MAX_ACTIONS = BUDGET

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        try:                      # one thread per game: the framework runs games in parallel
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        self.gid = self.game_id.split("-")[0]
        self.policy = make_policy(POLICY, ckpt=_weights_path())
        self.policy.reset(self.gid, zlib.crc32(self.gid.encode()))
        self.prev = None          # (obs, action, levels_completed) of the last real action

    @property
    def name(self) -> str:
        return f"{super().name}.{POLICY}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN or _elapsed() > STOP_DEADLINE_S

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        learn = _elapsed() < LEARN_DEADLINE_S
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            if learn and self.prev is not None and latest_frame.state is GameState.GAME_OVER:
                o, a, _ = self.prev
                self.policy.observe(o, a, 0.0, frame_to_grid(latest_frame)[None], True,
                                    {"game_id": self.gid, "state": "GAME_OVER",
                                     "levels_completed": latest_frame.levels_completed})
            self.prev = None
            return GameAction.RESET
        obs = frame_to_grid(latest_frame)[None]
        info = {"game_id": self.gid, "levels_completed": latest_frame.levels_completed,
                "state": latest_frame.state.name}
        if learn and self.prev is not None:
            o, a, lv = self.prev
            r = float(max(0, latest_frame.levels_completed - lv))
            self.policy.observe(o, a, r, obs, False, info)
        idx = self.policy.act(obs, mask_from_raw(latest_frame), info)
        action, data = decode_action(idx)
        if data is not None:
            action.set_data(data)
        action.reasoning = {"policy": POLICY, "index": int(idx)}
        self.prev = (obs, idx, latest_frame.levels_completed)
        return action
