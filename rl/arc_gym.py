"""Gymnasium adapter for ARC-AGI-3 (P1.1).

Semantics (documented for the report):
  Observation : (1, 64, 64) uint8 grid of colour ids 0-15 (last animation
                layer of the frame, zero-padded to 64x64).
  Actions     : Discrete(70)
                  0-4  -> ACTION1..ACTION5
                  5    -> ACTION7 (undo, where supported)
                  6-69 -> ACTION6 click at the centre of an 8x8 cell grid
  Reward      : +1 per newly completed level (delta of levels_completed).
                Training-only signal; official score is reported separately.
  Terminated  : state WIN or GAME_OVER.
  Truncated   : max_steps actions in the episode.
  Invalid act.: an action not in available_actions is a no-op (no engine
                call, still costs one step); action_masks() exposes legality.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

GRID = 64
N_COLORS = 16
CLICK_RES = 8
CELL = GRID // CLICK_RES

SIMPLE = [a for a in (
    GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
    GameAction.ACTION4, GameAction.ACTION5,
    getattr(GameAction, "ACTION7", None),
) if a is not None]
N_SIMPLE = len(SIMPLE)
N_ACTIONS = N_SIMPLE + CLICK_RES * CLICK_RES

_ARCADE = None


def get_arcade():
    """One shared Arcade (game catalogue + local engine) per process."""
    global _ARCADE
    if _ARCADE is None:
        logging.getLogger("arc_agi").setLevel(logging.WARNING)
        _ARCADE = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    return _ARCADE


def decode_action(idx: int) -> tuple[GameAction, Optional[dict]]:
    if idx < N_SIMPLE:
        return SIMPLE[idx], None
    row, col = divmod(idx - N_SIMPLE, CLICK_RES)
    return GameAction.ACTION6, {"x": col * CELL + CELL // 2,
                                "y": row * CELL + CELL // 2}


def frame_to_grid(raw) -> np.ndarray:
    grid = np.zeros((GRID, GRID), dtype=np.uint8)
    if raw is None or raw.frame is None:
        return grid
    f = np.asarray(raw.frame)
    if f.size == 0:
        return grid
    if f.ndim == 3:          # stack of animation layers -> final layer
        f = f[-1]
    h, w = min(f.shape[0], GRID), min(f.shape[1], GRID)
    grid[:h, :w] = f[:h, :w]
    return grid


def available_ids(raw) -> set[int]:
    if raw is None or not raw.available_actions:
        return set()
    return {int(getattr(a, "value", a)) for a in raw.available_actions}


class ArcGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, game_id: str, max_steps: int = 500, arcade=None):
        super().__init__()
        self.game_id = game_id.split("-")[0]
        self.max_steps = max_steps
        self.arc = arcade or get_arcade()
        self.env = None
        self.observation_space = spaces.Box(0, N_COLORS - 1, (1, GRID, GRID), np.uint8)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._last = None
        self._levels = 0
        self._t = 0

    # ---- helpers -------------------------------------------------------
    def _obs(self, raw) -> np.ndarray:
        return frame_to_grid(raw)[None]

    def _info(self, raw) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "state": getattr(raw.state, "name", str(raw.state)) if raw else "NONE",
            "levels_completed": raw.levels_completed if raw else 0,
            "win_levels": raw.win_levels if raw else 0,
            "t": self._t,
        }

    def action_masks(self) -> np.ndarray:
        av = available_ids(self._last)
        if not av:
            return np.ones(N_ACTIONS, dtype=bool)
        m = np.zeros(N_ACTIONS, dtype=bool)
        for i, a in enumerate(SIMPLE):
            m[i] = int(a.value) in av
        m[N_SIMPLE:] = int(GameAction.ACTION6.value) in av
        if not m.any():
            m[:] = True
        return m

    # ---- gym API -------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.env is None:
            self.env = self.arc.make(self.game_id)
            if self.env is None:
                raise RuntimeError(f"could not create env {self.game_id}")
        raw = self.env.reset()
        self._last = raw
        self._levels = raw.levels_completed if raw else 0
        self._t = 0
        return self._obs(raw), self._info(raw)

    def step(self, action):
        self._t += 1
        ga, data = decode_action(int(action))
        invalid = bool(available_ids(self._last)) and int(ga.value) not in available_ids(self._last)
        if invalid:
            raw = self._last
        else:
            raw = self.env.step(ga, data=data) or self._last
        levels = raw.levels_completed
        reward = float(max(0, levels - self._levels))
        self._levels = levels
        self._last = raw
        terminated = raw.state in (GameState.WIN, GameState.GAME_OVER)
        truncated = (self._t >= self.max_steps) and not terminated
        info = self._info(raw)
        info["invalid_action"] = invalid
        return self._obs(raw), reward, terminated, truncated, info
