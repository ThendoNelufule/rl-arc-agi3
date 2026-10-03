"""P1.4 offline check: find the assets dataset, pip install from it with no index,
build tiny MaskablePPO / RecurrentPPO / DQN nets, round-trip a state_dict."""
import glob, os, subprocess, sys, json, time
t0 = time.time()
hits = glob.glob("/kaggle/input/**/arc3-rl-assets*", recursive=True) + \
       [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/requirements-kaggle.txt", recursive=True)]
print("MOUNT candidates", sorted(set(hits)))
root = sorted(set(os.path.dirname(p) for p in glob.glob("/kaggle/input/**/requirements-kaggle.txt", recursive=True)))[0]
print("ASSETS", root, os.listdir(root))
comp = glob.glob("/kaggle/input/**/arc_agi_3_wheels", recursive=True)
print("COMP_WHEELS", comp)
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", f"{root}/wheels", *(["--find-links", comp[0]] if comp else []),
                    "stable-baselines3", "sb3-contrib", *(["arc-agi"] if comp else [])],
                   capture_output=True, text=True)
print("PIP rc", r.returncode); print(r.stdout[-1500:]); print(r.stderr[-1500:])
import numpy as np, torch, gymnasium as gym
from gymnasium import spaces
import stable_baselines3, sb3_contrib
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3 import DQN
print("VERSIONS sb3", stable_baselines3.__version__, "contrib", sb3_contrib.__version__,
      "torch", torch.__version__, "gym", gym.__version__, "np", np.__version__)

class Toy(gym.Env):
    observation_space = spaces.Box(0, 15, (1, 64, 64), np.uint8)
    action_space = spaces.Discrete(70)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.t = 0
        return np.zeros((1, 64, 64), np.uint8), {}
    def step(self, a):
        self.t += 1
        return np.zeros((1, 64, 64), np.uint8), 0.0, False, self.t >= 16, {}
    def action_masks(self):
        m = np.zeros(70, bool); m[:5] = True; return m

kw = dict(policy_kwargs=dict(normalize_images=False), verbose=0, device="cpu")
for name, cls, extra in [("maskppo", MaskablePPO, dict(n_steps=32, batch_size=32)),
                         ("rppo", RecurrentPPO, dict(n_steps=32, batch_size=32)),
                         ("dqn", DQN, dict(learning_starts=8, buffer_size=200))]:
    pol = "CnnLstmPolicy" if cls is RecurrentPPO else "CnnPolicy"
    m = cls(pol, Toy(), **extra, **kw)
    m.learn(64)
    sd = m.policy.state_dict()
    torch.save(sd, f"/tmp/{name}.pt")
    sd2 = torch.load(f"/tmp/{name}.pt", weights_only=True)
    m.policy.load_state_dict(sd2)
    print("OK", name, sum(v.numel() for v in sd2.values()), "params")
if comp:
    import arc_agi, arcengine; print("OK arc_agi import")
print("ALL_OK", round(time.time() - t0, 1), "s")
