import sys, time
sys.path.insert(0, "rl")
import numpy as np
from arc_gym import ArcGymEnv, N_ACTIONS

rng = np.random.default_rng(0)
print("N_ACTIONS =", N_ACTIONS)
for gid in ["ls20", "vc33", "ft09"]:
    env = ArcGymEnv(gid, max_steps=300)
    obs, info = env.reset(seed=0)
    print(f"\n== {gid} == obs {obs.shape} {obs.dtype} colours={np.unique(obs).tolist()}")
    print("  start info:", info)
    print("  legal action ids:", np.flatnonzero(env.action_masks()).tolist()[:10], "...")
    tot = inv = 0; eps = 1; t0 = time.time()
    for t in range(300):
        a = rng.choice(np.flatnonzero(env.action_masks()))
        obs, r, term, trunc, info = env.step(a)
        tot += r; inv += info["invalid_action"]
        if (term or trunc) and t < 299:
            obs, info = env.reset(); eps += 1
    print(f"  300 steps in {time.time()-t0:.1f}s | reward={tot} | invalid={inv} | episodes={eps}")
    print("  end info:", info)
