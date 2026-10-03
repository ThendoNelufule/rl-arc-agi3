"""Share of actions that repeat a (state, action) pair already seen to leave the frame unchanged
(evidence metric introduced in P2.2). Usage: .venv/bin/python rl/noop_repeats.py TAG [TAG ...]"""
import glob, sys
import numpy as np


def repeat_noop(tag):
    rr, nn = [], []
    for f in glob.glob(f"experiments/runs/{tag}/recordings/*.npz"):
        z = np.load(f)
        fr, ac = z["frames"], z["actions"]
        seen, rep, noop = set(), 0, 0
        for i, a in enumerate(ac):
            if np.array_equal(fr[i], fr[i + 1]):
                noop += 1
                key = (fr[i].tobytes(), int(a))
                rep += key in seen
                seen.add(key)
        rr.append(rep / len(ac)); nn.append(noop / len(ac))
    return float(np.mean(nn)), float(np.mean(rr))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        n, r = repeat_noop(t)
        print(f"{t:40s} noop {n:.3f}  repeated-known-noop {r:.3f}  ({r / max(n, 1e-9):.0%} of no-ops)")
