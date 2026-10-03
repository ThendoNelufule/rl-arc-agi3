"""Replay buffer with 4-bit packed frames (colour ids < 16): 2 KB per frame, so 100k
transitions (obs + next_obs) take ~410 MB. Stores next-state legality masks for masked targets."""
import numpy as np

FRAME = 64 * 64


def pack(o):
    f = o.reshape(-1, FRAME)
    return (f[:, 0::2] << 4) | f[:, 1::2]


def unpack(p):
    out = np.empty((p.shape[0], FRAME), np.uint8)
    out[:, 0::2] = p >> 4
    out[:, 1::2] = p & 15
    return out.reshape(-1, 1, 64, 64)


class Replay:
    def __init__(self, cap, n_actions=70, extra=()):
        self.cap, self.i, self.full = cap, 0, False
        self.o = np.zeros((cap, FRAME // 2), np.uint8)
        self.no = np.zeros((cap, FRAME // 2), np.uint8)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.nm = np.zeros((cap, n_actions), bool)
        self.chg = np.zeros(cap, np.float32)     # 1 if the action changed the frame (P2.3 label)

    def __len__(self):
        return self.cap if self.full else self.i

    def add(self, o, a, r, no, d, nm, chg=0.0):
        i = self.i
        self.chg[i] = chg
        self.o[i], self.no[i] = pack(o[None])[0], pack(no[None])[0]
        self.a[i], self.r[i], self.d[i], self.nm[i] = a, r, d, nm
        self.i = (i + 1) % self.cap
        self.full |= self.i == 0

    def sample(self, n, rng):
        idx = rng.integers(0, len(self), n)
        return (unpack(self.o[idx]), self.a[idx], self.r[idx],
                unpack(self.no[idx]), self.d[idx], self.nm[idx], self.chg[idx])
