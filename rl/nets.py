"""Shared network blocks (all DQN/PPO variants use the same encoder; documented in the report).

Input: (B,1,64,64) uint8 colour ids -> one-hot (B,16,64,64) inside the net.
Encoder: conv k4/s4 (16x16) -> conv k2/s2 (8x8) -> conv 3x3 (8x8) = (B,64,8,8) local map aligned
with the 8x8 click grid (map cell r,c is built from pixels [8r,8r+8) x [8c,8c+8), the cell clicked
by action 6+8r+c; the 3x3 conv adds neighbour context), plus a global vector from the flat map.
Chosen for CPU speed (~47 ms/update at batch 32 on a Ryzen 5 5500U) after benchmarking a k4/s2
stack (57 ms) and an embedding-sum stem (79 ms).
QNet head: 6 simple-action values from the global vector; 64 click values from a 1x1-conv
over [local features ; broadcast global], so click values are spatially grounded.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_COLORS = 16
N_SIMPLE = 6
N_CLICK = 64
N_ACTIONS = N_SIMPLE + N_CLICK


def one_hot(x: torch.Tensor) -> torch.Tensor:
    return F.one_hot(x[:, 0].long(), N_COLORS).permute(0, 3, 1, 2).float()


class Encoder(nn.Module):
    def __init__(self, ch=64, gdim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_COLORS, ch, 4, 4), nn.ReLU(),
            nn.Conv2d(ch, ch, 2, 2), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(),
        )
        self.glob = nn.Sequential(nn.Flatten(), nn.Linear(ch * 64, gdim), nn.ReLU())
        self.ch, self.gdim = ch, gdim

    def forward(self, x):
        local = self.conv(one_hot(x))
        return local, self.glob(local)


class QNet(nn.Module):
    def __init__(self, ch=64, gdim=256, hid=128):
        super().__init__()
        self.enc = Encoder(ch, gdim)
        self.simple = nn.Sequential(nn.Linear(gdim, hid), nn.ReLU(), nn.Linear(hid, N_SIMPLE))
        self.click = nn.Sequential(nn.Conv2d(ch + gdim, hid, 1), nn.ReLU(), nn.Conv2d(hid, 1, 1))

    def forward(self, x):
        local, g = self.enc(x)
        gb = g[:, :, None, None].expand(-1, -1, 8, 8)
        c = self.click(torch.cat([local, gb], 1)).flatten(1)      # (B,64) row-major = 8r+c
        return torch.cat([self.simple(g), c], 1)
