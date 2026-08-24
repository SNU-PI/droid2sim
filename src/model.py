"""Small action-conditioned video world model.

Deliberately a *chunk* predictor rather than an autoregressive one: given k
context frames it emits the next 8 frames in a single forward pass. That
removes autoregressive drift from the picture entirely, so if the model still
fails to reflect a physics change we know it is not a drift artifact -- the
setting is as generous to the world model as we can make it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PRED = 8


class FiLM(nn.Module):
    def __init__(self, cond_dim, ch):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, 128), nn.SiLU(),
                                 nn.Linear(128, ch * 2))

    def forward(self, x, c):
        s, b = self.net(c).chunk(2, dim=-1)
        return x * (1 + s[:, :, None, None]) + b[:, :, None, None]


def blk(i, o, down=True):
    conv = (nn.Conv2d(i, o, 4, 2, 1) if down
            else nn.ConvTranspose2d(i, o, 4, 2, 1))
    return nn.Sequential(conv, nn.GroupNorm(8, o), nn.SiLU())


class WorldModel(nn.Module):
    """k context frames + action (+ optionally the true physics params) -> next 8 frames."""

    def __init__(self, k, cond_params=False, width=64):
        super().__init__()
        self.k, self.cond_params = k, cond_params
        w = width
        cond_dim = 2 + (2 if cond_params else 0)   # action (+ physics params)

        self.e1 = blk(3 * k, w)          # 96 -> 48
        self.e2 = blk(w, w * 2)          # 48 -> 24
        self.e3 = blk(w * 2, w * 4)      # 24 -> 12
        self.e4 = blk(w * 4, w * 4)      # 12 -> 6
        self.film = FiLM(cond_dim, w * 4)
        self.mid = nn.Sequential(nn.Conv2d(w * 4, w * 4, 3, 1, 1),
                                 nn.GroupNorm(8, w * 4), nn.SiLU())

        self.d1 = blk(w * 4, w * 4, down=False)          # 6 -> 12
        self.d2 = blk(w * 8, w * 2, down=False)          # 12 -> 24
        self.d3 = blk(w * 4, w, down=False)              # 24 -> 48
        self.d4 = blk(w * 2, w, down=False)              # 48 -> 96
        self.out = nn.Conv2d(w, 3 * N_PRED, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ctx, action, params=None):
        """ctx: [B, k, 3, H, W] in [-1,1]. Returns [B, N_PRED, 3, H, W]."""
        B = ctx.shape[0]
        last = ctx[:, -1]                                  # [B,3,H,W]
        x = ctx.reshape(B, self.k * 3, *ctx.shape[-2:])

        c = action if not self.cond_params else torch.cat([action, params], -1)
        h1 = self.e1(x); h2 = self.e2(h1); h3 = self.e3(h2); h4 = self.e4(h3)
        h = self.mid(self.film(h4, c))

        y = self.d1(h)
        y = self.d2(torch.cat([y, h3], 1))
        y = self.d3(torch.cat([y, h2], 1))
        y = self.d4(torch.cat([y, h1], 1))
        delta = self.out(y).reshape(B, N_PRED, 3, *ctx.shape[-2:])
        # residual against the last observed frame: the background is static,
        # so the model only has to explain what moves
        return (last[:, None] + delta).clamp(-1, 1)
