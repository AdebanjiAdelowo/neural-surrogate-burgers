"""Learned surrogate: parameters (A, nu) -> full space-time solution field u(x,t).

Architecture decision (documented, not a silent deviation): ARCHITECTURE.md specified "CNN or MLP"
for the MVP. A CNN is architecturally inappropriate here -- the input is two scalars (A, nu), not
spatial data with local structure for a convolution to exploit. An MLP mapping the 2-dimensional
parameter vector to the flattened (n_save * Nx) output field is the "smallest appropriate" choice
given the actual input structure, per IMPLEMENTATION_PLAN-style reasoning ("begin there unless
evidence suggests another model is more appropriate" -- this is that evidence).
"""

import torch
import torch.nn as nn


class BurgersSurrogateMLP(nn.Module):
    """Small MLP: (A, nu) -> flattened (n_save, Nx) solution field.

    Deliberately small (a handful of hidden layers, modest width) -- the objective is to test the
    comparative research question, not to maximise model capacity.
    """

    def __init__(self, n_save: int, Nx: int, hidden: int = 128):
        super().__init__()
        self.n_save = n_save
        self.Nx = Nx
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_save * Nx),
        )

    def forward(self, params):
        # params: (B, 2) -- (A, nu), typically normalised before calling this.
        out = self.net(params)
        return out.view(-1, self.n_save, self.Nx)
