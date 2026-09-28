"""A compact 1D Fourier Neural Operator (Li et al. 2021, arXiv:2010.08895), adapted to this
project's task framing.

Research-extension module (see RESEARCH_EXTENSION.md). Not part of the original verified MVP.

Design choice, disclosed (not a silent deviation)
---------------------------------------------------
The "classical" FNO usage in Li et al. (2021) is an *autoregressive one-step* operator: given
u(x, t), predict u(x, t+dt), then roll forward. This module instead predicts the *entire*
space-time trajectory in a single forward pass from the initial condition field alone --
matching the exact task framing `src.surrogate_net.BurgersSurrogateMLP` already uses (so the
FNO-vs-MLP comparison isolates the effect of input/architecture type, not task framing), and
avoiding the extra design surface (rollout stability, teacher forcing, error accumulation) that a
faithful autoregressive reimplementation would add within this project's time budget.

This is a real, disclosed limitation, not a hidden one: this module does NOT test the classical
FNO capability of long-horizon autoregressive rollout, nor genuine resolution invariance across
*different* grid sizes at inference time (the spectral-convolution machinery is resolution-
invariant in principle, but the fixed-size output layer below reintroduces a resolution
dependence for this single-shot framing). See RESEARCH_EXTENSION.md, "Limitations."

What IS a genuine, meaningful difference from `BurgersSurrogateMLP` (not just relabelling): this
network's *input* is the actual discretised initial-condition field u0(x) in R^Nx, not two scalar
parameters (A, nu) in R^2. This is a structural capability difference: the MLP surrogate cannot
even be *evaluated* on an initial condition outside its 2-parameter sinusoidal family (there is no
way to encode e.g. a two-mode initial condition as (A, nu)), whereas this network can accept any
field of the correct grid size. That capability gap -- not merely an accuracy difference -- is the
main structural finding this module makes possible to test (see the out-of-family generalisation
experiment in scripts/evaluate_research_extension.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NU_LOW, NU_HIGH = 0.01, 0.1  # training viscosity range (scripts/generate_dataset.NU_RANGE)


class SpectralConv1d(nn.Module):
    """1D spectral convolution: rfft -> truncate to `modes` low frequencies -> learned complex
    linear map per mode -> zero-pad back -> irfft. This is the core FNO building block; it is a
    genuinely different computational primitive from the MLP's dense layers, respecting the
    periodic domain by construction (exactly matching this problem's periodic boundary condition).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        # Real and imaginary parts stored separately (real-valued nn.Parameter) since PyTorch's
        # autograd/optimizers historically have had partial/uneven complex-tensor support; this is
        # the standard workaround used in the reference FNO implementation.
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_channels, Nx)
        batch, _, Nx = x.shape
        x_hat = torch.fft.rfft(x, dim=-1)  # (batch, in_channels, Nx//2 + 1), complex
        modes = min(self.modes, x_hat.shape[-1])

        weight = torch.complex(self.weight_real[..., :modes], self.weight_imag[..., :modes])
        # einsum: (batch, in, modes) x (in, out, modes) -> (batch, out, modes)
        out_low = torch.einsum("bim,iom->bom", x_hat[..., :modes], weight)
        n_pad = x_hat.shape[-1] - modes
        out_hat = F.pad(out_low, (0, n_pad))  # zero-pad the untouched high frequencies
        return torch.fft.irfft(out_hat, n=Nx, dim=-1)


class FNOBlock(nn.Module):
    """One FNO layer: spectral conv (global, low-frequency) + pointwise conv (local, all
    frequencies, standard FNO residual design so the network isn't restricted to only the
    `modes` lowest frequencies) + GELU."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = nn.Conv1d(width, width, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.pointwise(x))


class FNO1d(nn.Module):
    """Field-to-trajectory 1D FNO: u0(x) in R^Nx -> flattened (n_save, Nx) trajectory.

    Deliberately small (matching the MLP surrogate's "smallest appropriate" philosophy) -- the
    objective is a fair comparison of architecture/input-representation type, not a capacity race.
    """

    def __init__(self, Nx: int, n_save: int, modes: int = 16, width: int = 32, n_layers: int = 4,
                 use_nu: bool = False):
        super().__init__()
        self.Nx = Nx
        self.n_save = n_save
        # use_nu=False (default) is the historical, viscosity-BLIND network: input [u0(x), x]. With
        # use_nu=True the only change is one extra input channel carrying the normalised viscosity
        # broadcast over the grid ([u0(x), x, nu_norm]); depth, modes, width and head are identical.
        self.use_nu = use_nu
        # Lift: [u0(x), x] -> width channels. Including the coordinate x as a second input channel
        # is standard FNO practice (Li et al. 2021 Appendix): the network is otherwise translation-
        # invariant by construction (spectral convs), so x must be given explicitly if the periodic
        # domain's absolute position matters (it does here, since the IC's phase varies).
        self.lift = nn.Conv1d(3 if use_nu else 2, width, kernel_size=1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes) for _ in range(n_layers)])
        self.project = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=1), nn.GELU(),
            nn.Conv1d(width, n_save, kernel_size=1),
        )
        x_grid = torch.linspace(0, 1, Nx + 1)[:-1]
        self.register_buffer("x_grid", x_grid)

    def build_input(self, u0: torch.Tensor, nu: torch.Tensor = None) -> torch.Tensor:
        """Network input (batch, 2 or 3, Nx): [u0(x), x] and, if use_nu, [.., nu_norm(broadcast)].
        nu is the RAW viscosity (batch,); it is normalised with the same convention as the MLP
        surrogate (scripts/train_surrogate.normalize_params): (nu - 0.01) / (0.1 - 0.01)."""
        batch = u0.shape[0]
        x_grid = self.x_grid.unsqueeze(0).expand(batch, -1)  # (batch, Nx)
        channels = [u0, x_grid]
        if self.use_nu:
            if nu is None:
                raise ValueError("this FNO was built with use_nu=True: pass the viscosity nu")
            nu_norm = (nu.to(u0.dtype).reshape(batch, 1) - NU_LOW) / (NU_HIGH - NU_LOW)
            channels.append(nu_norm.expand(-1, u0.shape[1]))
        return torch.stack(channels, dim=1)

    def forward(self, u0: torch.Tensor, nu: torch.Tensor = None) -> torch.Tensor:
        # u0: (batch, Nx) -- the actual initial-condition field, not scalar parameters. nu is only
        # read when the network was built with use_nu=True.
        h = self.lift(self.build_input(u0, nu))
        for block in self.blocks:
            h = block(h)
        out = self.project(h)  # (batch, n_save, Nx)
        return out
