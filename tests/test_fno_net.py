import torch

from src.fno_net import FNO1d, SpectralConv1d


def test_spectral_conv_shape_and_finiteness():
    layer = SpectralConv1d(in_channels=8, out_channels=8, modes=6)
    x = torch.randn(4, 8, 32)
    out = layer(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_spectral_conv_gradient_flows():
    layer = SpectralConv1d(in_channels=4, out_channels=4, modes=4)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = layer(x).sum()
    out.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight_real.grad).all()


def test_fno_forward_pass_shape():
    model = FNO1d(Nx=128, n_save=20, modes=16, width=16, n_layers=2)
    u0 = torch.randn(4, 128)
    out = model(u0)
    assert out.shape == (4, 20, 128)


def test_fno_output_is_finite():
    model = FNO1d(Nx=128, n_save=20, modes=16, width=16, n_layers=2)
    u0 = torch.randn(4, 128)
    out = model(u0)
    assert torch.isfinite(out).all()


def test_fno_accepts_field_input_not_just_two_scalars():
    """The structural point of using an FNO over the MLP surrogate: it must accept an arbitrary
    field of the right grid size, not just the 2-parameter (A, nu) family the MLP is limited to.
    """
    import numpy as np
    model = FNO1d(Nx=64, n_save=10, modes=8, width=16, n_layers=2)
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    # A two-mode initial condition -- outside the single-mode sinusoidal family the MLP surrogate
    # was trained on and structurally cannot represent as (A, nu).
    u0_two_mode = 1.0 * np.sin(x) + 0.5 * np.sin(2 * x)
    u0 = torch.tensor(u0_two_mode, dtype=torch.float32).unsqueeze(0)
    out = model(u0)
    assert out.shape == (1, 10, 64)
    assert torch.isfinite(out).all()


def test_fno_gradient_flows_end_to_end():
    model = FNO1d(Nx=64, n_save=10, modes=8, width=16, n_layers=2)
    u0 = torch.randn(2, 64, requires_grad=True)
    out = model(u0)
    loss = out.pow(2).mean()
    loss.backward()
    assert u0.grad is not None
    assert torch.isfinite(u0.grad).all()
    for p in model.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
