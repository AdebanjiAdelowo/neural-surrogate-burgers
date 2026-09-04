import torch

from src.surrogate_net import BurgersSurrogateMLP


def test_forward_pass_shape():
    model = BurgersSurrogateMLP(n_save=20, Nx=128)
    params = torch.randn(4, 2)
    out = model(params)
    assert out.shape == (4, 20, 128)


def test_output_is_finite():
    model = BurgersSurrogateMLP(n_save=20, Nx=128)
    params = torch.randn(4, 2)
    out = model(params)
    assert torch.isfinite(out).all()
