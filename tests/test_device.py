import json

import pytest
import torch

from src.device import device_info, select_device, synchronize, timing_devices
from src.fno_net import FNO1d
from src.surrogate_net import BurgersSurrogateMLP

HAS_MPS = torch.backends.mps.is_available()
HAS_CUDA = torch.cuda.is_available()


def test_auto_keeps_the_historical_mps_then_cpu_rule():
    assert select_device("auto").type == ("mps" if HAS_MPS else "cpu")


def test_explicit_cpu_and_unknown_name():
    assert select_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        select_device("tpu")


@pytest.mark.skipif(HAS_CUDA, reason="checks the failure path on machines without CUDA")
def test_explicit_cuda_fails_clearly_without_cuda():
    with pytest.raises(RuntimeError, match="cuda"):
        select_device("cuda")


@pytest.mark.skipif(HAS_MPS, reason="checks the failure path on machines without MPS")
def test_explicit_mps_fails_clearly_without_mps():
    with pytest.raises(RuntimeError, match="mps"):
        select_device("mps")


def test_timing_devices_label_each_backend_separately():
    devs = timing_devices()
    assert devs[0] == "cpu" and len(devs) == len(set(devs))
    assert ("mps" in devs) == HAS_MPS and ("cuda" in devs) == HAS_CUDA


def test_device_info_and_synchronize_on_selected_device():
    dev = select_device("auto")
    synchronize(dev)
    info = device_info(dev)
    assert info["device"] == str(dev) and info["torch_version"] == torch.__version__
    json.dumps(info)  # must be serialisable into run metadata


@pytest.mark.parametrize("dev", [d for d in timing_devices() if d != "cpu"])
def test_models_on_accelerator_match_cpu_and_checkpoint_round_trips(dev, tmp_path):
    torch.manual_seed(0)
    fno, mlp = FNO1d(Nx=64, n_save=5, modes=8, width=8, n_layers=2), BurgersSurrogateMLP(5, 64, hidden=16)
    u0, p = torch.randn(3, 64), torch.rand(3, 2)
    with torch.no_grad():
        ref_fno, ref_mlp = fno(u0), mlp(p)
        out_fno = fno.to(dev)(u0.to(dev)).cpu()
        out_mlp = mlp.to(dev)(p.to(dev)).cpu()
    torch.testing.assert_close(out_fno, ref_fno, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(out_mlp, ref_mlp, rtol=1e-4, atol=1e-5)
    # a checkpoint saved from the accelerator loads on CPU through map_location
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": fno.state_dict()}, path)
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state"]
    assert all(t.device.type == "cpu" for t in state.values())
    fresh = FNO1d(Nx=64, n_save=5, modes=8, width=8, n_layers=2)
    fresh.load_state_dict(state)
    with torch.no_grad():
        torch.testing.assert_close(fresh(u0), ref_fno, rtol=1e-4, atol=1e-5)


def test_cuda_run_never_reports_the_historical_mps_mlp_training_time(tmp_path):
    from scripts.evaluate_research_extension import HISTORICAL_MLP_TRAIN_TIME_S, mlp_training_time
    cuda = torch.device("cuda")  # constructing the device object does not need a GPU
    assert mlp_training_time("experiments", cuda)[0] is None
    assert mlp_training_time(str(tmp_path), torch.device("cpu"))[0] is None
    (tmp_path / "surrogate_run_meta.json").write_text(
        json.dumps({"total_train_time_s": 1.5, "epochs": 200, "device": "cuda"}))
    seconds, source = mlp_training_time(str(tmp_path), cuda)
    assert seconds == 1.5 and "measured" in source and seconds != HISTORICAL_MLP_TRAIN_TIME_S


@pytest.mark.parametrize("module, kwargs", [
    ("scripts.evaluate_research_extension", {}),
    ("scripts.verify_timing", {}),
    ("scripts.evaluate_fno_multiseed", {}),
    ("scripts.evaluate_comparison", {}),
])
def test_cuda_runs_refuse_to_overwrite_historical_results(module, kwargs, monkeypatch):
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(mod, "select_device", lambda name: torch.device("cuda"))
    with pytest.raises(SystemExit, match="historical"):
        mod.main("cuda", **kwargs)


@pytest.mark.parametrize("aligned", [False, True])
def test_nu_ablation_default_tag_does_not_overwrite_the_historical_result(aligned, monkeypatch):
    import hashlib
    import os

    import scripts.fno_nu_ablation as abl
    path = abl.output_path("run1", aligned)
    assert os.path.exists(path)  # committed historical result
    before = hashlib.sha256(open(path, "rb").read()).hexdigest()
    monkeypatch.setattr(abl, "environment_info", lambda: pytest.fail("ran past the overwrite guard"))
    with pytest.raises(SystemExit, match="already exists"):
        abl.stage("run1", aligned)
    assert hashlib.sha256(open(path, "rb").read()).hexdigest() == before
    assert abl.output_path("cuda_run", aligned) != path
