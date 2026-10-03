# Running the neural surrogates on an NVIDIA GPU

This project is intentionally hybrid. The numerical methods are NumPy and run on the CPU; only the two
PyTorch networks can use an accelerator. CUDA support adds a third neural backend next to CPU and
Apple MPS. It does not move the numerical baseline to the GPU, and it does not replace any existing
result.

## What runs where

**A. CPU numerical baseline (NumPy, always CPU)**

| Component | Code |
|---|---|
| Pseudo-spectral Burgers solver (ground truth, dataset generation) | `src/solver.py`, `src/general_ic_solver.py` |
| POD-Galerkin ROM | `src/pod_rom.py` |
| DEIM hyper-reduced ROM | `src/deim.py` |

These run in float64 and are deliberately not ported to PyTorch, CuPy or JAX. They are the reference
every surrogate is measured against, so changing their backend would change the baseline itself. At
the problem sizes studied here (Nx = 128, reduced dimension 8, 16 DEIM points) the work per time step
is too small for a GPU to help: kernel launch and transfer overhead would dominate.

**B. Neural accelerators (PyTorch, float32)**

| Backend | Selected by | Status |
|---|---|---|
| CPU | `--device cpu` | supported |
| Apple MPS | `--device mps`, or `--device auto` on a Mac | backend of the published results |
| NVIDIA CUDA | `--device cuda` only | additive backend, results kept separate |

The MLP (`src/surrogate_net.py`) and FNO (`src/fno_net.py`) are device-agnostic: the FNO's grid is a
registered buffer, its complex spectral weights are assembled from real parameters on the active
device, and `torch.fft.rfft`/`irfft` and `einsum` run natively on CUDA.

## Device policy

All neural scripts take `--device {auto,cpu,mps,cuda}`, implemented once in `src/device.py`.

* `auto` (the default) keeps the historical rule: MPS if available, otherwise CPU. It never picks
  CUDA, so every existing command still uses the backend its published numbers came from. On a CUDA
  machine it prints a hint and runs on the CPU.
* An explicitly requested backend that is not available raises an error instead of falling back to
  the CPU.
* On CUDA, TF32 is disabled for matmul and cuDNN convolutions, so the networks run in true float32,
  as they do on CPU and MPS. No mixed precision (AMP) is used: the FNO's spectral layers work on
  complex FFT coefficients, and lowering their precision would be a separate experiment.

Timing scripts that compare backends (`fno_nu_ablation.py`, `hyperreduction_benchmark.py fno`) time
`cpu`, then `mps` and `cuda` when present, each under its own key. `fno_nu_ablation.py` refuses to
replace an existing output, including the committed `run1` files its default tag points at; pass a
new `--tag`, or `--overwrite` deliberately.

CUDA compatibility depends on the installed PyTorch build and the GPU architecture, so it has to be
verified in the actual environment; the notebook's smoke tests do that before any full run.

## Results and provenance

* The committed results in `report/` (Apple MPS for the networks, NumPy/CPU for everything else)
  are historical and stay as they are. Evaluation scripts refuse to write a CUDA run into
  `report/research/` or `report/`; pass `--out-dir`/`--out` instead. Backend-specific runs go under
  `runs/` (git-ignored).
* Every training run writes a metadata sidecar (`surrogate_run_meta.json`, `fno_run_meta.json`) with
  the measured training time and a `device_info` block: device, GPU model, CUDA and cuDNN versions,
  TF32 flags.
* The 8.23 s MLP training time in `RESEARCH_EXTENSION.md` was measured once on Apple MPS. It is
  reported only for the historical checkpoint and never for a CUDA run; a new run reports its own
  measured time, or nothing. `offline_cost_provenance` in `results.json` states the source of every
  offline time.
* GPU timings bracket the timed region with `torch.cuda.synchronize()`. MPS keeps its existing
  `torch.mps.synchronize()` handling.
* CUDA timings are additional hardware results. They do not replace the CPU/MPS timing claims in the
  README; any comparison should quote both, each with its hardware label, and pair speed and accuracy
  figures from the same configuration.

## Reproducibility

The CUDA path changes only the backend. Datasets, seeds, solver settings, both architectures, 200
epochs, batch size 16, Adam with learning rate 1e-3, the evaluation metrics, float64 references and
float32 networks are unchanged. Weights are initialised on the CPU with the fixed seed and then moved,
and mini-batch order is drawn on the CPU generator, so the initial weights and batch order are the
same on every backend.

A CUDA retrain is still a new backend experiment, not a bit-for-bit reproduction of the MPS
checkpoints: floating-point kernels differ between backends, and some cuDNN kernels are
nondeterministic run to run. Compare CUDA metrics with a CPU reference trained on the same data in
the same session, and treat differences at the level of seed-to-seed variation as noise.

Existing checkpoints load on any backend through `map_location`, and checkpoints saved on CUDA load
on CPU or MPS the same way.

### Data provenance

Datasets are not committed. Regenerating them with the current solver gives the same (A, nu) samples,
but the trajectories `u` differ from the original datasets by up to about 0.45% of max |u|, because
of a later snapshot-count fix in `src/solver.py` (`report/audit/reproduce_run1.json` records
`test_u: False`). Two data modes follow from this:

* **`upload` (default, like-for-like).** The original files behind the published MPS results, checked
  against their SHA-256 hashes:

  | File | SHA-256 |
  |---|---|
  | `data/train.npz` | `233f9eeae594d82f0e97bb53a90d039dbcc4f10c85012fe5a34a086fd88450ee` |
  | `data/val.npz` | `0d083b211f660215976a52661df28702cf04d991ead8502ca394265f2068346f` |
  | `data/test.npz` | `a3542f1169912c0749571476640ee7e543dc0d35d61a0c3bb48f829854520e59` |
  | `data/ood_family_test.npz` | `f69a55bd219ae908b89676c45062f967a5975f19f754906f0fe0a0025d206116` |
  | `experiments/deim_basis.npz` | `97f28e8dc6e24e8ec0a34a1f25ecc19e660f3eee36f0333aa1cea2101592cb23` |

  Package them on the original machine, from the repository root:

  ```bash
  zip historical_inputs.zip data/train.npz data/val.npz data/test.npz data/ood_family_test.npz experiments/deim_basis.npz
  ```

* **`regenerate` (convenience only).** Rebuilds the data with the current solver. Results from this
  mode are not comparable with the published MPS numbers.

Each run writes `data_provenance.json` (mode, per-file hashes, whether each file is the historical
one) and `provenance.json` (commit, device, data mode).

A useful comparison has three columns: the committed MPS result on the historical data, the CUDA
result on the same data, and a CPU result trained in the same session on the same data. None of them
is a reference "best" backend; differences of the size of the seed-to-seed spread are noise.

## Google Colab (supported target)

Open `colab/run_cuda.ipynb` with a GPU runtime, set `REF` to the commit to test, and upload
`historical_inputs.zip` to `/content/` (or when prompted); the archive is always kept outside the
repository checkout. Every shell command goes through a helper that raises on failure, and every cell
after the gate re-checks it. A session gets one `RUN_ID` (section 3), which re-running section 0 to
set `RUN_FULL = True` does not reset; the run directory, `provenance.json` and the archive name all
use it. The notebook:

1. prints `nvidia-smi`, Python, PyTorch, the CUDA version and the GPU name;
2. clones the repository, checks out `REF`, and prints the commit and its dirty state;
3. installs dependencies without touching the preinstalled CUDA build of PyTorch, confirms CUDA is still available, and runs the test suite;
4. loads the data (verifying the historical hashes in `upload` mode) and builds the POD/DEIM basis with NumPy on the CPU; in `upload` mode the historical DEIM basis is kept;
5. runs a 5-epoch smoke test for each model and checks, separately for the MLP and the FNO: requested and resolved device are CUDA, the checkpoint and metadata exist, the checkpoint tensors were saved from CUDA, losses and test errors are finite, and the device information names the GPU with TF32 off;
6. stops at a gate unless both smoke tests and the evaluation smoke test passed and `RUN_FULL = True`;
7. trains MLP and FNO seeds 0 to 2 on CUDA, plus seed 0 of both on this machine's CPU on the same data;
8. runs the existing evaluation, the FNO multi-seed evaluation, an MLP per-seed summary, the MVP comparison, and timing on CUDA and on this machine's CPU;
9. checks that the source tree is unchanged since checkout (ignored generated files such as `data/`, `runs/` and `experiments/*.npz` are listed, not counted as changes) and that a CUDA checkpoint loads on CPU;
10. prints the three-column comparison and the provenance, and zips `runs/<RUN_ID>/`.

The archive holds `smoke/`, `cuda/` (checkpoints, run metadata and training logs for seeds 0 to 2,
and `results/` with `results.json`, `results_summary.txt`, `comparison_figure.png`,
`fno_multiseed_results.json`, `mlp_multiseed_results.json`, `mvp/`, and both timing files),
`cpu_reference/`, `data_provenance.json`, `provenance.json`, `deim_rank_sweep_this_machine.json`,
`build_deim.log`, `pytest.log`, `git_status.txt`, `nvidia_smi.txt` and `pip_freeze.txt`.

## Completed runs

* [`report/cuda/tesla_t4_20261003/`](report/cuda/tesla_t4_20261003/README.md): Tesla T4 on Colab,
  historical data. MLP and FNO accuracy matches the committed MPS results to within 1.2e-6 relative;
  timings are T4 versus the same runtime's CPU only.

## Kaggle (additional environment, not verified)

The same commands should work in a Kaggle GPU notebook, but this environment has not been tested;
whether its PyTorch build supports the assigned GPU has to be checked there. Upload
`historical_inputs.zip` as a dataset and unzip it into the repository first.

```bash
!git clone https://github.com/AdebanjiAdelowo/neural-surrogate-burgers.git
%cd neural-surrogate-burgers
!git checkout <REF>
!pip install -q scipy matplotlib pytest
!python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
!python -m pytest -q

# historical data (like-for-like); check the hashes listed above with sha256sum
!unzip -o -q /kaggle/input/<dataset>/historical_inputs.zip -d . && sha256sum data/*.npz experiments/deim_basis.npz

# POD/DEIM offline timing on this machine (NumPy, CPU), keeping the historical basis
!mkdir -p runs/kaggle && cp experiments/deim_basis.npz /tmp/deim_basis.npz && python scripts/build_deim.py
!mv report/research/deim_rank_sweep.json runs/kaggle/deim_rank_sweep.json && git checkout -- report/research/deim_rank_sweep.json
!cp /tmp/deim_basis.npz experiments/deim_basis.npz

# networks: CUDA
!for s in 0 1 2; do python scripts/train_surrogate.py --device cuda --seed $s --out-dir runs/kaggle/cuda && python scripts/train_fno.py --device cuda --seed $s --out-dir runs/kaggle/cuda; done
!python scripts/evaluate_research_extension.py --device cuda --checkpoint-dir runs/kaggle/cuda --out-dir runs/kaggle/cuda/results --deim-sweep runs/kaggle/deim_rank_sweep.json
!python scripts/evaluate_fno_multiseed.py --device cuda --checkpoint-dir runs/kaggle/cuda --out runs/kaggle/cuda/results/fno_multiseed_results.json
!python scripts/verify_timing.py --device cuda --checkpoint-dir runs/kaggle/cuda --out runs/kaggle/cuda/results/timing_verification_cuda.json
!test -z "$(git status --porcelain)" && echo "source tree unchanged"
!cd runs && zip -qr /kaggle/working/kaggle_cuda.zip kaggle
```

The same commands work on any Linux machine with an NVIDIA GPU and a CUDA build of PyTorch.
