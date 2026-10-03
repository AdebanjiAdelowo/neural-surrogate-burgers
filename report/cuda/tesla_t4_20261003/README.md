# Neural surrogates on an NVIDIA Tesla T4 (Google Colab, 2026-10-03)

The existing MLP and FNO configurations were trained and evaluated on CUDA with
`colab/run_cuda.ipynb`, using the same hash-verified datasets as the committed Apple MPS results in
`report/research/`. This directory preserves the run's small artifacts. It adds a backend result; it
does not replace any committed CPU or MPS result.

## Provenance

| Item | Value | Evidence |
|---|---|---|
| Source commit | `abaabba6e2b048f888f4a71595c7c4beb34a2503` (`REF` set to the same SHA) | `provenance.json` |
| Source tree | clean at checkout and unchanged before packaging | `provenance.json` (`dirty_at_start: false`), `git_status.txt` |
| GPU | Tesla T4, 14.6 GB, compute capability 7.5, driver 580.82.07 | `provenance.json`, `nvidia_smi.txt` |
| Software | Python 3.13.15, PyTorch 2.11.0+cu130 (CUDA 13.0, cuDNN 92700), NumPy 2.1.3, SciPy 1.16.3 | `provenance.json`, `pip_freeze.txt` |
| Precision | float32, TF32 disabled for matmul and cuDNN, no AMP | `device_info` in every run file |
| Device | requested `cuda` and resolved `cuda` for all six CUDA trainings; checkpoints saved from `cuda:0` | `cuda/*_run_meta*.json`; checkpoints inspected before archiving |
| Training config | 200 epochs, batch 16, Adam 1e-3, MLP and FNO seeds 0, 1, 2 | `cuda/*_run_meta*.json` |
| Test suite on the runtime | 111 passed, 1 skipped (the CUDA-unavailable failure test, which skips by design when CUDA is present) | `pytest.log` |
| Checkpoint portability | CUDA FNO checkpoint loaded on CPU: max abs prediction difference 1.07e-6 | `provenance.json` |

All of the above is read from files in this directory. Two further facts come from the executed
notebook (`run_cuda_executed.ipynb`, the copy Colab saved to GitHub as commit `028f4b4`) rather than
from a result file: both 5-epoch CUDA smoke tests and every full training passed the notebook's nine
programmatic checks, and the gate was passed with `smoke tests: {'mlp': True, 'fno': True,
'evaluation': True}`.

**Null `run_id`.** `provenance.json` records `"run_id": null`. The notebook at `abaabba` reset
`RUN_ID` when its configuration cell was re-run to set `RUN_FULL = True`; this was fixed in commit
`a8ef662`. The value is preserved as recorded. The run directory was `runs/cuda_20261003T091216Z`
(named in the notebook output and inside the downloaded archive, `None.zip`, SHA-256
`5aee4d9fc4b58e240769f263cc465c3c8d962e3c99a6f5f7270fb4ef2041e471`). The other provenance fields
were recorded independently of `RUN_ID` and are unaffected. The archive also held two directories
from earlier attempts in the same session, containing only `pytest.log`; they are not preserved here.

## Data

`DATA_SOURCE = "upload"`: the original datasets and DEIM basis behind the committed MPS results,
each checked against its SHA-256 before use (`data_provenance.json`):

| File | SHA-256 |
|---|---|
| `data/train.npz` | `233f9eeae594d82f0e97bb53a90d039dbcc4f10c85012fe5a34a086fd88450ee` |
| `data/val.npz` | `0d083b211f660215976a52661df28702cf04d991ead8502ca394265f2068346f` |
| `data/test.npz` | `a3542f1169912c0749571476640ee7e543dc0d35d61a0c3bb48f829854520e59` |
| `data/ood_family_test.npz` | `f69a55bd219ae908b89676c45062f967a5975f19f754906f0fe0a0025d206116` |
| `experiments/deim_basis.npz` | `97f28e8dc6e24e8ec0a34a1f25ecc19e660f3eee36f0333aa1cea2101592cb23` |

The DEIM basis rebuilt on the runtime from the uploaded data had identical interpolation points and
agreed with the historical basis to 9.4e-15 (up to column sign); the historical basis was used.

## Cross-backend accuracy (same data)

Relative L2 error, mean over the 40 test cases (in-distribution) and the out-of-family two-mode set,
seed 0. "Committed MPS" is `report/research/results.json`; "Colab CPU" is the same notebook session
retrained on the runtime's CPU on the same data (`cpu_reference/`).

| Metric | Committed MPS | Tesla T4 CUDA | Colab CPU |
|---|---|---|---|
| MLP, in-distribution | 0.007879 | 0.007879 | 0.007879 |
| FNO, in-distribution | 0.028641 | 0.028641 | 0.028641 |
| FNO, out-of-family | 0.241019 | 0.241019 | 0.241019 |
| POD-ROM, in-distribution | 0.001518 | 0.001518 | 0.001518 |
| DEIM-ROM, in-distribution | 0.043534 | 0.043534 | 0.043534 |

Parameter extrapolation (MLP / FNO) also agrees to four decimals on all three backends: A = 2.5:
0.0545 / 0.0613; nu = 0.15: 0.0305 / 0.0672; A = 0.2: 0.5595 / 0.1664.

The values agree to the precision shown but are not bit-identical. Across the 24 neural-network
metrics that have an MPS counterpart (seed-0 in-distribution, out-of-family and extrapolation errors,
and the per-seed FNO multi-seed values), the largest relative CUDA-vs-MPS difference is 1.2e-6 (MLP,
nu = 0.15 extrapolation); for the in-distribution means alone it is 6.7e-7 (MLP) and 8.7e-8 (FNO).
The same-session CPU differs from MPS by at most 1.0e-6.

The NumPy POD-ROM and DEIM-ROM errors differ from the committed values by at most 1.4e-6 relative.
They are not affected by the network backend; a relevant environment difference is NumPy 2.1.3 on
Colab versus 1.26.4 for the committed results.

### Training outcome

The seed-0 runs reach the historical training result to the printed precision: FNO best validation
MSE 5.3671e-4 at epoch 146, MLP 3.8605e-5 at epoch 198, on CUDA and on the Colab CPU, matching the
committed MPS run. Initial weights and mini-batch order are drawn on the CPU from the fixed seed on
every backend, so the optimisation path is the same up to floating-point differences.

### Multiple seeds (CUDA)

| | Seed 0 | Seed 1 | Seed 2 | Mean ± std |
|---|---|---|---|---|
| FNO, in-distribution | 2.864% | 3.068% | 2.860% | 2.931% ± 0.097% |
| MLP, in-distribution | 0.788% | 0.955% | 0.868% | 0.870% ± 0.068% |

The FNO values reproduce the committed MPS multi-seed study (2.931% ± 0.097%). FNO out-of-family
across seeds: 23.41% ± 0.53%, also matching. MLP seeds 1 and 2 are new CUDA-only measurements; the
committed results have no MPS counterpart for them.

## Timing on the Colab runtime

Training time (single run each, seed 0) and online cost (median of 7 trials x 20 calls, batch 1,
including the copy to host). Both columns were measured on the same Colab runtime in the same
session: the T4 against that runtime's own CPU.

| | Tesla T4 CUDA | Colab CPU |
|---|---|---|
| FNO training, 200 epochs | 20.6 s | 44.7 s |
| MLP training, 200 epochs | 5.6 s | 19.6 s |
| FNO inference | 1.97 ms | 2.90 ms |
| MLP inference | 0.216 ms | 0.210 ms |

* These figures compare the T4 with the CPU of the same Colab runtime. They are not a CUDA-vs-Apple
  comparison and give no speed-up over the committed Apple MPS results.
* The Colab CPU is substantially slower than the machine behind the committed results: the same NumPy
  solver timing gave a median of 22.7 ms and 36.6 ms in the two Colab timing runs, against 7.69 ms in
  `report/research/timing_verification.json`.
* The small MLP shows no inference benefit from CUDA in this measurement.
* CPU-side timings on the shared runtime were noisy (for example the plain ROM varied with a
  standard deviation of 13.7 ms across trials in one run); the training times are single runs.

In this run's `cuda/results/results.json`, the MLP training time in the offline-cost table (5.585 s)
is the time measured in this run, labelled in `offline_cost_provenance`; the historical 8.23 s Mac
figure is not used.

## Files

| Path | Content |
|---|---|
| `provenance.json`, `data_provenance.json`, `git_status.txt` | run, environment and data provenance |
| `pytest.log`, `nvidia_smi.txt`, `pip_freeze.txt` | test run and environment on the runtime |
| `deim_rank_sweep_this_machine.json` | POD/DEIM offline times measured on the runtime (NumPy, CPU) |
| `cuda/*_run_meta*.json` | CUDA training metadata, MLP and FNO, seeds 0 to 2 |
| `cuda/results/` | `results.json`, `results_summary.txt`, `comparison_figure.png`, `fno_multiseed_results.json`, `mlp_multiseed_results.json`, `timing_verification_cuda.json`, `timing_verification_this_machine_cpu.json` |
| `cuda/results/mvp/` | MVP comparison (`mvp_results.txt`, `mvp_comparison.png`) |
| `cpu_reference/` | same-session CPU training metadata and evaluation, seed 0 |
| `run_cuda_executed.ipynb` | the executed notebook with its outputs (commit `028f4b4`) |
| `MANIFEST.sha256` | SHA-256 of every file here |

Every file except this README and the executed notebook is a byte-for-byte copy from the downloaded
archive. Checkpoints (`*.pt`), datasets, the uploaded input archive, smoke-test outputs and training
logs are not stored; the checkpoints can be regenerated with the notebook at the commit above.

## Limitations

* One GPU model, one Colab session, one training run per configuration.
* The run directory name is known from the archive and the notebook output, not from
  `provenance.json` (see the null `run_id` above).
* Run-to-run variation on CUDA was not measured; some cuDNN kernels are nondeterministic.
