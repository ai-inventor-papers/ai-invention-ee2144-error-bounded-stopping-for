# Reproducing the results

This folder is one artifact of a repository. Start from a clone of that repository and `cd` into this folder (`early-stop-kws-energy` workspace). Every path below is relative to this folder.

## What was actually run

The results were produced on **macOS (Darwin 25.4, Apple silicon, 16 CPU cores, 32 GB unified memory)** with Python 3.12.10 and PyTorch's **MPS** backend (`device = mps if available`). They were **not** run on Ubuntu or on a CUDA GPU. An Ubuntu CPU run should follow the same commands (the code falls back to `cpu`), but it has not been tested, and runtimes will differ.

## 1. System packages and Python

- Python 3.12 (3.12.10 was used). The venv is created with `uv`, not `pip`.
- `uv` installed on the system (any recent version).
- No system packages beyond what `soundfile` needs (`libsndfile` 1.2.x with Opus support; the pip wheel bundles it on macOS arm64).

## 2. Environment

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-frozen.txt
```

`requirements-frozen.txt` is the full `uv pip freeze` of the working venv (48 pins). `pyproject.toml` pins the direct dependencies to the same versions:
numpy 2.5.5, scipy 1.18.1, soundfile 0.14.0, datasets 5.2.0, loguru 0.8.0, torch 2.16.0.dev20261116+aii1, pyarrow 26.0.1, matplotlib 3.11.2, jsonschema 4.26.0.

The torch pin is a local build of the guest's MPS wheel (`+aii1`). It is not on PyPI. On another machine, install the nearest torch 2.x build with MPS or CPU support, and expect small numerical differences.

## 3. Data, downloads, and environment variables

- Dataset: `danjacobellis/speech_commands_v2` on the Hugging Face Hub (Opus audio, official split sizes 84,843 / 9,981 / 11,005). It is downloaded by `restore.sh` into `data_raw/` (about 330 MB). The 254 MB `train` file is then split into three row-group-aligned shards, `data_raw/train_part_000..002.parquet` (about 86 MB each), by `shard_train.py`. The shards are read by `method.py prep`, and the original file is removed. `shard_train.py --verify` confirms the shards hold the same rows and content as the original. No account or token is needed.
- The original Speech Commands v0.02 archive (`download.tensorflow.org`) was **not** used: it was unreachable from the run. The split membership is the mirror's, not checked against `validation_list.txt` / `testing_list.txt`.
- No API keys. No OpenRouter or LLM calls.
- Environment variables: none required by the code. The run environment had `HF_HOME` set to a shared Hugging Face cache; the download works without it.
- Uploaded user files: none were supplied for this artifact.

```bash
bash restore.sh      # venv (optional), data_raw/ download, then features/ via prep
```

## 4. Commands, in the order they were run

All commands are run from this folder.

```bash
.venv/bin/python method.py units                 # framing, causality, Wald bounds, toy SPRT (~1 s)
.venv/bin/python method.py prep                  # decode 107k clips, MFCC 49x10, 12-class indexes (~2.5 min)
.venv/bin/python method.py train --seed 0 --epochs 2 --smoke   # 1,000-clip smoke check (seconds)
for s in 0 1 2; do
  .venv/bin/python method.py train --seed $s --epochs 30        # early stopping, patience 4 (~1 min/seed on MPS)
  .venv/bin/python method.py evaluate --seed $s                 # calibration, rules, errors, bootstrap (~20 s/seed)
done
.venv/bin/python method.py aggregate             # results/method_detailed.json (Holm, criteria)
.venv/bin/python to_schema.py                    # method_out.json (exp_gen_sol_out)
# aii-json format script (skill, external): full/mini/preview variants of method_out.json
.venv/bin/python make_report.py                  # method_report.md
.venv/bin/python audit_rederive.py > logs/audit_rederive.out   # independent re-derivation + placebo
```

Notes on the run that a reader should know:
- `evaluate` was run twice per seed. The first pass chose the decorrelated stride on column-level increments (stride 1, which is an artifact: even columns repeat a frame). The final results use the frame-level stride (5 frames). The bootstrap p-value formula was also changed to add one to the counts. Every reported number comes from the final `evaluate` pass.
- Figure: `figures/energy_error.png` and `.pdf` were rendered from `figures/energy_error_spec.json` with the aii-data-fig-gen skill's generator (`chart_gen.py --spec`). That generator is external to this folder.
- The seed is fixed for `numpy`, `random` and `torch` at each `train --seed S`, and for the bootstrap (`1000 + seed`) and the unknown-clip and silence sampling (`SPLIT_SEED`). Determinism was not tested: no second full re-run was done to confirm bit-identical numbers, on this platform or any other.

## 5. Outputs and where they appear

| file | content | paper location |
|---|---|---|
| `method_out.json` | schema-conformant summary: 18 examples (seed x alpha) plus summary, Holm and criteria in `metadata` | main results table, criteria |
| `results/method_detailed.json` | every per-seed, per-system, per-alpha number (FA/FR with Wilson CIs, m, stop fraction, energy for each f, bootstrap, error split) | all tables |
| `full_method_out.json`, `mini_method_out.json`, `preview_method_out.json` | full, first-3 and truncated copies of `method_out.json` | none (inspection only) |
| `method_report.md` | design, tables, error split, bootstrap, Holm, f sensitivity, criteria, departures | main text and appendix |
| `figures/energy_error.pdf` | energy-error curve (L-Perf, f = 0.2) | energy-error figure |
| `logs/audit_rederive.out` | independent re-derivation of the headline numbers and placebo test | reproducibility appendix |
| `checkpoints/seed{0,1,2}.pt` | trained DS-CNN weights (about 0.1 MB each) | not in the paper |

Headline numbers a reader should get (seed 0 agreement with `results/method_detailed.json` is shown in `logs/audit_rederive.out`):
- Primary SPRT, alpha = beta = 0.01: mean prefix about 26.5 columns, 85% of decisions stop before column 49, modelled energy saving about 37% against the full window at f = 0.2 (about 36% to 41% over f = 0.1 to 0.4).
- The measured error exceeds the designed bound (FA and FR <= alpha + 0.5 pt) for alpha <= 0.02. The error is inside the bound on boundary stops and is driven by the column-49 fallback (about 15% of clips).
- Energy is **modelled** (E = f E_full + (1 - f) E_full m / 49, with E_full from arXiv 2505.15622). It is not measured on a board.

## 6. Restoring removed files

`restore.sh` rebuilds `data_raw/` and `features/`. `.venv/` is rebuilt by step 2. The manifest in `.aii/manifest.yaml` gives the command for each deleted path.
