# Early-Stop Keyword Spotting Energy Test

Tests whether a causal DS-CNN keyword spotter that stops on a calibrated prefix sequential probability ratio test (SPRT) uses less **modelled** energy per decision than (a) a full-window classifier and (b) a confidence-threshold early exit matched to the same validation error. Dataset: Speech Commands v2 (HF mirror, see Departures). Energy is modelled, not measured.

Headline (3 seeds, mean over seeds, primary alpha = beta = 0.01, f = 0.2, L-Perf constant):

- The calibrated SPRT stops before column 49 on 85% of decisions, mean prefix 26.5 columns, saving 36.7% of modelled energy against the full window (CI per seed about 35-37%).
- The SPRT and the matched threshold have nearly the same error (one-vs-rest FA 3.1 / FR 3.1 vs 2.9 / 3.1 %). The SPRT is about 2% cheaper than the threshold at alpha = 0.01, and the threshold is cheaper at alpha = 0.001.
- The measured error does **not** meet the designed bound (FA and FR <= alpha + 0.5 pt) for alpha <= 0.02. Error on decisions made at a Wald boundary is inside the bound (FA 0.5%, FR 0.7% at alpha = 0.01). Error on the 15% of clips that reach column 49 without a boundary crossing is 18% FA and 13% FR, and this drives the failure. This is the expected disconfirmation of the literal prefix-SPRT guarantee (prefix log-odds are not independent observations). See `method_report.md`.

Full tables, departures from the plan, and pass or fail status of each pre-stated criterion are in **`method_report.md`**.

## Layout

- `method.py`: the whole pipeline. Subcommands: `prep`, `units`, `train --seed S`, `evaluate --seed S`, `aggregate`.
- `make_report.py`: renders `method_report.md` from `method_out.json`.
- `method_out.json`: schema-conformant summary (exp_gen_sol_out: one example per seed and alpha, with the numbers in `metadata_*`, and the summary, Holm and criteria in `metadata`). Built by `to_schema.py`.
- `results/method_detailed.json`: the full per-seed, per-system, per-alpha record: measured FA and FR with Wilson 95% intervals, mean prefix length, stop fraction, modelled energy for each f, bootstrap savings, error split, Holm, criteria.
- `full_method_out.json`, `mini_method_out.json`, `preview_method_out.json`: full, first-3-example and truncated copies of `method_out.json` (aii-json format script).
- `audit_rederive.py`, `logs/audit_rederive.out`: independent re-derivation of the headline numbers and a placebo test.
- `method_report.md`: design, tables, error split, bootstrap, sensitivity to f, criteria, departures and limits.
- `results/seed{0,1,2}.json`: per-seed results (temperatures, stride, autocorrelation, all systems).
- `results/train_log_seed*.json`: per-epoch training losses and causality check.
- `figures/energy_error.pdf` and `.png`: modelled energy against one-vs-rest accuracy for each system (spec in `figures/energy_error_spec.json`).
- `checkpoints/seed{0,1,2}.pt`: trained DS-CNN weights (about 0.1 MB each; kept at `checkpoints/` in this workspace and in the repository, see `.aii/manifest.yaml`). `smoke_seed0.pt` is the 1,000-clip smoke test.
- `logs/`: run logs.
- `features/`, `data_raw/`, `.venv/`: heavy, regenerable or redownloadable (see Restoring removed files).

## How to run

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy soundfile datasets loguru torch pyarrow matplotlib
.venv/bin/python method.py units            # framing, causality, Wald bounds, toy SPRT
.venv/bin/python method.py prep             # decode clips, MFCC features, 12-class indexes
for s in 0 1 2; do
  .venv/bin/python method.py train --seed $s --epochs 30
  .venv/bin/python method.py evaluate --seed $s
done
.venv/bin/python method.py aggregate        # results/method_detailed.json
.venv/bin/python to_schema.py               # method_out.json (exp_gen_sol_out)
.venv/bin/python audit_rederive.py          # independent re-derivation + placebo
.venv/bin/python make_report.py             # method_report.md
```

Training runs in about 1 minute per seed on MPS (about 3 s per epoch); without MPS it is slower but still short.

## Departures

- Data came from `danjacobellis/speech_commands_v2` on the Hugging Face Hub (Opus audio, split sizes equal to the official v2 counts). The official archive was unreachable from the run, so the official list files were not used to check membership.
- Silence is synthetic brown noise, since the `_background_noise_` files were not available.
- Energy is modelled from a sourced constant, not measured on a board.

## Restoring removed files

Heavy paths in `.aii/manifest.yaml` that are deleted after the round:

- `.venv/`: `uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy soundfile datasets loguru torch pyarrow matplotlib`
- `data_raw/`: download `train`, `validation` and `test` parquet (train is then split into `train_part_000..002.parquet` by `shard_train.py`, because the 254 MB file is over the 100 MB limit) from `https://huggingface.co/datasets/danjacobellis/speech_commands_v2/resolve/main/data/{split}-00000-of-00001.parquet`. `restore.sh` does this and rebuilds `features/`.
- `features/`: `.venv/bin/python method.py prep`, after `data_raw/` is restored.
- `__pycache__/`: `.venv/bin/python method.py units` (Python bytecode, recreated automatically on import).

`restore.sh` runs all three steps in order.
