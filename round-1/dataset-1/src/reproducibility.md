# Reproducing the frozen keyword-spotting feature cache

This describes what was actually run to produce the cache in this folder. It is a dataset artifact: it has no training or evaluation step, and no model is fit.

## 1. Getting this artifact

This folder is published as one folder of a public GitHub repository. Clone the repository, then change into this artifact's folder. All paths below are relative to that folder.

```
git clone <repository-url>
cd <repository>/<this-artifact-folder>
```

The raw archives are not in the repository (they are removed after the round and are about 2.5 GB). Step 3 downloads them.

No user-uploaded files are used. No other artifact is read.

## 2. System and Python

- Built on macOS (Apple silicon, arm64). The reproduction target is Ubuntu. The outputs have not been rebuilt on Ubuntu, so the hash check in step 5 is the test of whether they match.
- Python 3.12. The build used a uv-managed CPython 3.12.10. Later steps used CPython 3.12.14. Either works; `uv` picks the interpreter from `--python 3.12`.
- `uv` 0.6.14 (or newer). Install it from https://docs.astral.sh/uv/.
- No system packages are required beyond a C toolchain that `numba` and `librosa` wheels already cover on common Linux. `tar` and `curl` are used in step 3.

## 3. Environment and downloads

Create the environment from the pinned file:

```
uv sync --python 3.12
```

`pyproject.toml` pins every installed package. The ones the pipeline imports directly:

| package | version |
|---|---|
| numpy | 2.5.5 |
| librosa | 1.0.1 |
| soundfile | 0.14.0 |
| loguru | 0.8.0 |

The full list (with transitive dependencies) is in `pyproject.toml`; `uv.lock` holds the resolved set. Note that `numba==0.68.1` and `llvmlite==0.50.0` are included through librosa.

Download the two archives (about 2.5 GB total) into the folder that the build expects:

```
mkdir -p temp/datasets/archives
curl -L -o temp/datasets/archives/speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
curl -L -o temp/datasets/archives/speech_commands_test_set_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
```

Environment variables and API keys: none. The pipeline calls no API and reads no credentials. The Hugging Face cache variables set by the run environment are not used.

## 4. Commands, in the order they were run

1. **Build the cache.** This extracts the two archives (on the first run only), splits the clips with the official lists, generates the silence and unknown clips from `_background_noise_` with seeds 42 and 43, computes features with 2 worker processes, and writes `features/` and the index.

   ```
   uv run scripts/build_feature_cache.py --workers 2
   ```

   Runtime on the machine used: extraction of both archives took about 1 minute after download; featurization of all four splits took about 40 seconds (01:21:19 to 01:21:47 in `logs/build_stdout.txt`); total about 2 minutes. Peak memory was not measured.

   Three runs were needed. The first two failed, and the failures are kept in the logs:
   - The first run stopped with `shape mismatch (47, 10)`: the frame count came out at 47, not 49. `n_fft` is set to 640 in the final script, which with `center=False` gives 49 frames.
   - The second run stopped with `duplicate clip ids` for one silence clip. The silence sampler was changed so no clip is drawn twice, and the index then had zero duplicates.
   - The third run succeeded. Its log is `logs/build_stdout.txt`. The earlier failures are in `logs/build_feature_cache.log`.

2. **Verify the pinned environment reproduces the stored features.** This recomputes the log-mel for 200 sampled archive clips (50 per split) and checks each against `feature_sha256`.

   ```
   uv run scripts/verify_features.py
   ```

   Result on the machine used: all 200 sampled clips reproduced their stored hashes.

3. **Write the `exp_sel_data_out` JSON and the mini and preview files.** This re-checks every row's hash, checks archive sources exist, and writes the three JSON files at the folder root.

   ```
   uv run data.py
   ```

   Result: 51,148 examples in one dataset group, `speech_commands_v0.02_kws12`. The mini file keeps 3 examples and the preview file keeps 3 with strings cut at 200 characters.

4. **Check the checksums.** Run from the folder root:

   ```
   shasum -a 256 -c checksums.sha256
   ```

No seeds beyond 42 (unknown) and 43 (silence) are used. The split assignment is fixed by the official lists, so the cache has no other random choice.

## 5. What a reader should get

Compare your files to `checksums.sha256`. If your Ubuntu rebuild differs in a few bytes of the float16 arrays, the cause is most likely a different numerical build of `librosa` or `numpy`. Step 2 tells you whether the features match row by row.

Key numbers (also in `dataset_report.md` and `features/split_stats.json`):

| split | clips | source |
|---|---|---|
| train | 36,923 | `training` clips from `speech_commands_v0.02` (not in `testing_list.txt` or `validation_list.txt`), plus generated silence and unknown |
| validation | 4,445 | `validation_list.txt`, plus generated silence and unknown |
| testing | 4,890 | `testing_list.txt`, plus generated silence and unknown |
| test_bench | 4,890 | `speech_commands_test_set_v0.02` archive, as delivered |

Feature settings: 16 kHz, 640-sample window, 320-sample hop, 49 frames by 10 bins, float16. Label indices: `yes`..`go` = 0 to 9, `_silence_` = 10, `_unknown_` = 11.

The cache is an input to a later step: the next iteration's stopping sweep reads `features/features_testing.npy` (primary test population) and `features/features_test_bench.npy` (second test population) by row. Those arrays stay on the run's storage volume; the repository publishes the folder and its JSON files, not the arrays.

No accuracy result is reported. The optional DS-CNN check (plan step 8) was not run.

## 6. Limits of this reproduction

- Only one platform (macOS arm64) produced the cache. Ubuntu has not been tested.
- The archives are fetched from `download.tensorflow.org`. If that host changes, get the same files from the Hugging Face mirror `google/speech_commands` and check their SHA-256 against the dataset card.
