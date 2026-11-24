# Frozen keyword-spotting feature cache (Speech Commands v0.02)

A checksummed 49x10 feature cache for the 10-keyword task with unknown and silence classes, built from the official Speech Commands v0.02 archive and its separate test archive. Audio is not kept. The cache gives the next iteration a fixed test population for its stopping sweep.

## What was done

1. Downloaded `speech_commands_v0.02.tar.gz` (2.4 GB) and `speech_commands_test_set_v0.02.tar.gz` (113 MB) from download.tensorflow.org.
2. Used `validation_list.txt` (9,981 lines) and `testing_list.txt` (11,005 lines) from the archive to assign splits by speaker. The two lists do not overlap, and no speaker appears in more than one of train, validation and testing (checked). The separate test archive (`test_bench`) shares 239 speaker hashes with `testing`, so it is a second test population, not a speaker-disjoint one.
3. Built labels for the 12-label setup: 10 target words (yes, no, up, down, left, right, on, off, stop, go), `_unknown_` (the other 25 words) and `_silence_`. Unknown and silence are each 10% of the known-word count per split, following the TensorFlow speech_commands recipe.
4. Computed features at 16 kHz: 40 ms window (640 samples), 20 ms hop (320 samples), n_fft 640, 49 frames x 10 bins. Every clip is padded or trimmed to exactly 1 s before framing. Both a log-mel array (`features_<split>.npy`) and an MFCC array (`mfcc_features_<split>.npy`) are stored, float16.
5. Wrote an index with the requested fields and a SHA-256 per clip feature array, plus mini and preview variants and a checksum file.
6. Wrote `exp_sel_data_out` JSON for the cache (one dataset group, one example per clip) and checked it against the skill's schema.

## Layout

- `scripts/build_feature_cache.py`: the full pipeline (extract, split, generate silence and unknown, featurize, write cache, write index). Re-running it rebuilds everything from the archives.
- `scripts/verify_features.py`: recomputes the log-mel for 200 sampled archive clips with the build script's own function and compares each to its stored `feature_sha256`. Run it after installing the pinned environment to confirm the libraries reproduce the cache.
- `features/features_{train,validation,testing,test_bench}.npy`: log-mel features, float16, N x 49 x 10. **Kept.**
- `features/mfcc_features_{train,validation,testing,test_bench}.npy`: MFCC features, same shape. **Kept.**
- `features/index.json`: flat list, one record per clip, with fields `clip_id`, `speaker_hash`, `label`, `label_index`, `split`, `source_path_in_archive`, `feature_sha256`, plus `feature_index` (row in the split's `.npy`) and `is_generated`.
- `features/mini_index.json` (first 200 rows per split), `features/preview_index.json` (first 20 rows per split).
- `features/split_stats.json`: list line counts, per-split label counts, seeds, feature settings.
- `data.py`: reads `features/index.json` and the arrays, re-checks shape, dtype and every row's `feature_sha256`, checks that archive clips exist in `temp/datasets/extracted/`, and writes the files below.
- `full_data_out.json`: `exp_sel_data_out` JSON with one dataset group, `speech_commands_v0.02_kws12`, and 51,148 examples (one per clip). `input` = clip_id, `output` = label, `metadata_*` = split, label index, speaker hash, `metadata_feature_index` (row in the split's `.npy`), feature SHA-256, generated flag. Schema-valid.
- `mini_data_out.json` (first 3 examples) and `preview_data_out.json` (same, strings cut at 200 characters).
- `pyproject.toml` and `uv.lock`: every installed package pinned to the version used here.
- `checksums.sha256`: SHA-256 of every array, index, split-stats file, output JSON and script.
- `dataset_report.md`: counts, seeds, settings and checks.
- `reproducibility.md`: step-by-step reproduction on Ubuntu.
- `temp/datasets/`: raw archives and extraction directory (removed after the round; see below).

## How to run

```
uv sync --python 3.12                       # creates .venv from pyproject.toml (pinned)
uv run scripts/verify_features.py           # optional: checks the pinned libraries reproduce the stored features
uv run data.py                              # rebuilds full/mini/preview_data_out.json from features/
```

To rebuild the arrays from the archives, place the two archives in `temp/datasets/archives/` and run:

```
uv run scripts/build_feature_cache.py --workers 2
```

The script extracts the archives on first run, and then rebuilds `features/`.

## Label indices

| label | index |
|---|---|
| yes, no, up, down, left, right, on, off, stop, go | 0 to 9 |
| `_silence_` | 10 |
| `_unknown_` | 11 |

## Restoring removed files

The raw archives, the extraction directory and the virtual environment are removed after the round. To restore them and rebuild:

```
mkdir -p temp/datasets/archives
curl -L -o temp/datasets/archives/speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
curl -L -o temp/datasets/archives/speech_commands_test_set_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz
uv sync --python 3.12
uv run scripts/build_feature_cache.py --workers 2
```

The extraction directory `temp/datasets/extracted/` is rebuilt by the build command. `.venv/` is restored by `uv sync`.

The Python bytecode cache `scripts/__pycache__/` is regenerated by running any script, for example:

```
uv run data.py
```

## Departures from the plan, and why

- **FFT size.** The plan's example was n_fft 1024 with a 640 window. With `center=False`, librosa frames at n_fft, which gives 47 frames, not 49. The script uses n_fft 640 so the frame count is exactly 49 (`1 + (16000 - 640) // 320`). The window and hop are as specified.
- **Silence sampling.** Silence is cut from the 6 `_background_noise_` files. Each file is split by time into 80% train, 10% validation and 10% test, so no noise segment is shared across splits. The TensorFlow recipe does not do this. Clip IDs are `_background_noise_/<file>@<sample offset>`, and the sampling is without replacement with seed 43.
- **test_bench.** The separate 4,890-clip test archive is kept as split `test_bench`. Its `_silence_` and `_unknown_` folders are used as delivered. Its filenames reuse v0.02 names, so its `clip_id` is prefixed with `speech_commands_test_set_v0.02/`.
- **Validation of the index.** The aii-json validator (`aii_json_validate_schema.py`) and its mini/preview formatter both refuse any file outside the skill's project root (`input_file must be under the project directory or /tmp`), so they cannot run on this workspace. The mini and preview files were therefore produced by `data.py`, following the same rule (first 3 examples; strings cut at 200 characters). The full, mini and preview `exp_sel_data_out` files were checked with jsonschema directly, using the skill's schema, and each has zero errors.
- **Accuracy check.** Not run. The plan's step 8 is optional, and this artifact is the cache, not a result.
