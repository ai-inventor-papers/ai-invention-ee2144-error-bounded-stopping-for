# Dataset report: frozen keyword-spotting feature cache

Source: Speech Commands v0.02 (official archive) and speech_commands_test_set_v0.02.
Audio is not kept; features only.

## List line counts

| list | lines |
|---|---|
| validation_list.txt | 9,981 |
| testing_list.txt | 11,005 |
| validation and testing overlap | 0 |
| word clips in archive (excluding `_background_noise_`) | 105,829 |

Speakers: no speaker hash appears in more than one of train, validation and testing.

## Clip counts per split and label

Known words are the 10 target words from the official lists. `_unknown_` and `_silence_` are each 10% of the split's known-word count (seeds: unknown 42, silence 43).

| split | known | unknown | silence | total |
|---|---|---|---|---|
| train | 30,769 | 3,077 | 3,077 | 36,923 |
| validation | 3,703 | 371 | 371 | 4,445 |
| testing | 4,074 | 408 | 408 | 4,890 |
| test_bench (separate test archive, as delivered) | - | - | - | 4,890 |

Per-label counts for every split are in `features/split_stats.json`. In test_bench, `_silence_` (408) and `_unknown_` (408) come from the test archive's own folders.

Sizes: train features 36 MB (log-mel) and 36 MB (MFCC); validation 4.4 MB each; testing and test_bench 4.8 MB each. Index JSON 17 MB, full_data_out.json 16 MB (one dataset group, 51,148 examples). Total under 150 MB.

## Feature settings

| setting | value |
|---|---|
| sample rate | 16,000 Hz |
| clip length | 16,000 samples (padded or trimmed) |
| window | 640 samples (40 ms), Hann |
| hop | 320 samples (20 ms) |
| n_fft | 640 |
| frames | 49 (`center=False`) |
| log-mel bins | 10, log floor 1e-6 |
| MFCC | 40 mel bins, 10 coefficients, same framing |
| storage | float16, shape N x 49 x 10 |

## Sanity checks

- Shapes and dtype: all four arrays are N x 49 x 10, float16, for both feature sets.
- NaN/inf: none.
- No clip ID appears twice in the index (51,148 rows).
- Per-class feature means (train, log-mel): `yes` (label 0) mean -9.31, std 4.00; `_silence_` (10) mean -4.13, std 3.64; `_unknown_` (11) mean -9.32, std 4.21. Silence is louder than the word classes, as expected for background noise.
- Speaker overlap: zero between train, validation and testing (the three list-based splits).
- Speaker overlap between `testing` and `test_bench`: 239 speaker hashes appear in both. The separate test archive is not speaker-disjoint from `testing_list.txt`. Read `test_bench` as a second test population, not as an independent speaker set.
- Schema (`exp_sel_data_out` projection): zero errors under jsonschema.

## Not done

- DS-CNN accuracy check (plan step 8). It was optional and was not run. No accuracy figure is claimed here.
- The aii-json validator and formatter refuse files outside the skill's project root, so they cannot run on this workspace. Schema validation was done with jsonschema against the skill's `exp_sel_data_out.json`: zero errors for `full_data_out.json`, `mini_full_data_out.json` and `preview_full_data_out.json`. Mini and preview were produced by `data.py`.

## Files

See README.md for the layout, the run command and the restore commands.
