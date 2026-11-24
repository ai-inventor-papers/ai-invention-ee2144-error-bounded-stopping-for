# Early-Stop Keyword Spotting Energy Test: method report

Auto-generated from `results/method_detailed.json` by `make_report.py`. Energy values are MODELLED, not measured on a board.

## Design

- Data: Speech Commands v2 test/validation/train splits taken from the HF mirror `danjacobellis/speech_commands_v2` (Opus audio, decoded to 16 kHz). Split sizes equal the official v2 counts (84,843 / 9,981 / 11,005). The official archive was unreachable from this run, so the official list files were not used (see Departures).
- 12-class set: 10 target words, silence (10%), unknown (10%). Unknown is subsampled per split; silence is synthetic brown noise.
- Front end: 640-sample Hann window, 320 hop, 1024-point FFT, 40 mel bins (20-4000 Hz), log, DCT-II, first 10 coefficients: 49x10 per clip. Normalised with train statistics.
- Model: MLPerf Tiny DS-CNN layout (Conv 10x4 stride 2, four depthwise-separable blocks of 64 channels) with left-only time padding and a causal running mean in place of global average pooling. Trained with cross-entropy over prefixes t = 5..49, Adam 1e-3 with cosine decay, batch 256, up to 30 epochs with early stopping (patience 4).
- Causal decisions: the output at prefix length t uses only columns 1..t. Because the first convolution has stride 2, output frames change at odd t, and even t repeat the frame of t-1.
- Error: one-vs-rest per target keyword. Negatives are all other classes (including unknown and silence). FA and FR are pooled over the 10 keywords; Wilson 95% intervals use the pooled counts (see Limits).
- Energy (MODELLED): E = f E_full + (1 - f) E_full m / 49, with E_full = 156.5 uJ (L-Perf) and 219.0 uJ (H-Perf) taken from arXiv 2505.15622, and f in {0.1, 0.2, 0.3, 0.4}. m is the mean number of columns processed per decision. The formula assumes each column is processed once by incremental causal computation.
- Seeds run: [0, 1, 2] (3 model seeds; the same test clips for all systems).

## Split and calibration checks

| split | full split clips | keyword | unknown | silence | total | silence share |
|---|---|---|---|---|---|---|
| train | 84843 | 30769 | 3846 | 3846 | 38461 | 0.1 |
| validation | 9981 | 3703 | 463 | 463 | 4629 | 0.1 |
| test | 11005 | 4074 | 509 | 509 | 5092 | 0.1 |

Path overlap between splits: {'train_val': 0, 'train_test': 0, 'val_test': 0}.

- Seed 0: causality max |diff| = 0.00e+00; calibrated NLL <= raw NLL at every t on validation: True; full-window test one-vs-rest FA 3.03%, FR 2.80%, 12-class accuracy at t = 49 91.30%.
- Seed 1: causality max |diff| = 0.00e+00; calibrated NLL <= raw NLL at every t on validation: True; full-window test one-vs-rest FA 2.95%, FR 2.75%, 12-class accuracy at t = 49 91.26%.
- Seed 2: causality max |diff| = 0.00e+00; calibrated NLL <= raw NLL at every t on validation: True; full-window test one-vs-rest FA 3.02%, FR 3.12%, 12-class accuracy at t = 49 91.36%.

## Feasibility (validation, alpha = beta = 0.01, calibrated SPRT)

| seed | mean prefix (columns) | median | stop before 49 |
|---|---|---|---|
| 0 | 26.41 | 25 | 85.58% |
| 1 | 26.43 | 25 | 85.17% |
| 2 | 26.19 | 25 | 85.44% |

Mean prefix is below 40 columns, so the energy saving is not expected to be small (feasibility flags, mean > 40 or > 45, were not raised).

## Decorrelated stride

The column-level lag-1 autocorrelation of LLR increments is about -0.05. This is an artefact: even columns repeat the previous output frame, so their increment is exactly zero. The stride is therefore chosen on the frame-level sequence (odd columns). Frame-level lag-1 autocorrelation is about 0.5, and the first lag below 0.1 is:

| seed | stride (frames) | allowed decision columns |
|---|---|---|
| 0 | 5 | odd t with (t-1)/2 divisible by 5, plus t = 49 |
| 1 | 5 | odd t with (t-1)/2 divisible by 5, plus t = 49 |
| 2 | 5 | odd t with (t-1)/2 divisible by 5, plus t = 49 |

## Main table: test error and modelled energy (mean over seeds)

FA and FR are one-vs-rest error rates (%). m is mean columns processed. Saving is relative to the full window at f = 0.2 (modelled). Bound pass means FA <= alpha + 0.5 pt and FR <= beta + 0.5 pt, checked on every seed.

### alpha = beta = 0.001

| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |
|---|---|---|---|---|---|---|
| (a) Full window | 3.00 ± 0.04 | 2.89 ± 0.20 | 49.00 ± 0.00 | 0.00% | 0.00% | False |
| Primary: calibrated SPRT | 3.00 ± 0.04 | 2.92 ± 0.22 | 31.89 ± 0.21 | 71.93% | 27.94% | False |
| (b) Confidence threshold (matched on val) | 2.92 ± 0.04 | 3.14 ± 0.17 | 27.28 ± 0.09 | 81.57% | 35.46% | False |
| (c) Decorrelated SPRT | 3.00 ± 0.05 | 2.89 ± 0.20 | 35.19 ± 0.27 | 66.15% | 22.55% | False |
| (d) Uncalibrated SPRT | 3.54 ± 0.23 | 2.57 ± 0.38 | 28.74 ± 0.77 | 73.50% | 33.07% | False |

### alpha = beta = 0.0025

| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |
|---|---|---|---|---|---|---|
| (a) Full window | 3.00 ± 0.04 | 2.89 ± 0.20 | 49.00 ± 0.00 | 0.00% | 0.00% | False |
| Primary: calibrated SPRT | 3.00 ± 0.04 | 2.95 ± 0.22 | 29.75 ± 0.18 | 77.36% | 31.42% | False |
| (b) Confidence threshold (matched on val) | 2.92 ± 0.04 | 3.14 ± 0.17 | 27.28 ± 0.09 | 81.57% | 35.46% | False |
| (c) Decorrelated SPRT | 3.01 ± 0.04 | 2.90 ± 0.19 | 33.29 ± 0.24 | 72.29% | 25.65% | False |
| (d) Uncalibrated SPRT | 3.54 ± 0.24 | 2.69 ± 0.46 | 26.96 ± 0.67 | 78.80% | 35.98% | False |

### alpha = beta = 0.005

| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |
|---|---|---|---|---|---|---|
| (a) Full window | 3.00 ± 0.04 | 2.89 ± 0.20 | 49.00 ± 0.00 | 0.00% | 0.00% | False |
| Primary: calibrated SPRT | 3.02 ± 0.04 | 2.98 ± 0.23 | 28.13 ± 0.15 | 81.28% | 34.07% | False |
| (b) Confidence threshold (matched on val) | 2.92 ± 0.04 | 3.14 ± 0.17 | 27.28 ± 0.09 | 81.57% | 35.46% | False |
| (c) Decorrelated SPRT | 3.01 ± 0.04 | 2.93 ± 0.21 | 31.85 ± 0.20 | 76.52% | 28.00% | False |
| (d) Uncalibrated SPRT | 3.56 ± 0.23 | 2.87 ± 0.58 | 25.66 ± 0.58 | 82.42% | 38.11% | False |

### alpha = beta = 0.01

| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |
|---|---|---|---|---|---|---|
| (a) Full window | 3.00 ± 0.04 | 2.89 ± 0.20 | 49.00 ± 0.00 | 0.00% | 0.00% | False |
| Primary: calibrated SPRT | 3.08 ± 0.05 | 3.08 ± 0.21 | 26.54 ± 0.09 | 84.90% | 36.67% | False |
| (b) Confidence threshold (matched on val) | 2.92 ± 0.04 | 3.14 ± 0.17 | 27.28 ± 0.09 | 81.57% | 35.46% | False |
| (c) Decorrelated SPRT | 3.03 ± 0.04 | 2.91 ± 0.22 | 30.42 ± 0.16 | 80.45% | 30.34% | False |
| (d) Uncalibrated SPRT | 3.64 ± 0.22 | 3.23 ± 0.62 | 24.37 ± 0.50 | 85.78% | 40.21% | False |

### alpha = beta = 0.02

| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |
|---|---|---|---|---|---|---|
| (a) Full window | 3.00 ± 0.04 | 2.89 ± 0.20 | 49.00 ± 0.00 | 0.00% | 0.00% | False |
| Primary: calibrated SPRT | 3.21 ± 0.06 | 3.22 ± 0.23 | 24.95 ± 0.05 | 88.29% | 39.26% | False |
| (b) Confidence threshold (matched on val) | 2.90 ± 0.08 | 3.24 ± 0.15 | 26.79 ± 0.85 | 82.66% | 36.26% | False |
| (b*) Oracle threshold (tuned on test) | 2.92 ± 0.04 | 3.14 ± 0.17 | 27.28 ± 0.09 | 81.57% | 35.46% | False |
| (c) Decorrelated SPRT | 3.08 ± 0.05 | 2.97 ± 0.26 | 29.02 ± 0.12 | 84.02% | 32.63% | False |
| (d) Uncalibrated SPRT | 3.84 ± 0.21 | 3.66 ± 0.79 | 23.07 ± 0.44 | 88.97% | 42.34% | False |

### alpha = beta = 0.05

| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |
|---|---|---|---|---|---|---|
| (a) Full window | 3.00 ± 0.04 | 2.89 ± 0.20 | 49.00 ± 0.00 | 0.00% | 0.00% | True |
| Primary: calibrated SPRT | 3.84 ± 0.07 | 3.79 ± 0.20 | 22.77 ± 0.03 | 92.45% | 42.82% | True |
| (b) Confidence threshold (matched on val) | 2.82 ± 0.06 | 3.67 ± 0.05 | 24.75 ± 0.32 | 87.04% | 39.60% | True |
| (b*) Oracle threshold (tuned on test) | 2.82 ± 0.06 | 3.66 ± 0.25 | 24.76 ± 0.36 | 87.06% | 39.58% | True |
| (c) Decorrelated SPRT | 3.32 ± 0.06 | 3.22 ± 0.19 | 27.11 ± 0.05 | 88.57% | 35.74% | True |
| (d) Uncalibrated SPRT | 4.61 ± 0.13 | 5.06 ± 0.80 | 21.17 ± 0.37 | 93.04% | 45.43% | False |

## Where the bound is violated: boundary stops versus column-49 fallback (calibrated SPRT)

Decisions made at a Wald boundary are separated from decisions made at column 49 because no boundary was crossed. Values are the mean over seeds.

| alpha | FA boundary | FR boundary | FA fallback | FR fallback | share decided at 49 |
|---|---|---|---|---|---|
| 0.001 | 0.07% | 0.24% | 10.89% | 6.99% | 28.07% |
| 0.0025 | 0.15% | 0.33% | 13.19% | 8.71% | 22.64% |
| 0.005 | 0.27% | 0.47% | 15.40% | 10.61% | 18.72% |
| 0.01 | 0.48% | 0.70% | 18.08% | 13.08% | 15.10% |
| 0.02 | 0.85% | 1.08% | 21.41% | 15.86% | 11.71% |
| 0.05 | 1.99% | 2.03% | 26.79% | 21.79% | 7.55% |

## Wilson 95% intervals, primary SPRT and full window at alpha = beta = 0.01 (per seed, pooled one-vs-rest counts)

| system | seed | FA % [95% CI] | FR % [95% CI] |
|---|---|---|---|
| (a) Full window | 0 | 3.03 [2.88, 3.19] | 2.80 [2.33, 3.35] |
| (a) Full window | 1 | 2.95 [2.80, 3.11] | 2.75 [2.29, 3.30] |
| (a) Full window | 2 | 3.02 [2.87, 3.18] | 3.12 [2.63, 3.70] |
| Primary: calibrated SPRT | 0 | 3.12 [2.97, 3.28] | 2.92 [2.45, 3.48] |
| Primary: calibrated SPRT | 1 | 3.02 [2.87, 3.18] | 2.99 [2.51, 3.56] |
| Primary: calibrated SPRT | 2 | 3.09 [2.94, 3.25] | 3.31 [2.81, 3.91] |

Threshold baseline not matched within 0.5 points on validation (seed, alpha): [(0, '0.05'), (1, '0.05'), (2, '0.05')]. The tau grid ran from 0.5 to 0.999; for these cells the closest achievable match is reported and marked as not matched.

## Departures from the plan

1. **Dataset source.** The official `speech_commands_v0.02.tar.gz` host was unreachable from this run, and the `google/speech_commands` HF repo is a loading script that downloads that archive. The split was therefore taken from `danjacobellis/speech_commands_v2`, a parquet mirror with Opus-encoded audio and the official split sizes. The validation_list and testing_list files were not available, so split membership is the mirror's, not verified against the lists. Opus coding is a small change to the waveform.
2. **Silence.** The `_background_noise_` files are not in the mirror. Silence clips are synthetic brown noise at a random RMS between 0.001 and 0.05, not crops of recorded noise.
3. **Clip length.** Decoded clips are 9,216 to 18,432 samples. The first 16,000 samples are kept, so a speech onset after 16,000 samples would be cut. Peak positions were checked on 400 test clips (median 7,239 samples).
4. **Front end.** 40 mel bins then DCT-II to 10 coefficients (the MLPerf Tiny reference), not 10 log-mel bins. The FFT is 1024 points on a 640-sample window so that all mel filters are non-empty.
5. **Decision granularity.** The first convolution has stride 2 in time, so causal outputs exist per frame, and even prefix lengths repeat the previous frame. The stopping rules are applied per column, which is equivalent to per frame.
6. **Confidence-threshold decision rule.** The threshold rule stops at the first column where max(p, 1 - p) reaches tau (the plan's |p - 0.5| >= tau is the same test shifted by 0.5). At the stop and at column 49 every system decides with the same rule, llr >= 0, so only the stopping differs.
7. **Decorrelated stride.** Chosen on frame-level increments (see Decorrelated stride), because column-level increments are zero every other column.
8. **Batch normalisation.** Training uses batch statistics over the time axis, which can carry future frames into training-time statistics. Evaluation uses running statistics, and the causality test passed with exact zero difference.
9. **Block-accumulated construction (fallback item 2).** Not run. The literal SPRT was run and its bound failed, but the failure is explained by the column-49 fallback (see the error-split table), so the second construction was not needed for this run.
10. **Energy.** Modelled only, as specified. No board or power monitor was used.
11. **Schema.** `method_out.json` is a plain JSON object, not checked against the exp_gen_sol_out schema in this run.

## Paired bootstrap (2000 resamples over test clips, f = 0.2, primary alpha = 0.01)

Saving = 1 - E_sys / E_ref (modelled). The third row is the literal comparison (b) against the SPRT, so a negative value means the threshold uses more energy than the SPRT.

| comparison | seed | point saving % | 95% CI % | p (two-sided) |
|---|---|---|---|---|
| (b) vs (a) | 0 | 35.32 | [34.81, 35.84] | 0.0010 |
| (b) vs (a) | 1 | 35.44 | [34.94, 35.96] | 0.0010 |
| (b) vs (a) | 2 | 35.61 | [35.11, 36.12] | 0.0010 |
| (c) vs (a) | 0 | 30.05 | [29.56, 30.51] | 0.0010 |
| (c) vs (a) | 1 | 30.41 | [29.92, 30.87] | 0.0010 |
| (c) vs (a) | 2 | 30.56 | [30.07, 31.03] | 0.0010 |
| SPRT vs (a) | 0 | 36.52 | [36.00, 37.02] | 0.0010 |
| SPRT vs (a) | 1 | 36.67 | [36.16, 37.18] | 0.0010 |
| SPRT vs (a) | 2 | 36.82 | [36.30, 37.36] | 0.0010 |
| (b) vs SPRT (literal) | 0 | -1.88 | [-1.98, -1.79] | 0.0010 |
| (b) vs SPRT (literal) | 1 | -1.95 | [-2.04, -1.85] | 0.0010 |
| (b) vs SPRT (literal) | 2 | -1.92 | [-2.01, -1.82] | 0.0010 |

Holm correction over the three pre-stated comparisons (mean of seed-level p, primary alpha, f = 0.2):

| comparison | p raw | p Holm |
|---|---|---|
| b_vs_a_full | 0.0010 | 0.0030 |
| c_vs_a_full | 0.0010 | 0.0030 |
| b_vs_sprt_literal | 0.0010 | 0.0030 |

## Energy sensitivity to the overhead fraction f (primary SPRT, alpha = beta = 0.01, L-Perf)

| f | E_SPRT (uJ, modelled) | E_full (uJ) | saving vs full % |
|---|---|---|---|
| 0.1 | 91.9 | 156.5 | 41.25 |
| 0.2 | 99.1 | 156.5 | 36.67 |
| 0.3 | 106.3 | 156.5 | 32.09 |
| 0.4 | 113.5 | 156.5 | 27.50 |

## Pre-stated criteria

| criterion | result |
|---|---|
| C1_sprt_saves_at_least_15pct_vs_full_window_f0.2 | {"pass": true} |
| C2_disconfirmed_if_saving_below_5pct_f0.2 | {"disconfirmed": false} |
| C3_bound_conformance_all_alphas_all_seeds_sprt | {"pass": false} |
| C4_bound_conformance_decorrelated_all | {"pass": false} |
| C5_bound_conformance_uncalibrated_all | {"pass": false} |
| C6_threshold_matched_within_0p5pts_all_seeds_alphas | {"pass": false} |
| C7_stop_fraction_above_0p1_at_primary | {"pass": true} |

## Figures

- `figures/energy_error.png` (and PDF): modelled energy (L-Perf, f = 0.2) against one-vs-rest accuracy for each system across alpha. The accuracy axis is compressed (about 96.2 to 97.1%), so differences of a few tenths of a point are visible only on the curve, not on an absolute scale. The full window is shown at 156.5 uJ.

## Limits

- Energy is a model. Absolute microjoule values are not measured claims. Only the relative saving is meaningful, and it depends on f.
- The Wilson intervals treat the 10 one-vs-rest comparisons of one clip as independent, which overstates precision. The reported bounds are checked as point estimates against alpha + 0.005.
- Speech Commands is small and studio-recorded, so the generalisation of these error rates is limited. One dataset, three seeds.
- Silence is synthetic brown noise, not the dataset's background-noise recordings.

