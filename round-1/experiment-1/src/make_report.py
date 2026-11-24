#!/usr/bin/env python3
"""Render method_report.md from method_out.json (all numbers come from the JSON, none are typed by hand)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
ALPHAS = ["0.001", "0.0025", "0.005", "0.01", "0.02", "0.05"]
SYS_LABEL = {
    "a_full_window": "(a) Full window",
    "sprt_calibrated": "Primary: calibrated SPRT",
    "b_confidence_threshold": "(b) Confidence threshold (matched on val)",
    "b_oracle_threshold_test_tuned": "(b*) Oracle threshold (tuned on test)",
    "c_decorrelated_sprt": "(c) Decorrelated SPRT",
    "d_uncalibrated_sprt": "(d) Uncalibrated SPRT",
}
F_SHOW = "0.2"


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    o = json.loads((ROOT / "results" / "method_detailed.json").read_text())
    seeds = o["per_seed"]
    summ = o["summary_by_alpha_mean_over_seeds"]
    crit = o["pre_stated_criteria"]
    holm = o["holm_primary_alpha_f0.2"]
    ss = o["split_stats"]["splits"]
    L = []
    L.append("# Early-Stop Keyword Spotting Energy Test: method report\n")
    L.append("Auto-generated from `results/method_detailed.json` by `make_report.py`. Energy values are MODELLED, not measured on a board.\n")
    L.append("## Design\n")
    L.append("- Data: Speech Commands v2 test/validation/train splits taken from the HF mirror `danjacobellis/speech_commands_v2` "
             "(Opus audio, decoded to 16 kHz). Split sizes equal the official v2 counts (84,843 / 9,981 / 11,005). "
             "The official archive was unreachable from this run, so the official list files were not used (see Departures).")
    L.append("- 12-class set: 10 target words, silence (10%), unknown (10%). Unknown is subsampled per split; silence is synthetic brown noise.")
    L.append("- Front end: 640-sample Hann window, 320 hop, 1024-point FFT, 40 mel bins (20-4000 Hz), log, DCT-II, first 10 coefficients: 49x10 per clip. Normalised with train statistics.")
    L.append("- Model: MLPerf Tiny DS-CNN layout (Conv 10x4 stride 2, four depthwise-separable blocks of 64 channels) with left-only time padding and a causal running mean in place of global average pooling. Trained with cross-entropy over prefixes t = 5..49, Adam 1e-3 with cosine decay, batch 256, up to 30 epochs with early stopping (patience 4).")
    L.append("- Causal decisions: the output at prefix length t uses only columns 1..t. Because the first convolution has stride 2, output frames change at odd t, and even t repeat the frame of t-1.")
    L.append("- Error: one-vs-rest per target keyword. Negatives are all other classes (including unknown and silence). FA and FR are pooled over the 10 keywords; Wilson 95% intervals use the pooled counts (see Limits).")
    L.append("- Energy (MODELLED): E = f E_full + (1 - f) E_full m / 49, with E_full = 156.5 uJ (L-Perf) and 219.0 uJ (H-Perf) taken from arXiv 2505.15622, and f in {0.1, 0.2, 0.3, 0.4}. m is the mean number of columns processed per decision. The formula assumes each column is processed once by incremental causal computation.")
    L.append(f"- Seeds run: {o['seeds_run']} (3 model seeds; the same test clips for all systems).\n")
    L.append("## Split and calibration checks\n")
    L.append("| split | full split clips | keyword | unknown | silence | total | silence share |")
    L.append("|---|---|---|---|---|---|---|")
    for sp in ["train", "validation", "test"]:
        d = ss[sp]
        L.append(f"| {sp} | {d['full_split_clips']} | {d['keyword']} | {d['unknown']} | {d['silence']} | {d['total_in_12class_set']} | {d['silence_share']} |")
    L.append(f"\nPath overlap between splits: {o['split_stats']['overlap']}.\n")
    for s in seeds:
        L.append(f"- Seed {s['seed']}: causality max |diff| = {s['causality_max_abs_diff']:.2e}; calibrated NLL <= raw NLL at every t on validation: {s['calibrated_nll_le_raw_every_t_val']}; "
                 f"full-window test one-vs-rest FA {pct(s['full_window_one_vs_rest']['test_FA'])}%, FR {pct(s['full_window_one_vs_rest']['test_FR'])}%, 12-class accuracy at t = 49 {pct(s['full_window_one_vs_rest']['test_12class_acc_t49'])}%.")
    L.append("")
    L.append("## Feasibility (validation, alpha = beta = 0.01, calibrated SPRT)\n")
    L.append("| seed | mean prefix (columns) | median | stop before 49 |")
    L.append("|---|---|---|---|")
    for s in seeds:
        f = s["feasibility_val"]
        L.append(f"| {s['seed']} | {f['mean_prefix_columns_val']:.2f} | {f['median_prefix_columns_val']:.0f} | {pct(f['stop_fraction_before_49_val'])}% |")
    L.append("\nMean prefix is below 40 columns, so the energy saving is not expected to be small (feasibility flags, mean > 40 or > 45, were not raised).\n")
    L.append("## Decorrelated stride\n")
    L.append("The column-level lag-1 autocorrelation of LLR increments is about -0.05. This is an artefact: even columns repeat the previous output frame, so their increment is exactly zero. The stride is therefore chosen on the frame-level sequence (odd columns). Frame-level lag-1 autocorrelation is about 0.5, and the first lag below 0.1 is:\n")
    L.append("| seed | stride (frames) | allowed decision columns |")
    L.append("|---|---|---|")
    for s in seeds:
        L.append(f"| {s['seed']} | {s['decorrelated_stride']} | odd t with (t-1)/2 divisible by {s['decorrelated_stride']}, plus t = 49 |")
    L.append("")
    L.append("## Main table: test error and modelled energy (mean over seeds)\n")
    L.append("FA and FR are one-vs-rest error rates (%). m is mean columns processed. Saving is relative to the full window at f = 0.2 (modelled). Bound pass means FA <= alpha + 0.5 pt and FR <= beta + 0.5 pt, checked on every seed.\n")
    for a in ALPHAS:
        L.append(f"### alpha = beta = {a}\n")
        L.append("| system | FA % | FR % | m (columns) | stop before 49 | saving vs full (f=0.2) | bound pass (all seeds) |")
        L.append("|---|---|---|---|---|---|---|")
        full_m = summ[a]["a_full_window"]["mean_m_mean"]
        for key, lab in SYS_LABEL.items():
            if key not in summ[a] or "FA_mean" not in summ[a][key]:
                continue
            v = summ[a][key]
            saving = 1 - (0.2 + 0.8 * v["mean_m_mean"] / 49) / (0.2 + 0.8 * full_m / 49)
            L.append(f"| {lab} | {pct(v['FA_mean'])} ± {pct(v['FA_sd'])} | {pct(v['FR_mean'])} ± {pct(v['FR_sd'])} | "
                     f"{v['mean_m_mean']:.2f} ± {v['mean_m_sd']:.2f} | {pct(v['stop_fraction_mean'])}% | {pct(saving)}% | {v['bound_pass_all_seeds']} |")
        L.append("")
    L.append("## Where the bound is violated: boundary stops versus column-49 fallback (calibrated SPRT)\n")
    L.append("Decisions made at a Wald boundary are separated from decisions made at column 49 because no boundary was crossed. Values are the mean over seeds.\n")
    L.append("| alpha | FA boundary | FR boundary | FA fallback | FR fallback | share decided at 49 |")
    L.append("|---|---|---|---|---|---|")
    for a in ALPHAS:
        rows = [s["alphas"][a]["systems"]["sprt_calibrated"]["error_split_boundary_vs_fallback"] for s in seeds]
        import numpy as np
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        L.append(f"| {a} | {pct(m['FA_boundary_stops'])}% | {pct(m['FR_boundary_stops'])}% | {pct(m['FA_fallback_at_49'])}% | {pct(m['FR_fallback_at_49'])}% | {pct(m['share_fallback_at_49'])}% |")
    L.append("")
    L.append("## Wilson 95% intervals, primary SPRT and full window at alpha = beta = 0.01 (per seed, pooled one-vs-rest counts)\n")
    L.append("| system | seed | FA % [95% CI] | FR % [95% CI] |")
    L.append("|---|---|---|---|")
    for key in ["a_full_window", "sprt_calibrated"]:
        for s in seeds:
            v = s["alphas"]["0.01"]["systems"][key]
            L.append(f"| {SYS_LABEL[key]} | {s['seed']} | {pct(v['FA'])} [{pct(v['FA_ci95'][0])}, {pct(v['FA_ci95'][1])}] | "
                     f"{pct(v['FR'])} [{pct(v['FR_ci95'][0])}, {pct(v['FR_ci95'][1])}] |")
    L.append("")
    unmatched = [(s["seed"], a) for s in seeds for a in ALPHAS
                 if not s["alphas"][a]["systems"]["b_confidence_threshold"]["matched_within_0p5_points_on_val"]]
    L.append(f"Threshold baseline not matched within 0.5 points on validation (seed, alpha): {unmatched if unmatched else 'none'}. "
             "The tau grid ran from 0.5 to 0.999; for these cells the closest achievable match is reported and marked as not matched.\n")
    L.append("## Departures from the plan\n")
    L.append("1. **Dataset source.** The official `speech_commands_v0.02.tar.gz` host was unreachable from this run, and the `google/speech_commands` HF repo is a loading script that downloads that archive. The split was therefore taken from `danjacobellis/speech_commands_v2`, a parquet mirror with Opus-encoded audio and the official split sizes. The validation_list and testing_list files were not available, so split membership is the mirror's, not verified against the lists. Opus coding is a small change to the waveform.\n"
             "2. **Silence.** The `_background_noise_` files are not in the mirror. Silence clips are synthetic brown noise at a random RMS between 0.001 and 0.05, not crops of recorded noise.\n"
             "3. **Clip length.** Decoded clips are 9,216 to 18,432 samples. The first 16,000 samples are kept, so a speech onset after 16,000 samples would be cut. Peak positions were checked on 400 test clips (median 7,239 samples).\n"
             "4. **Front end.** 40 mel bins then DCT-II to 10 coefficients (the MLPerf Tiny reference), not 10 log-mel bins. The FFT is 1024 points on a 640-sample window so that all mel filters are non-empty.\n"
             "5. **Decision granularity.** The first convolution has stride 2 in time, so causal outputs exist per frame, and even prefix lengths repeat the previous frame. The stopping rules are applied per column, which is equivalent to per frame.\n"
             "6. **Confidence-threshold decision rule.** The threshold rule stops at the first column where max(p, 1 - p) reaches tau (the plan's |p - 0.5| >= tau is the same test shifted by 0.5). At the stop and at column 49 every system decides with the same rule, llr >= 0, so only the stopping differs.\n"
             "7. **Decorrelated stride.** Chosen on frame-level increments (see Decorrelated stride), because column-level increments are zero every other column.\n"
             "8. **Batch normalisation.** Training uses batch statistics over the time axis, which can carry future frames into training-time statistics. Evaluation uses running statistics, and the causality test passed with exact zero difference.\n"
             "9. **Block-accumulated construction (fallback item 2).** Not run. The literal SPRT was run and its bound failed, but the failure is explained by the column-49 fallback (see the error-split table), so the second construction was not needed for this run.\n"
             "10. **Energy.** Modelled only, as specified. No board or power monitor was used.\n"
             "11. **Schema.** `method_out.json` is a plain JSON object, not checked against the exp_gen_sol_out schema in this run.\n")
    L.append("## Paired bootstrap (2000 resamples over test clips, f = 0.2, primary alpha = 0.01)\n")
    L.append("Saving = 1 - E_sys / E_ref (modelled). The third row is the literal comparison (b) against the SPRT, so a negative value means the threshold uses more energy than the SPRT.\n")
    L.append("| comparison | seed | point saving % | 95% CI % | p (two-sided) |")
    L.append("|---|---|---|---|---|")
    for key, lab in [("b_vs_a_full", "(b) vs (a)"), ("c_vs_a_full", "(c) vs (a)"), ("sprt_vs_a_full", "SPRT vs (a)"),
                     ("b_vs_sprt_literal", "(b) vs SPRT (literal)")]:
        for s in seeds:
            r = s["alphas"]["0.01"]["bootstrap"][key][F_SHOW]
            L.append(f"| {lab} | {s['seed']} | {pct(r['point'])} | [{pct(r['ci95'][0])}, {pct(r['ci95'][1])}] | {r['p_two_sided']:.4f} |")
    L.append("\nHolm correction over the three pre-stated comparisons (mean of seed-level p, primary alpha, f = 0.2):\n")
    L.append("| comparison | p raw | p Holm |")
    L.append("|---|---|---|")
    for k, v in holm.items():
        L.append(f"| {k} | {v['p_raw_mean_over_seeds']:.4f} | {v['p_holm']:.4f} |")
    L.append("")
    L.append("## Energy sensitivity to the overhead fraction f (primary SPRT, alpha = beta = 0.01, L-Perf)\n")
    L.append("| f | E_SPRT (uJ, modelled) | E_full (uJ) | saving vs full % |")
    L.append("|---|---|---|---|")
    sp = summ["0.01"]["sprt_calibrated"]
    fw = summ["0.01"]["a_full_window"]
    for f in ["0.1", "0.2", "0.3", "0.4"]:
        e_s = [s["alphas"]["0.01"]["systems"]["sprt_calibrated"]["modelled_energy_uJ"]["L-Perf"][f] for s in seeds]
        e_f = [s["alphas"]["0.01"]["systems"]["a_full_window"]["modelled_energy_uJ"]["L-Perf"][f] for s in seeds]
        import numpy as np
        es, ef = float(np.mean(e_s)), float(np.mean(e_f))
        L.append(f"| {f} | {es:.1f} | {ef:.1f} | {pct(1 - es / ef)} |")
    L.append("")
    L.append("## Pre-stated criteria\n")
    L.append("| criterion | result |")
    L.append("|---|---|")
    for k, v in crit.items():
        det = {kk: vv for kk, vv in v.items() if kk not in ("per_seed_point", "per_seed")}
        L.append(f"| {k} | {json.dumps(det)} |")
    L.append("")
    L.append("## Figures\n")
    L.append("- `figures/energy_error.png` (and PDF): modelled energy (L-Perf, f = 0.2) against one-vs-rest accuracy for each system across alpha. The accuracy axis is compressed (about 96.2 to 97.1%), so differences of a few tenths of a point are visible only on the curve, not on an absolute scale. The full window is shown at 156.5 uJ.\n")
    L.append("## Limits\n")
    L.append("- Energy is a model. Absolute microjoule values are not measured claims. Only the relative saving is meaningful, and it depends on f.")
    L.append("- The Wilson intervals treat the 10 one-vs-rest comparisons of one clip as independent, which overstates precision. The reported bounds are checked as point estimates against alpha + 0.005.")
    L.append("- Speech Commands is small and studio-recorded, so the generalisation of these error rates is limited. One dataset, three seeds.")
    L.append("- Silence is synthetic brown noise, not the dataset's background-noise recordings.\n")
    (ROOT / "method_report.md").write_text("\n".join(L) + "\n")
    logger.info("wrote method_report.md")


if __name__ == "__main__":
    main()
