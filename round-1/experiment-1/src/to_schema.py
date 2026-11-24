#!/usr/bin/env python3
"""Convert results/method_detailed.json into the exp_gen_sol_out format -> method_out.json.

One example per (seed, alpha). input/output are strings; predict_<system> strings give the per-system
measured error and modelled energy; metadata_* keeps the full numbers. Numbers come from method_detailed.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEMS = ["a_full_window", "sprt_calibrated", "b_confidence_threshold", "b_oracle_threshold_test_tuned",
           "c_decorrelated_sprt", "d_uncalibrated_sprt"]


def pred(sysv: dict) -> str:
    if "FA" not in sysv:
        return sysv.get("note", "n/a")
    e = sysv["modelled_energy_uJ"]["L-Perf"]["0.2"]
    return (f"FA={100*sysv['FA']:.2f}% FR={100*sysv['FR']:.2f}% mean_prefix_columns={sysv['mean_m_columns']:.2f} "
            f"stop_before_49={100*sysv['stop_fraction_before_49']:.1f}% modelled_energy_LPerf_f0.2_uJ={e:.1f} "
            f"bound_pass={sysv['bound_pass']}")


def main() -> None:
    d = json.loads((ROOT / "results" / "method_detailed.json").read_text())
    examples = []
    for s in d["per_seed"]:
        for a, block in s["alphas"].items():
            ex = {
                "input": (f"Speech Commands v2 test split, 10 keywords one-vs-rest. Model seed {s['seed']}, "
                          f"target alpha = beta = {a}. Compare full window, calibrated SPRT, matched confidence "
                          f"threshold, oracle threshold, decorrelated SPRT and uncalibrated SPRT."),
                "output": (f"Calibrated SPRT: {pred(block['systems']['sprt_calibrated'])}. Full window: "
                           f"{pred(block['systems']['a_full_window'])}. Energy is MODELLED, not measured."),
                "metadata_seed": s["seed"],
                "metadata_alpha_beta": float(a),
                "metadata_decorrelated_stride_frames": s["decorrelated_stride"],
                "metadata_systems": {k: block["systems"][k] for k in SYSTEMS if k in block["systems"]},
                "metadata_bootstrap_f02": {k: v["0.2"] for k, v in block["bootstrap"].items()},
            }
            for k in SYSTEMS:
                if k in block["systems"]:
                    ex[f"predict_{k}"] = pred(block["systems"][k])
            examples.append(ex)
    out = {
        "metadata": {
            "method_name": "Causal DS-CNN with calibrated prefix SPRT early stopping vs full window and matched confidence threshold",
            "description": ("Early-stop keyword spotting on Speech Commands v2 (HF mirror danjacobellis/speech_commands_v2). "
                            "Energy is MODELLED (E = f*E_full + (1-f)*E_full*m/49, E_full from arXiv 2505.15622). "
                            "Full per-seed results: results/method_detailed.json. Report: method_report.md."),
            "seeds": d["seeds_run"],
            "pre_stated_criteria": d["pre_stated_criteria"],
            "holm_primary_alpha_f0.2": d["holm_primary_alpha_f0.2"],
            "summary_by_alpha_mean_over_seeds": d["summary_by_alpha_mean_over_seeds"],
        },
        "datasets": [{"dataset": "speech_commands_v2_test_one_vs_rest", "examples": examples}],
    }
    (ROOT / "method_out.json").write_text(json.dumps(out, indent=1))
    print(f"wrote method_out.json with {len(examples)} examples")


if __name__ == "__main__":
    main()
