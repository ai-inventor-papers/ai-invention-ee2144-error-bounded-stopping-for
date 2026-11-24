#!/usr/bin/env python3
"""Independent re-derivation of the headline numbers (seed 0, alpha = beta = 0.01, f = 0.2, L-Perf).

Different code path from method.py: temperatures come from scipy's bounded scalar minimiser (not a grid),
keyword log-LRs are computed per clip and per prefix in plain Python loops from explicit softmax terms,
stopping and error counting are written out per clip, and the bootstrap placebo swaps the two systems'
per-clip prefix lengths at random. Only the trained network forward pass is shared (build_model / weights).
The independent numbers are compared with results/method_detailed.json; the placebo must NOT reach significance."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

import method as M

ROOT = Path(__file__).resolve().parent
A = math.log(0.99 / 0.01)
B = -A
F = 0.2


def logits(split: str) -> np.ndarray:
    X, _ = M.load_split(split)
    model = M.build_model()
    model.load_state_dict(torch.load(M.CKPT / "seed0.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X)).numpy().astype(np.float64)  # (N,25,12)
    return out[:, [(t - 1) // 2 for t in range(1, 50)], :]          # (N,49,12) causal per prefix


def nll(Lt: np.ndarray, y: np.ndarray, T: float) -> float:
    s = Lt / T
    s = s - s.max(axis=1, keepdims=True)
    lp = s - np.log(np.exp(s).sum(axis=1, keepdims=True))
    return float(-lp[np.arange(len(y)), y].mean())


def main() -> None:
    yv = np.load(M.FEAT / "validation_y.npy")
    yt = np.load(M.FEAT / "test_y.npy")
    ytr = np.load(M.FEAT / "train_y.npy")
    pi = np.array([(ytr == k).mean() for k in range(10)])
    Lv, Lt = logits("validation"), logits("test")
    # temperatures by bounded scalar minimisation on log T
    T = np.ones(49)
    for t in range(49):
        r = minimize_scalar(lambda u: nll(Lv[:, t, :], yv, math.exp(u)), bounds=(math.log(0.2), math.log(5)),
                            method="bounded", options={"xatol": 1e-4})
        T[t] = math.exp(r.x)
    # per clip, per keyword, per prefix in plain loops
    def run(Lmat: np.ndarray, y: np.ndarray, Tv: np.ndarray, rule: str):
        N = len(y)
        stop_idx = np.full((N, 10), 48)
        dec = np.zeros((N, 10), dtype=bool)
        for i in range(N):
            for k in range(10):
                done = False
                for t in range(49):
                    z = Lmat[i, t] / Tv[t]
                    z = z - z.max()
                    ez = np.exp(z)
                    p = ez[k] / ez.sum()
                    llr = math.log(p / (1 - p)) - math.log(pi[k] / (1 - pi[k]))
                    if rule == "sprt" and (llr >= A or llr <= B):
                        stop_idx[i, k] = t
                        dec[i, k] = llr >= 0
                        done = True
                        break
                if not done:
                    z = Lmat[i, 48] / Tv[48]
                    z = z - z.max()
                    ez = np.exp(z)
                    p = ez[k] / ez.sum()
                    llr = math.log(p / (1 - p)) - math.log(pi[k] / (1 - pi[k]))
                    dec[i, k] = llr >= 0
        return stop_idx, dec

    idx, dec = run(Lt, yt, T, "sprt")
    truth = np.stack([yt == k for k in range(10)], axis=1)
    fa = float((dec & ~truth).sum() / (~truth).sum())
    fr = float((~dec & truth).sum() / truth.sum())
    m = (idx + 1).astype(float)
    E = F + (1 - F) * m.mean() / 49
    full_E = F + (1 - F) * 1.0
    saving = 1 - E / full_E
    bnd = idx < 48
    fa_b = float((dec & ~truth & bnd).sum() / (~truth & bnd).sum())
    fa_f = float((dec & ~truth & ~bnd).sum() / (~truth & ~bnd).sum())

    det = json.loads((ROOT / "results" / "method_detailed.json").read_text())
    blk = det["per_seed"][0]["alphas"]["0.01"]["systems"]["sprt_calibrated"]
    print("                 independent      method.py (detailed)")
    print(f"FA               {fa:.5f}          {blk['FA']:.5f}")
    print(f"FR               {fr:.5f}          {blk['FR']:.5f}")
    print(f"mean m           {m.mean():.4f}          {blk['mean_m_columns']:.4f}")
    print(f"saving vs full   {saving:.5f}          {1 - (F + (1-F) * blk['mean_m_columns']/49):.5f}")
    print(f"FA boundary      {fa_b:.5f}          {blk['error_split_boundary_vs_fallback']['FA_boundary_stops']:.5f}")
    print(f"FA fallback 49   {fa_f:.5f}          {blk['error_split_boundary_vs_fallback']['FA_fallback_at_49']:.5f}")

    # placebo for the paired-bootstrap test of (b) vs SPRT: swap per-clip prefix lengths at random
    rng = np.random.default_rng(0)
    tau = det["per_seed"][0]["alphas"]["0.01"]["systems"]["b_confidence_threshold"]["tau"]
    # threshold stop at the reported tau, same loops
    def thr(Lmat, Tv):
        N = Lmat.shape[0]
        out = np.full((N, 10), 48)
        for i in range(N):
            for k in range(10):
                for t in range(49):
                    z = Lmat[i, t] / Tv[t]
                    z = z - z.max()
                    ez = np.exp(z)
                    p = ez[k] / ez.sum()
                    if max(p, 1 - p) >= tau:
                        out[i, k] = t
                        break
        return out
    idx_b = thr(Lt, T)
    mb = (idx_b + 1).astype(float).mean(axis=1)
    ms = m.mean(axis=1)
    diff = mb - ms
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)])
    p_real = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    sw = rng.random(len(mb)) < 0.5
    placebo = np.where(sw, ms, mb) - np.where(sw, mb, ms)  # randomly swapped clip-level labels
    pboot = np.array([placebo[rng.integers(0, len(placebo), len(placebo))].mean() for _ in range(2000)])
    p_plac = 2 * min((pboot <= 0).mean(), (pboot >= 0).mean())
    print(f"threshold tau (reported) {tau:.4f}: independent threshold mean m {mb.mean():.3f}, "
          f"SPRT mean m {ms.mean():.3f}")
    print(f"bootstrap p real (b - SPRT mean m) = {p_real:.4f}; placebo (random swap) p = {p_plac:.4f} "
          f"(must not be significant)")


if __name__ == "__main__":
    main()
