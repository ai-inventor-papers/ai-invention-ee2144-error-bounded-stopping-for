#!/usr/bin/env python3
"""Early-stop keyword spotting energy test.

Subcommands
  prep            decode Speech Commands v2 (HF mirror), build 12-class index, MFCC 49x10 features
  units           unit checks: framing, causality, Wald boundaries, toy SPRT under independence
  train --seed S  train the causal DS-CNN for one seed (use --smoke for the 1,000-clip check)
  evaluate --seed S  calibration, stopping rules (SPRT, threshold, decorrelated, uncalibrated),
                  error with Wilson CIs, modelled energy, paired bootstrap
  aggregate       combine per-seed results into method_out.json and method_report.md
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.fft import dct
from scipy.signal import lfilter
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data_raw"
FEAT = ROOT / "features"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

KEYWORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
KW2I = {w: i for i, w in enumerate(KEYWORDS)}
SILENCE, UNKNOWN = 10, 11
N_CLS = 12
SR = 16000
CLIP = SR
WIN, HOP, NFFT = 640, 320, 1024
N_T = 49  # input columns (frames) per clip
N_MEL, N_MFCC = 40, 10
ALPHAS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05]
PRIMARY_ALPHA = 0.01
PRIMARY_F = 0.2
FS = [0.1, 0.2, 0.3, 0.4]
E_FULL = {"L-Perf": 156.5, "H-Perf": 219.0}  # uJ, arXiv 2505.15622 (sourced constant, modelled use)
TAUS = np.round(np.linspace(0.5, 0.999, 500), 5)
N_BOOT = 2000
SEEDS = [0, 1, 2]
SPLIT_SEED = {"train": 11, "validation": 22, "test": 33}
SPLIT_FILE = {"train": "train_part_*.parquet", "validation": "validation.parquet", "test": "test.parquet"}  # train is sharded (see shard_train.py)
# frame index j(t) of the causal output that uses exactly the first t columns (1-based t)
J_OF_T = np.array([(t - 1) // 2 for t in range(1, N_T + 1)])  # length 49, values 0..24
N_OUT_FRAMES = 25


def setup_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(LOGS / "run.log", rotation="30 MB", level="DEBUG")


# ----------------------------------------------------------------------------- features
def hann(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)  # periodic Hann


def mel_filterbank(n_mels: int = N_MEL, n_fft: int = NFFT, sr: int = SR,
                   fmin: float = 20.0, fmax: float = 4000.0) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (bins - lo) / (c - lo)
        down = (hi - bins) / (hi - c)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    if not np.all(fb.sum(axis=1) > 0):
        raise RuntimeError("empty mel filter; increase NFFT")
    return fb


FB = mel_filterbank()


def mfcc_batch(x: np.ndarray) -> np.ndarray:
    """x: (B, 16000) float. Returns (B, 49, 10) float32 MFCC: 640 window, 320 hop -> 49 frames."""
    if x.ndim != 2 or x.shape[1] != CLIP:
        raise ValueError(f"expected (B,{CLIP}) got {x.shape}")
    idx = np.arange(WIN)[None, :] + HOP * np.arange(N_T)[:, None]  # (49, 640)
    frames = x[:, idx] * hann(WIN)  # (B, 49, 640)
    spec = np.abs(np.fft.rfft(frames, n=NFFT, axis=-1)) ** 2
    mel = spec @ FB.T
    logmel = np.log(mel + 1e-6)
    mf = dct(logmel, type=2, norm="ortho", axis=-1)[..., :N_MFCC]
    return mf.astype(np.float32)


def fit_clip(x: np.ndarray) -> np.ndarray:
    """Pad with zeros or trim to exactly 16000 samples."""
    out = np.zeros(CLIP, dtype=np.float64)
    n = min(len(x), CLIP)
    out[:n] = x[:n]
    return out


def make_silence(rng: np.random.Generator, n: int) -> np.ndarray:
    """Synthetic silence: brown noise at a random RMS in [0.001, 0.05] (background-noise files unavailable)."""
    out = np.zeros((n, CLIP), dtype=np.float64)
    for i in range(n):
        w = rng.standard_normal(CLIP)
        b = lfilter([1.0], [1.0, -0.995], w)
        b = b - b.mean()
        rms = rng.uniform(0.001, 0.05)
        out[i] = b / (np.sqrt(np.mean(b ** 2)) + 1e-12) * rms
    return out


# ----------------------------------------------------------------------------- data prep
@logger.catch(reraise=True)
def prep() -> None:
    import pyarrow.parquet as pq
    import soundfile as sf

    FEAT.mkdir(parents=True, exist_ok=True)
    norm_train = None
    split_stats = {}
    all_paths = {}
    for split in ["train", "validation", "test"]:
        files = sorted(DATA_RAW.glob(SPLIT_FILE[split]))
        if not files:
            raise FileNotFoundError(f"no parquet files for {split} matching {SPLIT_FILE[split]} in {DATA_RAW}")
        logger.info(f"[{split}] reading {[f.name for f in files]}")
        import pyarrow as pa
        tbl = pa.concat_tables([pq.read_table(f, columns=["path", "opus"]) for f in files])
        paths = tbl.column("path").to_pylist()
        opus = tbl.column("opus").to_pylist()
        del tbl
        words = [p.split("/")[1] for p in paths]
        n_total = len(paths)
        kw_idx = [i for i, w in enumerate(words) if w in KW2I]
        unk_idx = [i for i, w in enumerate(words) if w not in KW2I]
        n_special = round(len(kw_idx) / 8)  # silence and unknown each ~10% of the 12-class set
        rng = np.random.default_rng(SPLIT_SEED[split])
        unk_keep = sorted(rng.choice(unk_idx, size=n_special, replace=False).tolist())
        sel = sorted(kw_idx + unk_keep)
        logger.info(f"[{split}] total={n_total} keyword={len(kw_idx)} unknown_pool={len(unk_idx)} "
                    f"unknown_kept={n_special} silence_generated={n_special}")
        feats = np.zeros((len(sel), N_T, N_MFCC), dtype=np.float32)
        labels = np.zeros(len(sel), dtype=np.int64)
        index = []
        chunk = 1000
        for c0 in range(0, len(sel), chunk):
            ids = sel[c0:c0 + chunk]
            wav = np.zeros((len(ids), CLIP), dtype=np.float64)
            for r, i in enumerate(ids):
                try:
                    x, sr = sf.read(io.BytesIO(opus[i]["bytes"]), dtype="float64")
                except (RuntimeError, ValueError, KeyError) as e:
                    logger.error(f"decode failed for {paths[i]}: {e}")
                    raise
                if sr != SR:
                    raise RuntimeError(f"unexpected sample rate {sr}")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                wav[r] = fit_clip(x)
            feats[c0:c0 + len(ids)] = mfcc_batch(wav)
            del wav
        for i in sel:
            w = words[i]
            lab = KW2I.get(w, UNKNOWN)
            index.append({"path": paths[i], "word": w, "label": lab, "source": "speech_commands_v2"})
        labels[:] = [d["label"] for d in index]
        del opus
        gc.collect()
        # silence
        rng_s = np.random.default_rng(SPLIT_SEED[split] + 1000)
        sil = make_silence(rng_s, n_special)
        sil_feats = mfcc_batch(sil)
        del sil
        feats = np.concatenate([feats, sil_feats])
        labels = np.concatenate([labels, np.full(n_special, SILENCE, dtype=np.int64)])
        index += [{"path": f"synthetic_silence/{split}/{k}", "word": "_silence_", "label": SILENCE,
                   "source": "synthetic brown noise (see README)"} for k in range(n_special)]
        all_paths[split] = set(paths[i] for i in sel)
        split_stats[split] = {"full_split_clips": n_total, "keyword": len(kw_idx),
                              "unknown": n_special, "silence": n_special,
                              "total_in_12class_set": int(len(labels)),
                              "silence_share": round(n_special / len(labels), 4),
                              "unknown_share": round(n_special / len(labels), 4)}
        if split == "train":
            norm_train = (feats.reshape(-1, N_MFCC).mean(0), feats.reshape(-1, N_MFCC).std(0) + 1e-6)
            np.save(FEAT / "norm_mean.npy", norm_train[0])
            np.save(FEAT / "norm_std.npy", norm_train[1])
        feats = (feats - norm_train[0].astype(np.float32)) / norm_train[1].astype(np.float32)
        np.save(FEAT / f"{split}_x.npy", feats.astype(np.float32))
        np.save(FEAT / f"{split}_y.npy", labels)
        (FEAT / f"{split}_index.json").write_text(json.dumps(index))
        logger.info(f"[{split}] saved features {feats.shape}, label counts "
                    f"{np.bincount(labels, minlength=N_CLS).tolist()}")
        del feats, labels, index
        gc.collect()
    overlap = {"train_val": len(all_paths["train"] & all_paths["validation"]),
               "train_test": len(all_paths["train"] & all_paths["test"]),
               "val_test": len(all_paths["validation"] & all_paths["test"])}
    logger.info(f"path overlap between splits: {overlap}")
    if any(overlap.values()):
        raise RuntimeError("split overlap detected")
    (FEAT / "split_stats.json").write_text(json.dumps({"splits": split_stats, "overlap": overlap}, indent=2))
    logger.info("prep done")


# ----------------------------------------------------------------------------- model
def get_torch():
    import torch
    return torch


def build_model():
    torch = get_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalDSCNN(nn.Module):
        """MLPerf Tiny DS-CNN layout with a causal time axis (left-only padding) and a causal
        running mean in place of the global average pool. Output frame j uses input columns 0..2j."""

        def __init__(self, ch: int = 64, n_blocks: int = 4):
            super().__init__()
            self.conv0 = nn.Conv2d(1, ch, (10, 4), stride=(2, 2), bias=False)
            self.bn0 = nn.BatchNorm2d(ch)
            self.dw = nn.ModuleList()
            self.bn_dw = nn.ModuleList()
            self.pw = nn.ModuleList()
            self.bn_pw = nn.ModuleList()
            for _ in range(n_blocks):
                self.dw.append(nn.Conv2d(ch, ch, (3, 3), groups=ch, bias=False))
                self.bn_dw.append(nn.BatchNorm2d(ch))
                self.pw.append(nn.Conv2d(ch, ch, 1, bias=False))
                self.bn_pw.append(nn.BatchNorm2d(ch))
            self.head = nn.Linear(ch * 5, N_CLS)

        def forward(self, x):  # x: (B, 49, 10)
            h = x.unsqueeze(1)
            h = F.pad(h, (1, 1, 9, 0))  # time left-only (9), freq symmetric (1)
            h = F.relu(self.bn0(self.conv0(h)))  # (B, 64, 25, 5)
            for dw, bdw, pw, bpw in zip(self.dw, self.bn_dw, self.pw, self.bn_pw):
                h = F.pad(h, (1, 1, 2, 0))  # causal depthwise along time
                h = F.relu(bdw(dw(h)))
                h = F.relu(bpw(pw(h)))
            b, c, t, f = h.shape
            h = h.permute(0, 2, 1, 3).reshape(b, t, c * f)
            cs = torch.cumsum(h, dim=1)
            cnt = torch.arange(1, t + 1, device=h.device, dtype=h.dtype).view(1, t, 1)
            return self.head(cs / cnt)  # (B, 25, 12): output frame j = causal running mean up to j

    return CausalDSCNN()


def device():
    torch = get_torch()
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_split(split: str):
    X = np.load(FEAT / f"{split}_x.npy")
    y = np.load(FEAT / f"{split}_y.npy")
    return X, y


def causality_test(model, n: int = 8, seed: int = 0) -> float:
    """Perturb all columns >= t; the frame that uses only columns < t must not change."""
    torch = get_torch()
    model.eval()
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, N_T, N_MFCC, generator=g)
    worst = 0.0
    with torch.no_grad():
        base = model(X)
        for t in range(1, N_T + 1):
            Xp = X.clone()
            Xp[:, t:, :] += 10.0 * torch.randn(n, N_T - t, N_MFCC, generator=g)
            out = model(Xp)
            j = int(J_OF_T[t - 1])
            worst = max(worst, float((out[:, j] - base[:, j]).abs().max()))
    return worst


def eval_loss(model, X, y, bs: int = 1024) -> float:
    torch = get_torch()
    import torch.nn.functional as F
    model.eval()
    tot, n = 0.0, 0
    jj = torch.tensor(J_OF_T[4:], dtype=torch.long)
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s:s + bs]).to(device())
            yb = torch.from_numpy(y[s:s + bs]).to(device())
            lt = model(xb)[:, jj.to(device())]
            tgt = yb.unsqueeze(1).expand(-1, lt.shape[1])
            loss = F.cross_entropy(lt.reshape(-1, N_CLS), tgt.reshape(-1), reduction="sum")
            tot += float(loss)
            n += tgt.numel()
    return tot / n


@logger.catch(reraise=True)
def train(seed: int, epochs: int, smoke: bool) -> None:
    torch = get_torch()
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    dev = device()
    logger.info(f"seed={seed} device={dev} epochs={epochs} smoke={smoke}")
    X, y = load_split("train")
    Xv, yv = load_split("validation")
    rng = np.random.default_rng(seed)
    if smoke:
        sel = rng.choice(len(X), size=1000, replace=False)
        X, y = X[sel], y[sel]
    model = build_model().to(dev)
    model.train()
    jj = torch.tensor(J_OF_T[4:], dtype=torch.long).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 256
    steps = math.ceil(len(X) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    best, bad, patience = float("inf"), 0, 4
    ckpt = CKPT / (f"smoke_seed{seed}.pt" if smoke else f"seed{seed}.pt")
    CKPT.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(X))
        tot, nb = 0.0, 0
        for s in range(0, len(X), bs):
            idx = perm[s:s + bs]
            xb = torch.from_numpy(X[idx]).to(dev)
            yb = torch.from_numpy(y[idx]).to(dev)
            lt = model(xb)[:, jj]
            tgt = yb.unsqueeze(1).expand(-1, lt.shape[1])
            loss = F.cross_entropy(lt.reshape(-1, N_CLS), tgt.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        ep_time = time.time() - t0
        vloss = eval_loss(model, Xv, yv)
        history.append({"epoch": ep + 1, "train_loss": tot / nb, "val_loss": vloss, "epoch_seconds": ep_time})
        logger.info(f"epoch {ep + 1}/{epochs} train_loss={tot / nb:.4f} val_loss={vloss:.4f} "
                    f"time={ep_time:.1f}s")
        if vloss < best:
            best, bad = vloss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                logger.info("early stopping")
                break
    cerr = causality_test(model.cpu())
    logger.info(f"causality max |diff| = {cerr:.3e}")
    if cerr > 1e-4:
        raise RuntimeError("causality violated")
    (RESULTS / f"train_log_seed{seed}{'_smoke' if smoke else ''}.json").write_text(
        json.dumps({"seed": seed, "history": history, "best_val_loss": best,
                    "causality_max_abs_diff": cerr, "n_train": int(len(X))}, indent=2))
    logger.info(f"checkpoint {ckpt}")


def logits_all_t(model, X: np.ndarray, bs: int = 1024) -> np.ndarray:
    """Causal 12-way logits for every prefix length t=1..49: returns (N, 49, 12) float64."""
    torch = get_torch()
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            outs.append(model(torch.from_numpy(X[s:s + bs]).to(device())).cpu().numpy())
    L = np.concatenate(outs).astype(np.float64)  # (N, 25, 12)
    return L[:, J_OF_T, :]


# ----------------------------------------------------------------------------- calibration and llr
TGRID = np.unique(np.append(np.exp(np.linspace(np.log(0.2), np.log(5.0), 81)), 1.0))


def fit_temperatures(Lval: np.ndarray, yval: np.ndarray) -> np.ndarray:
    T = np.ones(N_T)
    for t in range(N_T):
        Z = Lval[:, t, :]
        nlls = []
        for Tg in TGRID:
            Zs = Z / Tg
            lse = logsumexp(Zs, axis=1)
            nlls.append(float(np.mean(lse - Zs[np.arange(len(yval)), yval])))
        T[t] = TGRID[int(np.argmin(nlls))]
    return T


def nll_per_t(L: np.ndarray, y: np.ndarray, T: np.ndarray) -> np.ndarray:
    out = np.zeros(N_T)
    for t in range(N_T):
        Zs = L[:, t, :] / T[t]
        out[t] = np.mean(logsumexp(Zs, axis=1) - Zs[np.arange(len(y)), y])
    return out


def keyword_llr(L: np.ndarray, T: np.ndarray, pi: np.ndarray):
    """Log-LR llr_{k,t} = logit(p_{k,t}) - logit(pi_k) and p_{k,t}, for the 10 keywords.
    L: (N,49,12). Returns llr (N,49,10) and p (N,49,10)."""
    Z = L / T[None, :, None]
    lse_all = logsumexp(Z, axis=2)
    logp = Z[:, :, :10] - lse_all[:, :, None]
    logp = np.minimum(logp, -1e-12)
    log1mp = np.log1p(-np.exp(logp))
    logodds = logp - log1mp
    logit_pi = np.log(pi / (1 - pi))
    return logodds - logit_pi[None, None, :], np.exp(logp)


# ----------------------------------------------------------------------------- stopping rules
def stop_rule(llr: np.ndarray, p: np.ndarray, rule: str, A: float = 0.0, B: float = 0.0,
              tau: float = 0.5, allowed: np.ndarray | None = None):
    """Returns (stop index 0-based (N,10), decision bool (N,10)).
    Decision at any stop (and at column 49) is llr >= 0, i.e. the same posterior-vs-prior rule for all systems."""
    N = llr.shape[0]
    if rule == "full":
        idx = np.full((N, 10), N_T - 1)
    else:
        if rule == "sprt":
            stop = (llr >= A) | (llr <= B)
        elif rule == "threshold":
            stop = np.maximum(p, 1.0 - p) >= tau
        else:
            raise ValueError(rule)
        if allowed is not None:
            stop = stop & allowed[None, :, None]
        any_ = stop.any(axis=1)
        first = np.argmax(stop, axis=1)
        idx = np.where(any_, first, N_T - 1)
    dec = np.take_along_axis(llr, idx[:, None, :], axis=1)[:, 0, :] >= 0
    return idx, dec


def errors(dec: np.ndarray, y: np.ndarray):
    truth = y[:, None] == np.arange(10)[None, :]
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    fa_k = int((dec & ~truth).sum())
    fr_k = int((~dec & truth).sum())
    return fa_k / n_neg, fr_k / n_pos, fa_k, n_neg, fr_k, n_pos


def wilson(k: int, n: int, z: float = 1.959964):
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, c - h), min(1.0, c + h)]


def allowed_mask(stride: int | None) -> np.ndarray | None:
    if stride is None:
        return None
    tt = np.arange(1, N_T + 1)
    return (tt % stride == 0) | (tt == N_T)


def frame_allowed_mask(stride_frames: int) -> np.ndarray:
    """Decision points for the decorrelated SPRT: columns where a new frame appears (odd t), thinned to every
    stride_frames-th frame, plus the full window t = 49."""
    tt = np.arange(1, N_T + 1)
    frame = (tt - 1) // 2
    return ((tt % 2 == 1) & (frame % stride_frames == 0)) | (tt == N_T)


def wald_bounds(alpha: float, beta: float):
    return math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))


def summarize(idx: np.ndarray, dec: np.ndarray, y: np.ndarray, alpha: float, beta: float) -> dict:
    fa, fr, fak, nneg, frk, npos = errors(dec, y)
    m_pairs = (idx + 1).astype(np.float64)
    clip_m = m_pairs.mean(axis=1)
    truth = y[:, None] == np.arange(10)[None, :]
    bnd = idx < N_T - 1  # decided by a stopping boundary before column 49
    fb = ~bnd            # decided at column 49 (fallback)
    split_err = {
        "FA_boundary_stops": float((dec & ~truth & bnd).sum() / max(1, (~truth & bnd).sum())),
        "FR_boundary_stops": float((~dec & truth & bnd).sum() / max(1, (truth & bnd).sum())),
        "FA_fallback_at_49": float((dec & ~truth & fb).sum() / max(1, (~truth & fb).sum())),
        "FR_fallback_at_49": float((~dec & truth & fb).sum() / max(1, (truth & fb).sum())),
        "share_fallback_at_49": float(fb.mean()),
    }
    return {
        "error_split_boundary_vs_fallback": split_err,
        "FA": fa, "FR": fr, "FA_ci95": wilson(fak, nneg), "FR_ci95": wilson(frk, npos),
        "FA_counts": [fak, nneg], "FR_counts": [frk, npos],
        "bound_pass": bool(fa <= alpha + 0.005 and fr <= beta + 0.005),
        "mean_m_columns": float(m_pairs.mean()),
        "stop_fraction_before_49": float((idx < N_T - 1).mean()),
        "modelled_energy_uJ": {
            ref: {str(f): float(E * (f + (1 - f) * m_pairs.mean() / N_T)) for f in FS}
            for ref, E in E_FULL.items()
        },
        "_clip_m": clip_m,
    }


def bootstrap_saving(m_sys: np.ndarray, m_ref: np.ndarray, rng: np.random.Generator) -> dict:
    """Paired bootstrap over test clips. Saving = 1 - E_sys / E_ref, per modelled f."""
    N = len(m_sys)
    out = {str(f): [] for f in FS}
    done = 0
    while done < N_BOOT:
        b = min(500, N_BOOT - done)
        idx = rng.integers(0, N, size=(b, N))
        ms = m_sys[idx].mean(axis=1)
        mr = m_ref[idx].mean(axis=1)
        for f in FS:
            es = f + (1 - f) * ms / N_T
            er = f + (1 - f) * mr / N_T
            out[str(f)].append(1 - es / er)
        done += b
    res = {}
    for f in FS:
        s = np.concatenate(out[str(f)])
        point = 1 - (f + (1 - f) * m_sys.mean() / N_T) / (f + (1 - f) * m_ref.mean() / N_T)
        n_le = int((s <= 0).sum())
        n_ge = int((s >= 0).sum())
        p_two = 2 * min(n_le + 1, n_ge + 1) / (len(s) + 1)  # add-one correction for finite resamples
        res[str(f)] = {"point": float(point), "ci95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))],
                       "p_two_sided": float(min(1.0, p_two))}
    return res


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


@logger.catch(reraise=True)
def evaluate(seed: int) -> None:
    torch = get_torch()
    RESULTS.mkdir(parents=True, exist_ok=True)
    model = build_model()
    model.load_state_dict(torch.load(CKPT / f"seed{seed}.pt", map_location="cpu"))
    model = model.to(device())
    model.eval()
    causal_err = causality_test(model.cpu(), seed=seed)
    model = model.to(device())
    logger.info(f"[seed {seed}] causality max |diff| = {causal_err:.2e}")
    ytr = np.load(FEAT / "train_y.npy")
    pi = np.bincount(ytr, minlength=N_CLS)[:10] / len(ytr)
    Xv, yv = load_split("validation")
    Xt, yt = load_split("test")
    Lv = logits_all_t(model, Xv)
    Lt = logits_all_t(model, Xt)
    del Xv, Xt
    gc.collect()

    # calibration (validation only)
    T_cal = fit_temperatures(Lv, yv)
    ones = np.ones(N_T)
    nll_raw_v = nll_per_t(Lv, yv, ones)
    nll_cal_v = nll_per_t(Lv, yv, T_cal)
    cal_ok = bool(np.all(nll_cal_v <= nll_raw_v + 1e-12))
    logger.info(f"[seed {seed}] calibration: calibrated NLL <= raw NLL at every t: {cal_ok}; "
                f"mean NLL raw {nll_raw_v.mean():.4f} cal {nll_cal_v.mean():.4f}; "
                f"T at t=49: {T_cal[-1]:.3f}")
    nll_raw_t = nll_per_t(Lt, yt, ones)
    nll_cal_t = nll_per_t(Lt, yt, T_cal)

    llr_cal_v, p_cal_v = keyword_llr(Lv, T_cal, pi)
    llr_raw_v, p_raw_v = keyword_llr(Lv, ones, pi)
    llr_cal_t, p_cal_t = keyword_llr(Lt, T_cal, pi)
    llr_raw_t, p_raw_t = keyword_llr(Lt, ones, pi)

    # full-window accuracy (one-vs-rest, t=49 decision)
    _, dec_full_v = stop_rule(llr_cal_v, p_cal_v, "full")
    full_acc_v = float(1 - errors(dec_full_v, yv)[0] / 2 - errors(dec_full_v, yv)[1] / 2)
    _, dec_full_t = stop_rule(llr_cal_t, p_cal_t, "full")
    fa_f, fr_f, *_ = errors(dec_full_t, yt)
    # 12-class multiclass accuracy of the full window (reference check)
    mc_acc_t = float((Lt[:, -1, :].argmax(1) == yt).mean())
    logger.info(f"[seed {seed}] full window test one-vs-rest mean error FA={fa_f:.4f} FR={fr_f:.4f}; "
                f"12-class accuracy at t=49 = {mc_acc_t:.4f}")

    # feasibility on validation at alpha=beta=0.01
    A0, B0 = wald_bounds(PRIMARY_ALPHA, PRIMARY_ALPHA)
    idx_fe, _ = stop_rule(llr_cal_v, p_cal_v, "sprt", A=A0, B=B0)
    stop_hist = np.bincount(idx_fe.ravel() + 1, minlength=N_T + 1)[1:].tolist()
    feas = {"alpha_beta": PRIMARY_ALPHA, "mean_prefix_columns_val": float((idx_fe + 1).mean()),
            "median_prefix_columns_val": float(np.median(idx_fe + 1)),
            "stop_fraction_before_49_val": float((idx_fe < N_T - 1).mean()),
            "stop_index_histogram_1_to_49": stop_hist,
            "flag_mean_gt_40": bool((idx_fe + 1).mean() > 40),
            "flag_mean_gt_45": bool((idx_fe + 1).mean() > 45)}
    logger.info(f"[seed {seed}] feasibility: mean prefix {feas['mean_prefix_columns_val']:.2f} columns, "
                f"stop fraction {feas['stop_fraction_before_49_val']:.3f}")

    # decorrelated stride: lag autocorrelation of llr increments on validation.
    # Output frame j first becomes available at column t = 2j+1 (1-based), so even columns repeat
    # the previous frame and their increments are exactly zero. The autocorrelation is therefore
    # computed on the frame-level sequence (odd columns t = 1, 3, ..., 49), and the stride is in frames.
    naive_d = np.diff(llr_cal_v, axis=1)
    naive_autoc = {}
    for lag in range(1, 4):
        naive_autoc[lag] = float(np.corrcoef(naive_d[:, :-lag, :].ravel(), naive_d[:, lag:, :].ravel())[0, 1])
    frame_llr = llr_cal_v[:, 0::2, :]  # t = 1,3,...,49 -> 25 frame-level points
    d = np.diff(frame_llr, axis=1)  # (N,24,10)
    autoc = {}
    stride = None
    for lag in range(1, d.shape[1]):
        a = d[:, :-lag, :].ravel()
        b = d[:, lag:, :].ravel()
        autoc[lag] = float(np.corrcoef(a, b)[0, 1])
        if stride is None and autoc[lag] < 0.1:
            stride = lag
    if stride is None:
        stride = int(min(autoc, key=autoc.get))
        logger.warning(f"no frame lag with autocorrelation < 0.1; using frame lag {stride} (minimum)")
    logger.info(f"[seed {seed}] decorrelated stride s={stride} frames (frame lag-1 autocorr {autoc[1]:.3f}; "
                f"column-level lag-1 autocorr {naive_autoc[1]:.3f})")
    allow = frame_allowed_mask(stride)
    rng_boot = np.random.default_rng(1000 + seed)

    per_alpha = {}
    for alpha in ALPHAS:
        beta = alpha
        A, B = wald_bounds(alpha, beta)
        systems = {}
        # (a) full window
        idx_t, dec_t = stop_rule(llr_cal_t, p_cal_t, "full")
        systems["a_full_window"] = summarize(idx_t, dec_t, yt, alpha, beta)
        # SPRT primary (calibrated)
        idx_sv, dec_sv = stop_rule(llr_cal_v, p_cal_v, "sprt", A=A, B=B)
        fa_sv, fr_sv, *_ = errors(dec_sv, yv)
        idx_st, dec_st = stop_rule(llr_cal_t, p_cal_t, "sprt", A=A, B=B)
        systems["sprt_calibrated"] = summarize(idx_st, dec_st, yt, alpha, beta)
        systems["sprt_calibrated"]["val_FA"] = fa_sv
        systems["sprt_calibrated"]["val_FR"] = fr_sv
        # (b) confidence threshold, tuned on validation to match the SPRT validation error
        best = None
        for tau in TAUS:
            iv, dv = stop_rule(llr_cal_v, p_cal_v, "threshold", tau=tau)
            fav, frv, *_ = errors(dv, yv)
            gap = abs(fav - fa_sv) + abs(frv - fr_sv)
            if best is None or gap < best[0]:
                best = (gap, float(tau), fav, frv)
        _, tau_m, fav_m, frv_m = best
        matched = abs(fav_m - fa_sv) <= 0.005 and abs(frv_m - fr_sv) <= 0.005
        idx_bt, dec_bt = stop_rule(llr_cal_t, p_cal_t, "threshold", tau=tau_m)
        systems["b_confidence_threshold"] = summarize(idx_bt, dec_bt, yt, alpha, beta)
        systems["b_confidence_threshold"].update({"tau": tau_m, "val_FA": fav_m, "val_FR": frv_m,
                                                   "matched_within_0p5_points_on_val": bool(matched)})
        # oracle tau (tuned on test): least mean prefix with test error no worse than the SPRT's
        oracle = None
        for tau in TAUS:
            it, dt = stop_rule(llr_cal_t, p_cal_t, "threshold", tau=tau)
            fat, frt, *_ = errors(dt, yt)
            if fat <= systems["sprt_calibrated"]["FA"] and frt <= systems["sprt_calibrated"]["FR"]:
                m = float((it + 1).mean())
                if oracle is None or m < oracle[0]:
                    oracle = (m, float(tau))
        if oracle is not None:
            io_, do_ = stop_rule(llr_cal_t, p_cal_t, "threshold", tau=oracle[1])
            systems["b_oracle_threshold_test_tuned"] = summarize(io_, do_, yt, alpha, beta)
            systems["b_oracle_threshold_test_tuned"]["tau"] = oracle[1]
        else:
            systems["b_oracle_threshold_test_tuned"] = {"note": "no tau gives test error no worse than SPRT"}
        # (c) decorrelated SPRT
        idx_ct, dec_ct = stop_rule(llr_cal_t, p_cal_t, "sprt", A=A, B=B, allowed=allow)
        systems["c_decorrelated_sprt"] = summarize(idx_ct, dec_ct, yt, alpha, beta)
        systems["c_decorrelated_sprt"]["stride"] = stride
        # (d) uncalibrated SPRT
        idx_dt, dec_dt = stop_rule(llr_raw_t, p_raw_t, "sprt", A=A, B=B)
        systems["d_uncalibrated_sprt"] = summarize(idx_dt, dec_dt, yt, alpha, beta)
        # paired bootstrap over test clips
        m_full = systems["a_full_window"]["_clip_m"]
        boot = {
            "b_vs_a_full": bootstrap_saving(systems["b_confidence_threshold"]["_clip_m"], m_full, rng_boot),
            "c_vs_a_full": bootstrap_saving(systems["c_decorrelated_sprt"]["_clip_m"], m_full, rng_boot),
            "b_vs_sprt_literal": bootstrap_saving(systems["b_confidence_threshold"]["_clip_m"],
                                                  systems["sprt_calibrated"]["_clip_m"], rng_boot),
            "sprt_vs_a_full": bootstrap_saving(systems["sprt_calibrated"]["_clip_m"], m_full, rng_boot),
        }
        for s in systems.values():
            s.pop("_clip_m", None)
        per_alpha[str(alpha)] = {"systems": systems, "bootstrap": boot, "A": A, "B": B}
        logger.info(f"[seed {seed}] alpha=beta={alpha}: SPRT FA={systems['sprt_calibrated']['FA']:.4f} "
                    f"FR={systems['sprt_calibrated']['FR']:.4f} m={systems['sprt_calibrated']['mean_m_columns']:.2f} | "
                    f"threshold tau={tau_m:.3f} FA={systems['b_confidence_threshold']['FA']:.4f} "
                    f"FR={systems['b_confidence_threshold']['FR']:.4f} "
                    f"m={systems['b_confidence_threshold']['mean_m_columns']:.2f} | "
                    f"decorr m={systems['c_decorrelated_sprt']['mean_m_columns']:.2f} | "
                    f"uncal m={systems['d_uncalibrated_sprt']['mean_m_columns']:.2f} "
                    f"FA={systems['d_uncalibrated_sprt']['FA']:.4f} FR={systems['d_uncalibrated_sprt']['FR']:.4f}")

    out = {
        "seed": seed,
        "causality_max_abs_diff": causal_err,
        "priors_train": pi.tolist(),
        "temperatures_T_t": T_cal.tolist(),
        "nll_raw_val_mean": float(nll_raw_v.mean()), "nll_cal_val_mean": float(nll_cal_v.mean()),
        "nll_raw_test_mean": float(nll_raw_t.mean()), "nll_cal_test_mean": float(nll_cal_t.mean()),
        "calibrated_nll_le_raw_every_t_val": cal_ok,
        "full_window_one_vs_rest": {"test_FA": fa_f, "test_FR": fr_f, "test_12class_acc_t49": mc_acc_t,
                                    "test_one_vs_rest_accuracy": float(1 - (fa_f + fr_f) / 2)},
        "feasibility_val": feas,
        "decorrelated_stride": stride,
        "lag_autocorrelation_val": autoc,
        "alphas": per_alpha,
    }
    (RESULTS / f"seed{seed}.json").write_text(json.dumps(out, indent=2))
    logger.info(f"saved results/seed{seed}.json")


# ----------------------------------------------------------------------------- aggregate
@logger.catch(reraise=True)
def aggregate() -> None:
    seeds = []
    for s in SEEDS:
        p = RESULTS / f"seed{s}.json"
        if p.exists():
            seeds.append(json.loads(p.read_text()))
    if not seeds:
        raise RuntimeError("no per-seed results")
    split_stats = json.loads((FEAT / "split_stats.json").read_text())
    summary = {}
    for alpha in ALPHAS:
        a = str(alpha)
        summary[a] = {}
        for sysname in seeds[0]["alphas"][a]["systems"]:
            rows = [s["alphas"][a]["systems"][sysname] for s in seeds]
            if "FA" not in rows[0]:
                continue
            fa = [r["FA"] for r in rows]
            fr = [r["FR"] for r in rows]
            mm = [r["mean_m_columns"] for r in rows]
            summary[a][sysname] = {
                "FA_mean": float(np.mean(fa)), "FA_sd": float(np.std(fa, ddof=1)) if len(fa) > 1 else 0.0,
                "FR_mean": float(np.mean(fr)), "FR_sd": float(np.std(fr, ddof=1)) if len(fr) > 1 else 0.0,
                "mean_m_mean": float(np.mean(mm)),
                "mean_m_sd": float(np.std(mm, ddof=1)) if len(mm) > 1 else 0.0,
                "stop_fraction_mean": float(np.mean([r["stop_fraction_before_49"] for r in rows])),
                "bound_pass_all_seeds": all(r["bound_pass"] for r in rows),
                "modelled_energy_uJ_L_Perf_f0.2_mean": float(np.mean(
                    [r["modelled_energy_uJ"]["L-Perf"]["0.2"] for r in rows])),
            }
    # Holm over the three pre-stated comparisons at primary alpha and f=0.2
    pa = str(PRIMARY_ALPHA)
    keys = ["b_vs_a_full", "c_vs_a_full", "b_vs_sprt_literal"]
    pvals = []
    for k in keys:
        pvals.append(float(np.mean([s["alphas"][pa]["bootstrap"][k][str(PRIMARY_F)]["p_two_sided"] for s in seeds])))
    holm_adj = holm(pvals)
    holm_table = {k: {"p_raw_mean_over_seeds": p, "p_holm": h} for k, p, h in zip(keys, pvals, holm_adj)}
    # criteria at primary alpha, f=0.2
    sprt_save = [s["alphas"][pa]["bootstrap"]["sprt_vs_a_full"][str(PRIMARY_F)]["point"] for s in seeds]
    crit = {
        "C1_sprt_saves_at_least_15pct_vs_full_window_f0.2": {
            "per_seed_point": sprt_save, "pass": bool(np.mean(sprt_save) >= 0.15)},
        "C2_disconfirmed_if_saving_below_5pct_f0.2": {
            "per_seed_point": sprt_save, "disconfirmed": bool(np.mean(sprt_save) < 0.05)},
        "C3_bound_conformance_all_alphas_all_seeds_sprt": {
            "pass": all(s["alphas"][str(a)]["systems"]["sprt_calibrated"]["bound_pass"] for s in seeds for a in ALPHAS)},
        "C4_bound_conformance_decorrelated_all": {
            "pass": all(s["alphas"][str(a)]["systems"]["c_decorrelated_sprt"]["bound_pass"] for s in seeds for a in ALPHAS)},
        "C5_bound_conformance_uncalibrated_all": {
            "pass": all(s["alphas"][str(a)]["systems"]["d_uncalibrated_sprt"]["bound_pass"] for s in seeds for a in ALPHAS)},
        "C6_threshold_matched_within_0p5pts_all_seeds_alphas": {
            "pass": all(s["alphas"][str(a)]["systems"]["b_confidence_threshold"]["matched_within_0p5_points_on_val"]
                        for s in seeds for a in ALPHAS)},
        "C7_stop_fraction_above_0p1_at_primary": {
            "per_seed": [s["alphas"][pa]["systems"]["sprt_calibrated"]["stop_fraction_before_49"] for s in seeds],
            "pass": all(s["alphas"][pa]["systems"]["sprt_calibrated"]["stop_fraction_before_49"] > 0.1 for s in seeds)},
    }
    out = {
        "seeds_run": [s["seed"] for s in seeds],
        "split_stats": split_stats,
        "per_seed": [{k: v for k, v in s.items() if k != "alphas"} | {"alphas": s["alphas"]} for s in seeds],
        "summary_by_alpha_mean_over_seeds": summary,
        "holm_primary_alpha_f0.2": holm_table,
        "pre_stated_criteria": crit,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "method_detailed.json").write_text(json.dumps(out, indent=1))
    logger.info(f"wrote results/method_detailed.json ({len(seeds)} seeds); run to_schema.py for method_out.json")
    logger.info(json.dumps({"holm": holm_table, "criteria": crit}, indent=1)[:3000])


# ----------------------------------------------------------------------------- unit checks
@logger.catch(reraise=True)
def units() -> None:
    torch = get_torch()
    logger.info("U1 framing: 1 s clip -> frames")
    x = np.random.default_rng(0).standard_normal((2, CLIP))
    mf = mfcc_batch(x)
    assert mf.shape == (2, 49, 10), mf.shape
    logger.info(f"  shape {mf.shape} OK; frames = 1 + (16000-640)/320 = {1 + (CLIP - WIN) // HOP}")
    logger.info(f"U1b mel filterbank rows nonzero: {int((FB.sum(1) > 0).sum())}/{N_MEL}")

    logger.info("U2 causality of the causal DS-CNN (eval mode)")
    torch.manual_seed(0)
    m = build_model()
    m.eval()
    # randomise BN running stats so the test is non-trivial
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.running_var.uniform_(0.5, 1.5)
    err = causality_test(m)
    logger.info(f"  max |diff| over all prefixes = {err:.3e}")
    assert err < 1e-5, err

    logger.info("U3 Wald boundaries at alpha=beta=0.01")
    A, B = wald_bounds(0.01, 0.01)
    logger.info(f"  A = {A:.4f}, B = {B:.4f}")
    assert abs(A - 4.595) < 1e-3 and abs(B + 4.595) < 1e-3

    logger.info("U4 toy SPRT with independent Gaussian increments: measured error vs bound")
    rng = np.random.default_rng(7)
    n_rep, n_steps = 20000, 1000
    A_, B_ = wald_bounds(0.01, 0.01)
    # H0: x~N(0,1), H1: x~N(1,1); LLR increment = x - 0.5 (log N(x;1,1)/N(x;0,1))
    for hyp, mu in [("H0", 0.0), ("H1", 1.0)]:
        x = rng.normal(mu, 1.0, size=(n_rep, n_steps))
        S = np.cumsum(x - 0.5, axis=1)
        acc = (S >= A_) | (S <= B_)
        first = np.where(acc.any(1), np.argmax(acc, 1), n_steps - 1)
        Sf = S[np.arange(n_rep), first]
        dec = Sf >= 0
        err = dec.mean() if hyp == "H0" else (~dec).mean()
        logger.info(f"  {hyp}: error rate {err:.4f} (bound 0.01)")
        assert err <= 0.012, err
    logger.info("units passed")


# ----------------------------------------------------------------------------- main
def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("units")
    sub.add_parser("aggregate")
    p_tr = sub.add_parser("train")
    p_tr.add_argument("--seed", type=int, required=True)
    p_tr.add_argument("--epochs", type=int, default=30)
    p_tr.add_argument("--smoke", action="store_true")
    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    if args.cmd == "prep":
        prep()
    elif args.cmd == "units":
        units()
    elif args.cmd == "train":
        train(args.seed, args.epochs, args.smoke)
    elif args.cmd == "evaluate":
        evaluate(args.seed)
    elif args.cmd == "aggregate":
        aggregate()


if __name__ == "__main__":
    main()
