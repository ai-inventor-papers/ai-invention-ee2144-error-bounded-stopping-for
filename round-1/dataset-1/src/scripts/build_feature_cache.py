#!/usr/bin/env python3
"""Build a frozen 49x10 keyword-spotting feature cache from Speech Commands v0.02.

Inputs (archives, downloaded beforehand, see README.md):
  temp/datasets/archives/speech_commands_v0.02.tar.gz
  temp/datasets/archives/speech_commands_test_set_v0.02.tar.gz

Outputs (under features/):
  features_<split>.npy        log-mel, float16, N x 49 x 10   (splits: train, validation, testing, test_bench)
  mfcc_features_<split>.npy   MFCC,     float16, N x 49 x 10
  index.json                  flat list, one record per clip (all listed fields)
  full_data_out.json          same records in the exp_sel_data_out projection (validated by aii-json)
  mini_index.json             first 200 rows per split
  preview_index.json          first 20 rows per split
  split_stats.json            counts per split and label, list line counts, seeds
Plus checksums.sha256 written by a separate step (see README.md).

Run: .venv/bin/python scripts/build_feature_cache.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import sys
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from loguru import logger

WORKSPACE = Path(__file__).resolve().parent.parent
ARCH_DIR = WORKSPACE / "temp/datasets/archives"
EXTRACT_DIR = WORKSPACE / "temp/datasets/extracted"
OUT_DIR = WORKSPACE / "features"
LOG_DIR = WORKSPACE / "logs"

V002_TAR = ARCH_DIR / "speech_commands_v0.02.tar.gz"
TEST_TAR = ARCH_DIR / "speech_commands_test_set_v0.02.tar.gz"
V002_DIR = EXTRACT_DIR / "v0.02"
TEST_DIR = EXTRACT_DIR / "test_set"

SR = 16000
CLIP_SAMPLES = 16000
WIN = 640          # 40 ms
HOP = 320          # 20 ms
N_FFT = 640  # equals the window; librosa frames at n_fft, so n_fft=1024 with center=False gives 47 frames, not 49
N_BINS = 10
N_FRAMES = 1 + (CLIP_SAMPLES - WIN) // HOP  # = 49 with center=False
LOG_FLOOR = 1e-6

TARGETS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SILENCE_LABEL = "_silence_"
UNKNOWN_LABEL = "_unknown_"
LABEL_INDEX = {w: i for i, w in enumerate(TARGETS)}
LABEL_INDEX[SILENCE_LABEL] = 10
LABEL_INDEX[UNKNOWN_LABEL] = 11
SEED_UNKNOWN = 42
SEED_SILENCE = 43
PCT = 0.10  # TF speech_commands recipe: unknown and silence = 10% of the known-word count per split
BG_DIR = "_background_noise_"


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_DIR / "build_feature_cache.log", rotation="30 MB", level="DEBUG")


def extract(tar_path: Path, dest: Path, marker: str) -> Path:
    """Extract a tar.gz once; a marker file records completion."""
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / marker
    if done.exists():
        logger.info(f"Already extracted: {dest}")
        return dest
    logger.info(f"Extracting {tar_path.name} -> {dest}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    done.write_text("ok\n")
    return dest


def read_list(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    logger.info(f"{path.name}: {len(lines)} lines")
    return lines


def compute_features(wav_path: str | None, samples: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (logmel, mfcc), each float16 of shape (49, 10). Pads or trims to 1 s first."""
    if samples is None:
        y, sr = sf.read(wav_path, dtype="float32")
        if sr != SR:
            raise ValueError(f"{wav_path}: sample rate {sr} != {SR}")
        if y.ndim > 1:
            y = y.mean(axis=1)
    else:
        y = samples.astype(np.float32)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    elif len(y) > CLIP_SAMPLES:
        y = y[:CLIP_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        window="hann", center=False, n_mels=N_BINS, power=2.0,
    )
    logmel = np.log(mel + LOG_FLOOR).T
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
        window="hann", center=False, n_mels=40, n_mfcc=N_BINS,
    ).T
    if logmel.shape != (N_FRAMES, N_BINS) or mfcc.shape != (N_FRAMES, N_BINS):
        raise AssertionError(f"shape mismatch {logmel.shape} {mfcc.shape} for {wav_path}")
    return logmel.astype(np.float16), mfcc.astype(np.float16)


def _worker(task: tuple[str, str | None, np.ndarray | None]) -> tuple[bytes, bytes]:
    wav_path, _, samples = task
    lm, mf = compute_features(wav_path if samples is None else None, samples)
    return lm.tobytes(), mf.tobytes()


def featurize(tasks: list[tuple[str, str | None, np.ndarray | None]], workers: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(tasks)
    logmel = np.zeros((n, N_FRAMES, N_BINS), dtype=np.float16)
    mfcc = np.zeros((n, N_FRAMES, N_BINS), dtype=np.float16)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        for i, (lm_b, mf_b) in enumerate(pool.map(_worker, tasks, chunksize=64)):
            logmel[i] = np.frombuffer(lm_b, dtype=np.float16).reshape(N_FRAMES, N_BINS)
            mfcc[i] = np.frombuffer(mf_b, dtype=np.float16).reshape(N_FRAMES, N_BINS)
            if (i + 1) % 5000 == 0:
                logger.info(f"  featurized {i + 1}/{n}")
    return logmel, mfcc


def speaker_of(rel: str) -> str:
    name = Path(rel).name
    return name.split("_nohash_")[0] if "_nohash_" in name else Path(rel).stem


def split_background(bg_dir: Path) -> dict[str, list[tuple[str, int, int]]]:
    """Cut each background-noise file into time ranges: 80% train, 10% validation, 10% test.

    Returns split -> list of (file_name, start_sample, end_sample) ranges, so no noise
    segment is shared across splits.
    """
    ranges: dict[str, list[tuple[str, int, int]]] = {"train": [], "validation": [], "testing": []}
    for wav in sorted(bg_dir.glob("*.wav")):
        n = sf.info(str(wav)).frames
        a = int(0.8 * n)
        b = int(0.9 * n)
        ranges["train"].append((wav.name, 0, a))
        ranges["validation"].append((wav.name, a, b))
        ranges["testing"].append((wav.name, b, n))
    return ranges


def make_silence(bg_dir: Path, ranges: list[tuple[str, int, int]], count: int, rng: random.Random) -> list[tuple[str, str | None, np.ndarray | None]]:
    tasks = []
    seen: set[tuple[str, int]] = set()
    usable = [r for r in ranges if r[2] - r[1] >= CLIP_SAMPLES]
    while len(tasks) < count:
        fname, a, b = usable[rng.randrange(len(usable))]
        offset = rng.randrange(a, b - CLIP_SAMPLES + 1)
        if (fname, offset) in seen:  # sample without replacement so clip IDs stay unique
            continue
        seen.add((fname, offset))
        y, _ = sf.read(str(bg_dir / fname), start=offset, frames=CLIP_SAMPLES, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        tasks.append((f"{BG_DIR}/{fname}@{offset}", None, y))
    return tasks


def build(stage_workers: int) -> None:
    setup_logging()
    logger.info("Stage: extract archives")
    extract(V002_TAR, V002_DIR, ".extracted")
    extract(TEST_TAR, TEST_DIR, ".extracted")
    root = V002_DIR
    # archives may nest a top folder; locate the folder holding validation_list.txt
    val_list_path = next(root.rglob("validation_list.txt"))
    base = val_list_path.parent
    logger.info(f"Archive root: {base}")

    validation = set(read_list(val_list_path))
    testing = set(read_list(base / "testing_list.txt"))
    assert not (validation & testing), "validation and testing lists overlap"
    logger.info(f"validation & testing overlap: {len(validation & testing)} (checked)")

    all_wavs = sorted(
        p.relative_to(base).as_posix()
        for p in base.rglob("*.wav")
        if BG_DIR not in p.relative_to(base).parts
    )
    logger.info(f"Word clips found: {len(all_wavs)}")

    # Split membership by official lists; everything else is training.
    word_clips: dict[str, list[tuple[str, str]]] = {"train": [], "validation": [], "testing": []}
    unknown_pool: dict[str, list[str]] = {"train": [], "validation": [], "testing": []}
    for rel in all_wavs:
        word = Path(rel).parent.name
        split = "validation" if rel in validation else "testing" if rel in testing else "train"
        if word in LABEL_INDEX and word in TARGETS:
            word_clips[split].append((rel, word))
        else:
            unknown_pool[split].append(rel)

    tasks_by_split: dict[str, list[tuple[str, str, str, str]]] = {}  # (clip_id, label, split, src)
    tasks_meta: dict[str, list[dict]] = {}
    rng_unknown = random.Random(SEED_UNKNOWN)
    rng_silence = random.Random(SEED_SILENCE)
    bg_ranges = split_background(base / BG_DIR)
    logger.info(f"Background ranges: {({k: len(v) for k, v in bg_ranges.items()})}")

    audio_tasks: dict[str, list[tuple[str, str | None, np.ndarray | None]]] = {}
    meta: dict[str, list[dict]] = {}
    for split_key, split_name in [("train", "train"), ("validation", "validation"), ("testing", "testing")]:
        known = sorted(word_clips[split_key])
        n_extra = math.ceil(len(known) * PCT)
        unk = sorted(unknown_pool[split_key])
        unk_pick = sorted(rng_unknown.sample(unk, n_extra))
        sil_tasks = make_silence(base / BG_DIR, bg_ranges[split_key], n_extra, rng_silence)
        recs = []
        tasks = []
        for rel, word in known:
            recs.append({"clip_id": rel, "label": word, "label_index": LABEL_INDEX[word],
                         "split": split_name, "source_path_in_archive": rel,
                         "speaker_hash": speaker_of(rel), "is_generated": False})
            tasks.append((str(base / rel), None, None))
        for rel in unk_pick:
            recs.append({"clip_id": rel, "label": UNKNOWN_LABEL, "label_index": LABEL_INDEX[UNKNOWN_LABEL],
                         "split": split_name, "source_path_in_archive": rel,
                         "speaker_hash": speaker_of(rel), "is_generated": False})
            tasks.append((str(base / rel), None, None))
        for cid, _, y in sil_tasks:
            recs.append({"clip_id": cid, "label": SILENCE_LABEL, "label_index": LABEL_INDEX[SILENCE_LABEL],
                         "split": split_name, "source_path_in_archive": cid,
                         "speaker_hash": None, "is_generated": True})
            tasks.append((cid, None, y))
        audio_tasks[split_name] = tasks
        meta[split_name] = recs
        logger.info(f"{split_name}: known={len(known)} unknown={len(unk_pick)} silence={len(sil_tasks)} total={len(recs)}")

    # test_bench: the separate 4,890-clip test archive, taken as delivered
    if TEST_DIR.exists():
        bench_wavs = sorted(p for p in TEST_DIR.rglob("*.wav"))
        recs, tasks = [], []
        for p in bench_wavs:
            rel = p.relative_to(TEST_DIR).as_posix()
            label = p.parent.name
            if label not in LABEL_INDEX:
                logger.warning(f"Skipping test-archive file with unexpected label folder: {rel}")
                continue
            # the test archive reuses v0.02 filenames, so its clip_id is prefixed with the archive name
            recs.append({"clip_id": f"speech_commands_test_set_v0.02/{rel}", "label": label,
                         "label_index": LABEL_INDEX[label],
                         "split": "test_bench", "source_path_in_archive": rel,
                         "speaker_hash": speaker_of(rel), "is_generated": False})
            tasks.append((str(p), None, None))
        audio_tasks["test_bench"] = tasks
        meta["test_bench"] = recs
        logger.info(f"test_bench: total={len(recs)}")

    # Featurize every split; write npy and index
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    stats: dict[str, dict] = {}
    seen_ids: Counter = Counter()
    for split in ["train", "validation", "testing", "test_bench"]:
        if split not in meta:
            continue
        logger.info(f"Featurizing {split}: {len(audio_tasks[split])} clips with {stage_workers} workers")
        lm, mf = featurize(audio_tasks[split], stage_workers)
        assert lm.shape[1:] == (N_FRAMES, N_BINS) and mf.shape[1:] == (N_FRAMES, N_BINS)
        assert lm.dtype == np.float16 and mf.dtype == np.float16
        np.save(OUT_DIR / f"features_{split}.npy", lm)
        np.save(OUT_DIR / f"mfcc_features_{split}.npy", mf)
        logger.info(f"Saved features_{split}.npy {lm.shape} and mfcc_features_{split}.npy")
        for i, rec in enumerate(meta[split]):
            rec = dict(rec)
            rec["feature_sha256"] = hashlib.sha256(lm[i].tobytes()).hexdigest()
            rec["feature_index"] = i
            seen_ids[rec["clip_id"]] += 1
            index.append(rec)
        counts = Counter(r["label"] for r in meta[split])
        stats[split] = {"n": len(meta[split]), "label_counts": dict(sorted(counts.items()))}

    dups = [k for k, v in seen_ids.items() if v > 1]
    logger.info(f"Duplicate clip_ids: {len(dups)}")
    assert not dups, f"duplicate clip ids: {dups[:5]}"

    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=1))
    logger.info(f"Wrote index.json with {len(index)} rows")

    # exp_sel_data_out projection for aii-json validation
    examples_by_split = defaultdict(list)
    for r in index:
        examples_by_split[r["split"]].append({
            "input": r["clip_id"],
            "output": r["label"],
            "metadata_split": r["split"],
            "metadata_label_index": r["label_index"],
            "metadata_speaker_hash": r["speaker_hash"] if r["speaker_hash"] is not None else "",
            "metadata_source_path_in_archive": r["source_path_in_archive"],
            "metadata_feature_sha256": r["feature_sha256"],
            "metadata_is_generated": r["is_generated"],
        })
    full = {
        "metadata": {
            "description": "Frozen keyword-spotting feature cache index (audio excluded)",
            "source": "Speech Commands v0.02 (Warden 2018) + speech_commands_test_set_v0.02",
            "feature": {"sr": SR, "window_samples": WIN, "hop_samples": HOP, "n_fft": N_FFT,
                        "frames": N_FRAMES, "bins": N_BINS, "log_floor": LOG_FLOOR},
            "seeds": {"unknown": SEED_UNKNOWN, "silence": SEED_SILENCE},
            "label_index": {**{w: LABEL_INDEX[w] for w in TARGETS}, SILENCE_LABEL: 10, UNKNOWN_LABEL: 11},
        },
        "datasets": [
            {"dataset": f"speech_commands_v0.02_kws12_{s}", "examples": examples_by_split[s]}
            for s in ["train", "validation", "testing", "test_bench"] if examples_by_split[s]
        ],
    }
    (OUT_DIR / "full_data_out.json").write_text(json.dumps(full))

    # mini and preview variants: first 200 rows and first 20 rows per split
    mini, preview = [], []
    for s in ["train", "validation", "testing", "test_bench"]:
        rows = [r for r in index if r["split"] == s]
        mini.extend(rows[:200])
        preview.extend(rows[:20])
    (OUT_DIR / "mini_index.json").write_text(json.dumps(mini, indent=1))
    (OUT_DIR / "preview_index.json").write_text(json.dumps(preview, indent=1))

    (OUT_DIR / "split_stats.json").write_text(json.dumps({
        "list_lines": {"validation_list.txt": len(validation), "testing_list.txt": len(testing)},
        "splits": stats,
        "seeds": {"unknown": SEED_UNKNOWN, "silence": SEED_SILENCE},
        "unknown_silence_percent": PCT * 100,
        "label_index": {**{w: LABEL_INDEX[w] for w in TARGETS}, SILENCE_LABEL: 10, UNKNOWN_LABEL: 11},
        "feature_settings": {"sr": SR, "window_samples": WIN, "hop_samples": HOP, "n_fft": N_FFT,
                             "frames": N_FRAMES, "bins": N_BINS, "log_floor": LOG_FLOOR,
                             "log_mel_n_mels": N_BINS, "mfcc_n_mels": 40, "mfcc_n_coeffs": N_BINS},
    }, indent=2))
    logger.info("Done. Stage complete.")


@logger.catch(reraise=True)
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    build(args.workers)


if __name__ == "__main__":
    main()
