#!/usr/bin/env python3
"""Build the exp_sel_data_out file for the frozen keyword-spotting feature cache.

Reads features/index.json and the four split arrays (log-mel and MFCC) written by
scripts/build_feature_cache.py, verifies shape, dtype, row counts and the per-row
SHA-256 of the log-mel bytes, checks that every archive-sourced clip exists in the
extracted archive (when temp/datasets/extracted is still present), and writes
features/full_data_out.json: one dataset group with one example per clip.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
FEAT_DIR = ROOT / "features"
EXTRACTED = ROOT / "temp" / "datasets" / "extracted"
INDEX_PATH = FEAT_DIR / "index.json"
STATS_PATH = FEAT_DIR / "split_stats.json"
OUT_PATH = ROOT / "full_data_out.json"
MINI_PATH = ROOT / "mini_data_out.json"
PREVIEW_PATH = ROOT / "preview_data_out.json"
DATASET_NAME = "speech_commands_v0.02_kws12"
SPLITS = ["train", "validation", "testing", "test_bench"]
N_FRAMES, N_BINS = 49, 10
SOURCE_DIRS = {"v0.02": EXTRACTED / "v0.02", "test_set": EXTRACTED / "test_set"}


def setup_logging() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    (ROOT / "logs").mkdir(exist_ok=True)
    logger.add(ROOT / "logs" / "data.log", rotation="30 MB", level="DEBUG")


def load_split_arrays(split: str, n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    lm = np.load(FEAT_DIR / f"features_{split}.npy", mmap_mode="r")
    mf = np.load(FEAT_DIR / f"mfcc_features_{split}.npy", mmap_mode="r")
    for name, arr in (("log-mel", lm), ("MFCC", mf)):
        if arr.shape != (n_expected, N_FRAMES, N_BINS):
            raise ValueError(f"{split} {name} shape {arr.shape} != {(n_expected, N_FRAMES, N_BINS)}")
        if arr.dtype != np.float16:
            raise ValueError(f"{split} {name} dtype {arr.dtype} != float16")
    return lm, mf


def main() -> None:
    setup_logging()
    stats = json.loads(STATS_PATH.read_text())
    index = json.loads(INDEX_PATH.read_text())
    logger.info(f"Loaded {len(index)} index rows from {INDEX_PATH}")

    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for rec in index:
        by_split[rec["split"]].append(rec)
    for split in SPLITS:
        logger.info(f"{split}: {len(by_split[split])} index rows")

    examples: list[dict] = []
    missing_sources = 0
    hash_mismatch = 0
    checked_sources = 0
    sources_available = EXTRACTED.exists()
    if not sources_available:
        logger.warning(f"{EXTRACTED} missing; skipping source-file existence check")

    for split in SPLITS:
        rows = sorted(by_split[split], key=lambda r: r["feature_index"])
        n = len(rows)
        if [r["feature_index"] for r in rows] != list(range(n)):
            raise ValueError(f"{split}: feature_index is not 0..{n - 1}")
        lm, _mf = load_split_arrays(split, n)
        logger.info(f"{split}: arrays OK, shape {lm.shape}, dtype {lm.dtype}")

        for rec in rows:
            i = rec["feature_index"]
            digest = hashlib.sha256(np.ascontiguousarray(lm[i]).tobytes()).hexdigest()
            if digest != rec["feature_sha256"]:
                hash_mismatch += 1
            if sources_available and not rec["is_generated"]:
                root = SOURCE_DIRS.get(split_source_dir(split))
                checked_sources += 1
                if root is None or not (root / rec["source_path_in_archive"]).exists():
                    missing_sources += 1

            examples.append({
                "input": rec["clip_id"],
                "output": rec["label"],
                "metadata_split": split,
                "metadata_label_index": rec["label_index"],
                "metadata_speaker_hash": rec["speaker_hash"],
                "metadata_feature_index": i,
                "metadata_feature_sha256": rec["feature_sha256"],
                "metadata_is_generated": rec["is_generated"],
            })

    if hash_mismatch:
        raise ValueError(f"{hash_mismatch} rows failed the feature_sha256 check")
    if missing_sources:
        raise ValueError(f"{missing_sources} of {checked_sources} archive clips are missing from {EXTRACTED}")
    logger.info(f"Per-row SHA-256 check passed for {len(examples)} rows")
    if sources_available:
        logger.info(f"Source-file existence check passed for {checked_sources} archive clips")

    counts = Counter(e["output"] for e in examples)
    logger.info(f"Label counts over all splits: {dict(sorted(counts.items()))}")

    output = {
        "metadata": {
            "description": "Frozen keyword-spotting feature cache (audio excluded). One example per clip; "
                           "features are in features_<split>.npy (log-mel) and mfcc_features_<split>.npy (MFCC), "
                           "row i = metadata_feature_index of the example in that split.",
            "source": "Speech Commands v0.02 (Warden 2018) official archive and speech_commands_test_set_v0.02",
            "splits": {s: len(by_split[s]) for s in SPLITS},
            "feature_settings": stats.get("feature_settings"),
            "seeds": stats.get("seeds"),
            "label_index": {lbl: idx for lbl, idx in sorted(
                {(e["output"], e["metadata_label_index"]) for e in examples}, key=lambda t: t[1])},
        },
        "datasets": [{"dataset": DATASET_NAME, "examples": examples}],
    }

    tmp = OUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False))
    tmp.replace(OUT_PATH)
    logger.info(f"Wrote {len(examples)} examples to {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")

    # Mini and preview variants, following the aii-json rule: first 3 examples per
    # dataset group; preview also truncates every string to 200 characters.
    mini = {**output, "datasets": [{**ds, "examples": ds["examples"][:3]} for ds in output["datasets"]]}
    preview = truncate_strings(mini)
    MINI_PATH.write_text(json.dumps(mini, indent=2, ensure_ascii=False))
    PREVIEW_PATH.write_text(json.dumps(preview, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {MINI_PATH.name} and {PREVIEW_PATH.name}")


def truncate_strings(value):
    """Recursively cut every string to 200 characters, as the aii-json preview variant does."""
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "...[truncated]"
    if isinstance(value, list):
        return [truncate_strings(v) for v in value]
    if isinstance(value, dict):
        return {k: truncate_strings(v) for k, v in value.items()}
    return value


def split_source_dir(split: str) -> str:
    """Map a split to the extracted archive directory its clips come from."""
    return "test_set" if split == "test_bench" else "v0.02"


if __name__ == "__main__":
    main()
