#!/usr/bin/env python3
"""Recompute log-mel features for a sample of archive clips and compare them to the stored hashes.

Checks that the currently pinned librosa/soundfile/numpy reproduce the feature bytes
recorded in features/index.json, without rerunning the full extraction. Generated
(_silence_ / _unknown_ from background noise) clips are skipped.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
EXTRACTED = ROOT / "temp" / "datasets" / "extracted"
SAMPLE_PER_SPLIT = 50


def main() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    sys.path.insert(0, str(SCRIPTS))
    from build_feature_cache import compute_features  # the exact extraction code

    index = json.loads((ROOT / "features" / "index.json").read_text())
    by_split: dict[str, list[dict]] = {}
    for rec in index:
        if not rec["is_generated"]:
            by_split.setdefault(rec["split"], []).append(rec)

    checked = 0
    mismatches: list[str] = []
    for split, rows in by_split.items():
        step = max(1, len(rows) // SAMPLE_PER_SPLIT)
        sample = rows[::step][:SAMPLE_PER_SPLIT]
        source_dir = EXTRACTED / ("test_set" if split == "test_bench" else "v0.02")
        for rec in sample:
            wav = source_dir / rec["source_path_in_archive"]
            lm, _mf = compute_features(str(wav), None)
            digest = hashlib.sha256(np.asarray(lm, dtype=np.float16).tobytes()).hexdigest()
            checked += 1
            if digest != rec["feature_sha256"]:
                mismatches.append(f"{split}:{rec['clip_id']}")
        logger.info(f"{split}: checked {len(sample)} clips")

    if mismatches:
        logger.error(f"{len(mismatches)} of {checked} clips do not reproduce; first: {mismatches[:5]}")
        raise SystemExit(1)
    logger.info(f"All {checked} sampled clips reproduce their stored feature_sha256")


if __name__ == "__main__":
    main()
