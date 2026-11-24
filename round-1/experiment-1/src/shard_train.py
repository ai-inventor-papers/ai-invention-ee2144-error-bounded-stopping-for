#!/usr/bin/env python3
"""Split data_raw/train.parquet (254 MB) into row-group-aligned shards under 100 MB each.

Shards: data_raw/train_part_000.parquet, train_part_001.parquet, ... Rows keep their original order,
so concatenating the shards in name order reproduces the original table exactly (checked by --verify)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data_raw"
N_PARTS = 3


def shard(src: Path) -> None:
    pf = pq.ParquetFile(src)
    n_rg = pf.metadata.num_row_groups
    bounds = [round(i * n_rg / N_PARTS) for i in range(N_PARTS + 1)]
    for p in range(N_PARTS):
        out = RAW / f"train_part_{p:03d}.parquet"
        writer = None
        for rg in range(bounds[p], bounds[p + 1]):
            tbl = pf.read_row_group(rg)
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema)
            writer.write_table(tbl)
        writer.close()
        print(f"{out.name}: row groups {bounds[p]}..{bounds[p + 1] - 1}, {out.stat().st_size / 1e6:.1f} MB")


def verify(src: Path) -> None:
    import hashlib
    h_src, n_src = hashlib.sha256(), 0
    for rg_tbl in pq.ParquetFile(src).iter_batches(batch_size=4096, columns=["path", "opus"]):
        for p, o in zip(rg_tbl.column("path").to_pylist(), rg_tbl.column("opus").to_pylist()):
            h_src.update(p.encode()); h_src.update(o["bytes"]); n_src += 1
    h_new, n_new = hashlib.sha256(), 0
    for part in sorted(RAW.glob("train_part_*.parquet")):
        for b in pq.ParquetFile(part).iter_batches(batch_size=4096, columns=["path", "opus"]):
            for p, o in zip(b.column("path").to_pylist(), b.column("opus").to_pylist()):
                h_new.update(p.encode()); h_new.update(o["bytes"]); n_new += 1
    print(f"rows original={n_src} shards={n_new}; content hash equal: {h_src.hexdigest() == h_new.hexdigest()}")
    if n_src != n_new or h_src.hexdigest() != h_new.hexdigest():
        raise SystemExit("verification failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--src", default=str(RAW / "train.parquet"))
    a = ap.parse_args()
    if a.verify:
        verify(Path(a.src))
    else:
        shard(Path(a.src))
