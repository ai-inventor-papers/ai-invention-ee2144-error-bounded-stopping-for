#!/usr/bin/env bash
# Rebuild every file marked `delete` in .aii/manifest.yaml. Run from the workspace root.
set -euo pipefail
cd "$(dirname "$0")"
# 1. environment (regenerable)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-frozen.txt
# 2. raw Speech Commands v2 parquet mirror (redownloadable). train is downloaded whole, then split
#    into <100 MB shards by shard_train.py (the shards are what method.py reads).
mkdir -p data_raw
curl -L -o data_raw/validation.parquet "https://huggingface.co/datasets/danjacobellis/speech_commands_v2/resolve/main/data/validation-00000-of-00001.parquet"
curl -L -o data_raw/test.parquet "https://huggingface.co/datasets/danjacobellis/speech_commands_v2/resolve/main/data/test-00000-of-00001.parquet"
curl -L -o data_raw/train.parquet "https://huggingface.co/datasets/danjacobellis/speech_commands_v2/resolve/main/data/train-00000-of-00001.parquet"
.venv/bin/python shard_train.py
.venv/bin/python shard_train.py --verify
rm -f data_raw/train.parquet
# 3. features (regenerable from data_raw)
.venv/bin/python method.py prep
