#!/bin/bash
# Run from the package root (AAC_v1.2_clean/)
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/train_canonical_transformer.py "$@"
