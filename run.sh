#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence for demonstration purposes.

set -euo pipefail

echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"

# Futtatjuk a szkripteket sorban
echo "[run.sh] === Running 01-data-preprocessing.py ==="
python src/01-data-preprocessing.py

echo "[run.sh] === Running 02-training.py ==="
python src/02-training.py

echo "[run.sh] === Running 03-evaluation.py ==="
python src/03-evaluation.py

echo "[run.sh] === Running 04-inference.py on holdout (unseen) data ==="
# Futtatjuk az inference szkriptet a holdout adatokon (unseen data)
python src/04-inference.py

echo "[run.sh] Pipeline finished at $(date --iso-8601=seconds)"
echo "[run.sh] All stages completed successfully."
