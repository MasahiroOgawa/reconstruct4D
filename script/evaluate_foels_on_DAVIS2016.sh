#!/bin/bash
# This script is not intended to run directly, but called by `run_foels_on_davis4eval.sh.

# stop immediately when error occurred
set -eu

echo "[INFO] set parameters."
ROOT_DIR=$(dirname "$0")/..

echo "[INFO] deactivate current uv venv first. otherwise, conda env will be hide."
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[INFO] No virtualenv activated."
else
    echo "[INFO] deactivate current uv venv: $VIRTUAL_ENV"
    PATH=$(echo $PATH | tr ':' '\n' | grep -v "$VIRTUAL_ENV" | tr '\n' ':')
    unset VIRTUAL_ENV
fi

echo "[INFO] run evaluation on DAVIS dataset."
set +eu
eval "$(conda shell.bash activate contextual-information-separation)"
set -eu
echo "[INFO] env: $CONDA_DEFAULT_ENV"
python ${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection/scripts/test_DAVIS2016_foels.py