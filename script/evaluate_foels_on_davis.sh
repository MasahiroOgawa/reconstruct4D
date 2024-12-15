#!/bin/bash

# stop immediately when error occurred
set -eu

echo "[INFO] set parameters."
ROOT_DIR=$(dirname "$0")/..

if [ ! -d "${ROOT_DIR}/output/davis" ]; then
    echo "Output doesn't exist, running script"
    ${ROOT_DIR}/script/run_foels_on_davis4eval.sh
fi

echo "[INFO] deactivate current uv venv first. otherwise, conda env will be hide."
if [ -z "${VIRTUAL_ENV}" ]; then
    echo "[INFO] No virtualenv activated."
else
    echo "[INFO] deactivate current uv venv: $VIRTUAL_ENV"
    PATH=$(echo $PATH | tr ':' '\n' | grep -v "$VIRTUAL_ENV" | tr '\n' ':')
    unset VIRTUAL_ENV
fi


# run eval
${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection/scripts/test_DAVIS2016_foels.sh