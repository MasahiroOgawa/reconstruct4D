#!/bin/bash
###

# stop immediately when error occurred
set -eu

echo "[INFO] set parameters."
ROOT_DIR=$(dirname "$0")/..
ROOT_DATA_DIR=${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection/download/DAVIS/JPEGImages/480p


echo "[INFO] check input file existence."
if [ -n "$(ls -A ${ROOT_DATA_DIR})" ]; then
    echo "[INFO] input files exist. skip downloading."
else
    echo "[INFO] input files do not exist. set up and download it."
    (
        cd ${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection
        # to avoid "ADDR2LINE: unbound variable" error, unset -u.
        set +eu
        eval "$(conda shell.bash activate contextual-information-separation)"
        set -eu
        echo "[INFO] env: $CONDA_DEFAULT_ENV"
        bash ./scripts/test_DAVIS2016_raw.sh
    )
fi


for DATA_DIR in ${ROOT_DATA_DIR}/*
do
    echo "[INFO] read DATA_DIR=${DATA_DIR}"
    $ROOT_DIR/script/run_foels.sh $DATA_DIR
done