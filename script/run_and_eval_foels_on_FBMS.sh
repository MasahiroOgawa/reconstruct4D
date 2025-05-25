#!/bin/bash
###

# stop immediately when error occurred
set -eu

echo "[INFO] set parameters."
ROOT_DIR=$(dirname "$0")/..
ROOT_DATA_DIR=${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection/download/FBMS
DATASET_NAME="FBMS"


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
        python ./scripts/test_${DATASET_NAME}_raw.py
    )
fi


echo "[INFO] run foels on the dataset."
# Trainset and Testset exists for FBMS. so we need to run foels on both.
for DATA_TYPE_FULLPATH in ${ROOT_DATA_DIR}/*
do
    DATA_TYPE=$(basename $DATA_TYPE_FULLPATH)
    for DATA_DIR in ${DATA_TYPE_FULLPATH}/*
    do
        echo "[INFO] read DATA_DIR=${DATA_DIR}"
        $ROOT_DIR/script/run_foels.sh $DATA_DIR ${ROOT_DIR}/result/${DATASET_NAME}/${DATA_TYPE}
    done
done

echo "[INFO] run evaluation."
$ROOT_DIR/script/evaluate_foels_on_${DATASET_NAME}.sh

echo "[INFO] copy results to result directory."
cp ${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection/results/Foels/${DATASET_NAME}/result.csv ${ROOT_DIR}/result/${DATASET_NAME}

echo "[INFO] finish."