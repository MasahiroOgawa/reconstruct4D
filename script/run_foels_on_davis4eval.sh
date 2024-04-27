#!/bin/bash
###

# stop immediately when error occurred
set -eu

ROOT_DIR=$(dirname "$0")/..
for DATA_DIR in ${ROOT_DIR}/reconstruct4D/ext/unsupervised_detection/download/DAVIS/JPEGImages/480p/*
do
    echo "[INFO] read DATA_DIR=${DATA_DIR}"
    $ROOT_DIR/script/run_foels.sh $DATA_DIR
done