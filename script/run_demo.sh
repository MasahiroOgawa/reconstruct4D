#!/bin/bash

ROOTDIR=$(dirname "$0")/..

cd ${ROOTDIR}/reconstruct4D

python compute_flow.py \
       --inference_dir ${ROOTDIR}/unimatch/demo/todaiura \
       --resume ${ROOTDIR}/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
