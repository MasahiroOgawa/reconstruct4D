#!/bin/bash

# set root directory
ROOTDIR=$(dirname "$0")/..
cd ${ROOTDIR}/reconstruct4D

# run demo
python compute_foe.py \
       --inference_dir ${ROOTDIR}/unimatch/demo/todaiura \
       --resume ${ROOTDIR}/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
