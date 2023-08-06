#!/bin/bash

# set root directory
ROOT_DIR=$(dirname "$0")/..

# variables. You can change this.
INPUT_IMAGE_DIR=${ROOT_DIR}/data/sample

# automatically defined from INPUT_IMAGE_DIR
OUTPUT_DIR=${ROOT_DIR}/output/$(basename ${INPUT_IMAGE_DIR})
OUTPUT_FLOW_DIR=${OUTPUT_DIR}/flow

####################

echo "[INFO] compute optical flow"
if [ -d ${OUTPUT_FLOW_DIR} ]; then
       echo "[INFO] ${OUTPUT_FLOW_DIR} already exists. Skip computing optical flow."
else
       mkdir -p ${OUTPUT_FLOW_DIR}
       export OMP_NUM_THREADS=1
       CUDA_VISIBLE_DEVICES=0 python ${ROOT_DIR}/ext/unimatch/main_flow.py \
       --inference_dir ${INPUT_IMAGE_DIR} \
       --output_path ${OUTPUT_FLOW_DIR} \
       --resume ${ROOT_DIR}/ext/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
       --padding_factor 32 \
       --upsample_factor 4 \
       --num_scales 2 \
       --attn_splits_list 2 8 \
       --corr_radius_list -1 4 \
       --prop_radius_list -1 1 \
       --reg_refine \
       --num_reg_refine 6 \
       --save_flo_flow
       echo "[INFO] save optical flow to ${OUTPUT_FLOW_DIR}"
fi

echo "[INFO] run extract moving objects"
python ${ROOT_DIR}/reconstruct4D/extract_moving_objects.py \
       --input_dir ${INPUT_IMAGE_DIR} \
       --flow_result_dir ${OUTPUT_FLOW_DIR} \
       --output_dir ${OUTPUT_DIR}
