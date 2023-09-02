#!/bin/bash

# set root directory
ROOT_DIR=$(dirname "$0")/..

# variables. You can change this.
# INPUT_IMAGE_DIR=${ROOT_DIR}/data/sample
INPUT_IMAGE_DIR=${ROOT_DIR}/data/todaiura

# automatically defined from INPUT_IMAGE_DIR
OUTPUT_DIR=${ROOT_DIR}/output/$(basename ${INPUT_IMAGE_DIR})
OUTPUT_FLOW_DIR=${OUTPUT_DIR}/flow
OUTPUT_SEG_DIR=${OUTPUT_DIR}/seg

####################


echo "[INFO] compute optical flow"
eval "$(conda shell.bash activate reconstruct4D)"
echo "[INFO] env: $CONDA_DEFAULT_ENV" 
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


echo "[INFO] run segmentation"
eval "$(conda shell.bash activate internimage)"
echo "[INFO] env: $CONDA_DEFAULT_ENV"
if [ -d ${OUTPUT_SEG_DIR} ]; then
       echo "[INFO] ${OUTPUT_SEG_DIR} already exists. Skip running segmentation."
else
       mkdir -p ${OUTPUT_SEG_DIR}
       CUDA_VISIBLE_DEVICES=0 python ${ROOT_DIR}/ext/InternImage/segmentation/image_demo.py \
              ${INPUT_IMAGE_DIR} \
              ${ROOT_DIR}/ext/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k.py  \
              ${ROOT_DIR}/ext/InternImage/segmentation/checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth  \
               --palette ade20k --out ${OUTPUT_SEG_DIR}

       # if you have strong GPU, you can use the following model.
       # CUDA_VISIBLE_DEVICES=0 python ${ROOT_DIR}/ext/InternImage/segmentation/image_demo.py \
       #        ${INPUT_IMAGE_DIR} \
       #        ${ROOT_DIR}/ext/InternImage/segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py  \
       #        ${ROOT_DIR}/ext/InternImage/segmentation/checkpoint_dir/seg/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth \
       #         --palette ade20k --out ${OUTPUT_SEG_DIR}
fi


echo "[INFO] run extract moving objects"
eval "$(conda shell.bash activate reconstruct4D)"
echo "[INFO] env: $CONDA_DEFAULT_ENV"
python ${ROOT_DIR}/reconstruct4D/extract_moving_objects.py \
       --input_dir ${INPUT_IMAGE_DIR} \
       --flow_result_dir ${OUTPUT_FLOW_DIR} \
       --output_dir ${OUTPUT_DIR}
