#!/bin/bash

# stop immediately when error occurred
set -eu

# set root directory
ROOT_DIR=$(dirname "$0")/..

# input image directory or video variables. You can change this.
# INPUT=${ROOT_DIR}/data/sample
# INPUT=${ROOT_DIR}/data/todaiura
INPUT="/home/mas/Downloads/reirun"
LOG_LEVEL=3 # 0: no log but save the result images, 1: print log, 2: display image, 3: debug with detailed image
IMG_HEIGHT=480
SKIP_FRAMES=80


####################

echo "[INFO] check input is whether a directory or movie."
if [ -d ${INPUT} ]; then
       echo "[INFO] input is a directory."
elif [ -f ${INPUT} ]; then
       echo "[INFO] input is a movie."
       echo "[INFO] convert movie to images"
       INPUT_DIR=$(dirname ${INPUT})
       ffmpeg -i ${INPUT} -r 30 -vf scale=-1:${IMG_HEIGHT} ${INPUT_DIR}/%06d.png
       INPUT="${INPUT_DIR}"
else
       echo "[ERROR] input is neither a directory nor a movie."
       exit 1
fi

# automatically defined from INPUT
OUTPUT_PARENT_DIR=${ROOT_DIR}/output/$(basename ${INPUT})
OUTPUT_FLOW_DIR=${OUTPUT_PARENT_DIR}/flow
OUTPUT_SEG_DIR=${OUTPUT_PARENT_DIR}/segmentation
OUTPUT_MOVOBJ_DIR=${OUTPUT_PARENT_DIR}/moving_object

echo "[INFO] compute optical flow"
eval "$(conda shell.bash activate reconstruct4D)"
echo "[INFO] env: $CONDA_DEFAULT_ENV" 
if [ -d ${OUTPUT_FLOW_DIR} ]; then
       echo "[INFO] ${OUTPUT_FLOW_DIR} already exists. Skip computing optical flow."
else
       mkdir -p ${OUTPUT_FLOW_DIR}
       export OMP_NUM_THREADS=1
       CUDA_VISIBLE_DEVICES=0 python ${ROOT_DIR}/reconstruct4D/ext/unimatch/main_flow.py \
       --inference_dir ${INPUT} \
       --output_path ${OUTPUT_FLOW_DIR} \
       --resume ${ROOT_DIR}/reconstruct4D/ext/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth \
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
       echo "[INFO] creating a flow movie"
       ffmpeg -framerate 30  -pattern_type glob -i "${OUTPUT_FLOW_DIR}/*.png" \
              -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${OUTPUT_FLOW_DIR}/flow.mp4
fi


echo "[INFO] run segmentation"
# to avoid error: "anaconda3/envs/internimage/etc/conda/activate.d/libblas_mkl_activate.sh: 
# line 1: MKL_INTERFACE_LAYER: unbound variable", we set +u.
set +eu
eval "$(conda shell.bash activate internimage)"
set -eu
echo "[INFO] env: $CONDA_DEFAULT_ENV"
if [ -d ${OUTPUT_SEG_DIR} ]; then
       echo "[INFO] ${OUTPUT_SEG_DIR} already exists. Skip running segmentation."
else
       mkdir -p ${OUTPUT_SEG_DIR}
       CUDA_VISIBLE_DEVICES=0 python ${ROOT_DIR}/reconstruct4D/ext/InternImage/segmentation/image_demo.py \
              ${INPUT} \
              ${ROOT_DIR}/reconstruct4D/ext/InternImage/segmentation/configs/ade20k/upernet_internimage_t_512_160k_ade20k.py  \
              ${ROOT_DIR}/reconstruct4D/ext/InternImage/segmentation/checkpoint_dir/seg/upernet_internimage_t_512_160k_ade20k.pth  \
               --palette ade20k --out ${OUTPUT_SEG_DIR}

       # if you have strong GPU, you can use the following model.
       # CUDA_VISIBLE_DEVICES=0 python ${ROOT_DIR}/ext/InternImage/segmentation/image_demo.py \
       #        ${INPUT} \
       #        ${ROOT_DIR}/ext/InternImage/segmentation/configs/ade20k/mask2former_internimage_h_896_80k_cocostuff2ade20k_ss.py  \
       #        ${ROOT_DIR}/ext/InternImage/segmentation/checkpoint_dir/seg/mask2former_internimage_h_896_80k_cocostuff2ade20k.pth \
       #         --palette ade20k --out ${OUTPUT_SEG_DIR}
fi


echo "[INFO] run extract moving objects"
eval "$(conda shell.bash activate reconstruct4D)"
echo "[INFO] env: $CONDA_DEFAULT_ENV"
mkdir -p ${OUTPUT_MOVOBJ_DIR}
python ${ROOT_DIR}/reconstruct4D/extract_moving_objects.py \
       --input_dir ${INPUT} \
       --flow_result_dir ${OUTPUT_FLOW_DIR} \
       --segment_result_dir ${OUTPUT_SEG_DIR} \
       --output_dir ${OUTPUT_MOVOBJ_DIR} \
       --skip_frames ${SKIP_FRAMES} \
       --loglevel ${LOG_LEVEL}

echo "[INFO] creating a segmentation movie (ffmpeg in InternImage conda env doesn't support libx264, so we create it here.)"
# for segmentation, the image file format is jpg or png. so detect it first.
IMG_EXT=
if [ `ls -1 ${OUTPUT_SEG_DIR}/ls -1 "${OUTPUT_SEG_DIR}/*.png" 2>/dev/null*.jpg 2>/dev/null | wc -l` != 0 ]; then
       IMG_EXT=jpg
elif [ `ls -1 ${OUTPUT_SEG_DIR}/*.png 2>/dev/null | wc -l` != 0 ]; then
       IMG_EXT=png
else
       echo "[ERROR] no jpg or png image file in ${OUTPUT_SEG_DIR}"
       exit 1
fi
ffmpeg -framerate 30  -pattern_type glob -i "${OUTPUT_MOVOBJ_DIR}/*.${IMG_EXT}" \
       -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${OUTPUT_SEG_DIR}/segmentation.mp4

echo "[INFO] creating a final movie"
ffmpeg -framerate 30  -pattern_type glob -i "${OUTPUT_MOVOBJ_DIR}/*.png" \
       -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${OUTPUT_MOVOBJ_DIR}/moving_object.mp4