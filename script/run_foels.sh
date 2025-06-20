#!/bin/bash
###
USAGE="Usage: $0 [input image directory or movie (default: data/sample)] [result directory (default: result)]"
echo $USAGE

# -e: stop immediately when error occurred
# -u: stop immediately when undefined variable is used
set -eu

# set root directory
ROOT_DIR=$(dirname "$0")/..


# parameters
####################
# input image directory or video variables. You can change this.
INPUT=${1:-${ROOT_DIR}/data/sample}
RESULT_PARENT_DIR=${2:-${ROOT_DIR}/result}
 # LOG_LEVEL=0: no log but save the result images, 1: print log, 2: display image
 # 3: display detailed debug image but without stopping, 4: display debug image and stop every frame.
 # 5: run python debugger. push F5 after running the script.
LOG_LEVEL=1
IMG_HEIGHT=480
# FRAME 79 #parallel moving track  #107 #stopping pedestrians for Todaiura data.
SKIP_FRAMES=0 
# SEG_MODEL_NAME options = {"upernet_internimage_t_512_160k_ade20k.pth", "upernet_internimage_xl_640_160k_ade20k.pth", 
# "upernet_internimage_h_896_160k_ade20k.pth", "mask_rcnn_internimage_t_fpn_1x_coco.pth"}
# "shi-labs/oneformer_coco_swin_large"
SEG_MODEL_NAME="shi-labs/oneformer_coco_swin_large"
# number of iteration in RANSAC
NUM_RANSAC=20
# whether run ransac all inlier estimation or not.
RANSAC_ALL_INLIER_ESTIMATION=True
# FOE_SEARCH_STEP: the number of steps to search the focus of expansion (FOE) in the image.
FOE_SEARCH_STEP=5
THRE_MOVING_FRACTION_IN_OBJ=0.01
####################

if [ $LOG_LEVEL -ge 3 ]; then
       echo "[INFO] set debug mode"
       set -x
fi

# to define INPUT_DIR, we need to do below first.
echo "[INFO] check input is whether a directory or movie."
if [ -d ${INPUT} ]; then
       echo "[INFO] input is a directory."
       INPUT_DIR=${INPUT}
elif [ -f ${INPUT} ]; then
       echo "[INFO] input is a movie."
       echo "[INFO] convert movie to images"
       INPUT_DIR=$(dirname ${INPUT})
       ffmpeg -i ${INPUT} -r 30 -vf scale=-1:${IMG_HEIGHT} ${INPUT_DIR}/%06d.png
else
       echo "[ERROR] input is neither a directory nor a movie."
       exit 1
fi

# automatically defined variables from INPUT
RESULT_PARENT_DIR=${RESULT_PARENT_DIR}/$(basename ${INPUT_DIR})
RESULT_FLOW_DIR=${RESULT_PARENT_DIR}/flow
RESULT_SEG_DIR=${RESULT_PARENT_DIR}/segmentation
RESULT_MOVOBJ_DIR=${RESULT_PARENT_DIR}/moving_object
FLOW_MODEL_NAME=gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth
case ${SEG_MODEL_NAME} in
       "upernet_internimage_t_512_160k_ade20k.pth" |\
       "upernet_internimage_xl_640_160k_ade20k.pth" |\
       "upernet_internimage_h_896_160k_ade20k.pth")
              SEG_MODEL_TYPE="internimage"
              SEG_CHECKPOINT_DIR="${ROOT_DIR}/reconstruct4D/ext/InternImage/checkpoint_dir/seg"
              SEG_TASK_TYPE="semantic";;
       "mask_rcnn_internimage_t_fpn_1x_coco.pth")
              SEG_MODEL_TYPE="internimage"
              SEG_CHECKPOINT_DIR="${ROOT_DIR}/reconstruct4D/ext/InternImage/checkpoint_dir/det"
              SEG_TASK_TYPE="instance";;
       "shi-labs/oneformer_coco_swin_large")
              SEG_MODEL_TYPE="oneformer"
              SEG_TASK_TYPE="panoptic";;
       *)
              echo "[ERROR] unknown segmentation model name: ${SEG_MODEL_NAME}"
              exit 1;;
esac

# print variables
if [ $LOG_LEVEL -ge 1 ]; then
       echo "[INFO] print all variables"
       echo -e "\tINPUT_DIR: ${INPUT_DIR}"
       echo -e "\tRESULT_PARENT_DIR: ${RESULT_PARENT_DIR}"
       echo -e "\tRESULT_FLOW_DIR: ${RESULT_FLOW_DIR}"
       echo -e "\tRESULT_SEG_DIR: ${RESULT_SEG_DIR}"
       echo -e "\tRESULT_MOVOBJ_DIR: ${RESULT_MOVOBJ_DIR}"
       echo -e "\tFLOW_MODEL_NAME: ${FLOW_MODEL_NAME}"
       echo -e "\tSEG_MODEL_NAME: ${SEG_MODEL_NAME}"
       echo -e "\tSEG_MODEL_TYPE: ${SEG_MODEL_TYPE}"
       echo -e "\tSEG_TASK_TYPE: ${SEG_TASK_TYPE}"
fi
if [ $LOG_LEVEL -ge 5 ]; then
       echo "[NOTE] Please press F5 to start debugging!"
fi

deactivate_allenvs() {
       while [ -n "$VIRTUAL_ENV" ]; do
              echo "[INFO] deactivate env: $VIRTUAL_ENV"
              deactivate || conda deactivate
       done
       echo "[INFO] deactivate all envs. current env: $VIRTUAL_ENV"

       # remove .venv from PATH
       PATH=$(echo $PATH | tr ':' '\n' | grep -v "\.venv" | tr '\n' ':' | sed 's/:$//')
       echo "[INFO] PATH: $PATH"
}


echo "[INFO] compute optical flow"
source ${ROOT_DIR}/.venv/bin/activate
echo "[INFO] env: $VIRTUAL_ENV"
CMD_PREFIX=""
if [ "$(uname -s)" = "Linux" ]; then
       CMD_PREFIX="env CUDA_VISIBLE_DEVICES=0"
fi
if [ -d ${RESULT_FLOW_DIR} ] && [ -n "$(ls -A ${RESULT_FLOW_DIR}/*.mp4)" ]; then
       echo "[INFO] optical flow output files already exist. Skip computing optical flow."
else
       if [ ! -f ${ROOT_DIR}/reconstruct4D/ext/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth ]; then
              echo "[INFO] download pretrained model"
              mkdir -p ${ROOT_DIR}/reconstruct4D/ext/unimatch/pretrained
              wget https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/${FLOW_MODEL_NAME} -P ${ROOT_DIR}/reconstruct4D/ext/unimatch/pretrained
       fi

       mkdir -p ${RESULT_FLOW_DIR}
       export OMP_NUM_THREADS=1
       # to avoid CUDA out of memory error.
       export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
       ${CMD_PREFIX} python ${ROOT_DIR}/reconstruct4D/ext/unimatch/main_flow.py \
       --inference_dir ${INPUT} \
       --output_path ${RESULT_FLOW_DIR} \
       --resume ${ROOT_DIR}/reconstruct4D/ext/unimatch/pretrained/${FLOW_MODEL_NAME} \
       --padding_factor 32 \
       --upsample_factor 4 \
       --num_scales 2 \
       --attn_splits_list 2 8 \
       --corr_radius_list -1 4 \
       --prop_radius_list -1 1 \
       --reg_refine \
       --num_reg_refine 6 \
       --save_flo_flow
       echo "[INFO] save optical flow to ${RESULT_FLOW_DIR}"
       echo "[INFO] creating a flow movie"
       ffmpeg -framerate 30  -pattern_type glob -i "${RESULT_FLOW_DIR}/*.png" \
              -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${RESULT_FLOW_DIR}/flow.mp4
fi


echo "[INFO] run segmentation"
if [ -d ${RESULT_SEG_DIR} ] && [ -n "$(ls -A ${RESULT_SEG_DIR}/*.mp4)" ]; then
       echo "[INFO] segmentation output files already exist. Skip running segmentation."
else
       mkdir -p ${RESULT_SEG_DIR}
       case ${SEG_MODEL_TYPE} in
              "internimage")
                     echo "[INFO] activate InternImage conda env"
                     # to avoid error: "anaconda3/envs/internimage/etc/conda/activate.d/libblas_mkl_activate.sh: 
                     # line 1: MKL_INTERFACE_LAYER: unbound variable", we set +u.
                     set +eu
                     deactivate_allenvs
                     source $(conda info --base)/etc/profile.d/conda.sh
                     conda activate internimage
                     # eval "$(conda shell.bash activate internimage)"
                     set -eu
                     echo "[INFO] env: $CONDA_DEFAULT_ENV"
                     # swith python interpretor to the one in the conda env.       
                     PYTHON_INTERPRETER=$(which python)
                     echo "[INFO] python interpretor: $PYTHON_INTERPRETER"

                     if [ ! -f ${SEG_CHECKPOINT_DIR}/${SEG_MODEL_NAME} ]; then
                            echo "[INFO] download pretrained model"
                            mkdir -p ${SEG_CHECKPOINT_DIR}
                            # swith download link by the segmentation model
                            # add --content-disposition to prevent adding download=true in the downloded file name.
                            case ${SEG_MODEL_NAME} in
                                   "upernet_internimage_t_512_160k_ade20k.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91/upernet_internimage_t_512_160k_ade20k.pth?download=true -P ${SEG_CHECKPOINT_DIR};;
                                   "upernet_internimage_xl_640_160k_ade20k.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91/upernet_internimage_xl_640_160k_ade20k.pth -P ${SEG_CHECKPOINT_DIR};; 
                                   "upernet_internimage_h_896_160k_ade20k.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/fc1e4e7e01c3e7a39a3875bdebb6577a7256ff91/upernet_internimage_h_896_160k_ade20k.pth?download=true -P ${SEG_CHECKPOINT_DIR};;
                                   "mask_rcnn_internimage_t_fpn_1x_coco.pth")
                                          wget --content-disposition https://huggingface.co/OpenGVLab/InternImage/resolve/main/mask_rcnn_internimage_t_fpn_1x_coco.pth?download=true -P ${SEG_CHECKPOINT_DIR};;
                            *) echo "[ERROR] unknown segmentation model name: ${SEG_MODEL_NAME}"; exit 1;;
                            esac
                     fi

                     echo "[INFO] run segmentation using: ${SEG_MODEL_TYPE} ${SEG_TASK_TYPE}"
                     if [ "$SEG_TASK_TYPE" = "instance" ]; then
                            ${CMD_PREFIX} python ${ROOT_DIR}/reconstruct4D/ext/InternImage/detection/image_demo.py \
                            ${INPUT} \
                            ${ROOT_DIR}/reconstruct4D/ext/InternImage/detection/configs/coco/${SEG_MODEL_NAME%.*}.py  \
                            ${ROOT_DIR}/reconstruct4D/ext/InternImage/checkpoint_dir/det/${SEG_MODEL_NAME} \
                            --out ${RESULT_SEG_DIR}
                     elif [ "$SEG_TASK_TYPE" = "semantic" ]; then
                            ${CMD_PREFIX} python ${ROOT_DIR}/reconstruct4D/ext/InternImage/segmentation/image_demo.py \
                                   ${INPUT} \
                                   ${ROOT_DIR}/reconstruct4D/ext/InternImage/segmentation/configs/ade20k/${SEG_MODEL_NAME%.*}.py  \
                                   ${ROOT_DIR}/reconstruct4D/ext/InternImage/checkpoint_dir/seg/${SEG_MODEL_NAME} \
                                   --palette ade20k --out ${RESULT_SEG_DIR}
                     else
                            echo "[ERROR] unknown segmentation task type: ${SEG_TASK_TYPE}"
                            exit 1
                     fi
                     eval "$(conda shell.bash deactivate)";;
              "oneformer")
                     echo "[INFO] you choose segmentation: ${SEG_MODEL_TYPE} ${SEG_TASK_TYPE}\
                      The process will be done during moving object extraction."

       esac
fi


echo "[INFO] run extract moving objects"
source ${ROOT_DIR}/.venv/bin/activate
echo "[INFO] env: $VIRTUAL_ENV"
if [ -n "$(ls -A ${RESULT_MOVOBJ_DIR}/*.mp4)" ]; then
       echo "[INFO] moving objects output files already exist. So skip running extract moving objects."
       exit 0
fi
mkdir -p ${RESULT_MOVOBJ_DIR}
MOVOBJEXT_OPTS="--input_dir ${INPUT_DIR} \
       --flow_result_dir ${RESULT_FLOW_DIR} \
       --segment_model_type ${SEG_MODEL_TYPE} \
       --segment_model_name ${SEG_MODEL_NAME} \
       --segment_task_type ${SEG_TASK_TYPE} \
       --segment_result_dir ${RESULT_SEG_DIR} \
       --result_dir ${RESULT_MOVOBJ_DIR} \
       --skip_frames ${SKIP_FRAMES} \
       --ransac_all_inlier_estimation ${RANSAC_ALL_INLIER_ESTIMATION} \
       --foe_search_step ${FOE_SEARCH_STEP} \
       --num_ransac ${NUM_RANSAC} \
       --thre_moving_fraction_in_obj ${THRE_MOVING_FRACTION_IN_OBJ}\
       --loglevel ${LOG_LEVEL}"
if [ $LOG_LEVEL -ge 5 ]; then
       echo "[NOTE] Please press F5 to start debugging!"
       python -Xfrozen_modules=off -m debugpy --listen 5678 --wait-for-client ${ROOT_DIR}/reconstruct4D/extract_moving_objects.py ${MOVOBJEXT_OPTS}
else
       python ${ROOT_DIR}/reconstruct4D/extract_moving_objects.py ${MOVOBJEXT_OPTS}
fi


echo "[INFO] creating a segmentation movie (ffmpeg in InternImage conda env doesn't support libx264, so we create it here.)"
# for segmentation, the image file format is jpg or png. so detect it first.
IMG_EXT=
if [ $(ls -1 ${RESULT_SEG_DIR}/*.jpg 2>/dev/null | wc -l) != 0 ]; then
       IMG_EXT=jpg
elif [ $(ls -1 ${RESULT_SEG_DIR}/*.png 2>/dev/null | wc -l) != 0 ]; then
       IMG_EXT=png
else
       echo "[INFO] no jpg or png image file in ${RESULT_SEG_DIR}\
        So skip creating a segmentation movie."
fi
if [ $(ls -1 ${RESULT_SEG_DIR}/*.${IMG_EXT} 2>/dev/null | wc -l) != 0 ]; then
       ffmpeg -y -framerate 30  -pattern_type glob -i "${RESULT_SEG_DIR}/*.${IMG_EXT}" \
       -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${RESULT_SEG_DIR}/segmentation.mp4
fi

echo "[INFO] creating a final movie"
ffmpeg -y -framerate 30  -pattern_type glob -i "${RESULT_MOVOBJ_DIR}/*_result.png" \
       -vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p ${RESULT_MOVOBJ_DIR}/moving_object.mp4

if [ $LOG_LEVEL -ge 4 ]; then
       echo "[INFO] display the final movie"
       vlc ${RESULT_MOVOBJ_DIR}/moving_object.mp4
fi