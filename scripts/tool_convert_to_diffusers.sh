DIFFUSERS_PATH=${1:-/projects/diffusers}
CHECKPOINT_PATH=${2:-/projects/ControlNet/models/ckpt}
ORIGINAL_CONFIG_FILE=${3:-/projects/ControlNet/models/cldm_v21.yaml}
DUMP_PATH=${4:-/projects/ControlNet/models/dump}
DEVICE=${5:-cpu}

python ${DIFFUSERS_PATH}/scripts/convert_original_controlnet_to_diffusers.py \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --original_config_file ${ORIGINAL_CONFIG_FILE} \
    --dump_path ${DUMP_PATH} \
    --device ${DEVICE} \
    --to_safetensor
