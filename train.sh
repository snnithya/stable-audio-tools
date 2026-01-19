export REPO_ROOT="/home/zachary/code/stable-audio-tools/stable_audio_tools"

# export MODEL_CONFIG="/home/zachary/code/sat-sony/saos_base_model_config.json"
# export MODEL_CONFIG="/home/zachary/code/sat-sony/saos_debug.json"
# export MODEL_CONFIG="/home/zachary/code/sat-sony/saos_debug_start.json"
export MODEL_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/model_configs/txt2audio/saos_start_inpaint.json"
# export MODEL_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/model_configs/txt2audio/sao_short.json"
# export MODEL_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/model_configs/txt2audio/sao_short_inpaint.json"
# export MODEL_CONFIG="/home/zachary/code/sat-sony/sao_base_model_config.json"

# export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/ossl2_preextract.json"
# export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/jamendo_preextract.json"
export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/jamendo_ossl2_preextract.json"
# export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/jamendo_ossl2_preextract_longer.json"
# export DATASET_CONFIG="/home/zachary/code/sat-sony/stable_audio_tools/configs/dataset_configs/ossl2_preextract_long.json"




# export SCRIPT_CONFIG="$REPO_ROOT/defaults.ini"
# export PRETRAINED_CKPT_PATH="/home/zachary/code/stable-audio-tools/saos_vgm_jam_800k.ckpt"
export PRETRAINED_CKPT_PATH="/home/zachary/.cache/huggingface/hub/models--stabilityai--stable-audio-open-small/snapshots/dc620d91535857b72ebb59b4ca45978db6d417f5/base_model.ckpt"
# export PRETRAINED_CKPT_PATH="/home/zachary/.cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0/snapshots/f21265c1e2710b3bd2386596943f0007f55f802e/model.safetensors"
export SAVE_DIR="/home/zachary/checkpoints/s2s"

# export CKPT_PATH="/home/zachary/checkpoints/s2s/ossl2_experiments/p6vymayg/checkpoints/epoch=2-step=5000.ckpt"
export CKPT_PATH=""
export ENABLE_TORCH_COMPILE="1"
export USE_CHECKPOINTING="0"
export USE_LORA='false'

CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --model-config $MODEL_CONFIG \
        --dataset-config $DATASET_CONFIG \
        --config-file /home/zachary/code/stable-audio-tools/defaults.ini \
        --pretrained-ckpt-path $PRETRAINED_CKPT_PATH \
        --save-dir $SAVE_DIR \
        --num-workers 128 --strategy ddp_find_unused_parameters_true --accum-batches 1 \
        --batch-size 64 --checkpoint-every 5000 --precision 16-mixed --name "ossl2_experiments" # --use-lora $USE_LORA # --ckpt-path $CKPT_PATH \


