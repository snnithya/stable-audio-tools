# #!/usr/bin/env bash
# #SBATCH --partition=ct
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --job-name=saos-s2s-train
# #SBATCH --output=/group2/ct/zack-novack/sat-sony/logs/%x_%j.log
# #SBATCH --account=ct
# #SBATCH --gres=gpu:h100:4
# #SBATCH --requeue

# export WANDB_API_KEY="b20f5b160a3857e09e5172ac958bbfc6b2d018ae"
# echo "[INFO] JobID=${SLURM_JOBID} scratch=${SLURM_SCRATCH}"

# # ORIG_SHARDS=/group2/ct/text_to_audio/dataset/OGameData/feature/dinov2_vits14_reg_fps_30_scale_854_480_float16/webdataset/pca_60
# ORIG_CSV=/group2/ct/text_to_audio/dataset/audiocaps/train.csv

# # echo "[INFO] Copy shards -> $SLURM_SCRATCH/shards"
# # srun cp -rf --no-preserve=mode "$ORIG_SHARDS" "$SLURM_SCRATCH/shards"

# echo "[INFO] Copy train CSV -> $SLURM_SCRATCH"
# srun cp -f "$ORIG_CSV" "$SLURM_SCRATCH/"

# SCRATCH_SHARDS="$SLURM_SCRATCH/shards"

# echo "copy done"

# export TRITON_CACHE_DIR=${SLURM_SCRATCH}/triton_cache
# mkdir -p "$TRITON_CACHE_DIR"

# export SIF_DIR="/scratch2/zachary.novack"
# export SINGULARITY_IMAGE="${SIF_DIR}/mar_v4.sif"
# export MOUNT_PATHS="/group2,/scratch2"

export REPO_ROOT="/home/zachary/code/stable-audio-tools/stable_audio_tools"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_rms_pitch.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_rms.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_rms_centroid.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_rms_pitch_centroid.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_rms2_pitch_centroid.json"
# export MODEL_CONFIG="/home/zachary/code/sat-sony/saos_base_model_config.json"
# export MODEL_CONFIG="/home/zachary/code/sat-sony/saos_debug.json"
export MODEL_CONFIG="/home/zachary/code/sat-sony/saos_debug_start.json"
# export MODEL_CONFIG="/home/zachary/code/sat-sony/sao_base_model_config.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_rms2_pitch_centroid_adapt_t2.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_adapt_t2.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base_pitch.json"
# export MODEL_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/model_configs/txt2audio/saos_base.json"


# export DATASET_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/dataset_configs/crusoe_audiocaps2_preextract.json"
# export DATASET_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/dataset_configs/crusoe_audiocaps2_preextract_10s.json"
# export DATASET_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/dataset_configs/crusoe_y8m_preextract_10s.json"
# export DATASET_CONFIG="/group2/ct/zack-novack/sat-sony/stable_audio_tools/configs/dataset_configs/crusoe_wavcaps_preextract_10s.json"
# export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/ossl2_preextract.json"
# export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/jamendo_preextract.json"
# export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/jamendo_ossl2_preextract.json"
export DATASET_CONFIG="/home/zachary/code/stable-audio-tools/stable_audio_tools/configs/dataset_configs/jamendo_ossl2_preextract_longer.json"
# export DATASET_CONFIG="/home/zachary/code/sat-sony/stable_audio_tools/configs/dataset_configs/ossl2_preextract_long.json"




# export SCRIPT_CONFIG="$REPO_ROOT/defaults.ini"
export PRETRAINED_CKPT_PATH="/home/zachary/.cache/huggingface/hub/models--stabilityai--stable-audio-open-small/snapshots/dc620d91535857b72ebb59b4ca45978db6d417f5/base_model.ckpt"
# export PRETRAINED_CKPT_PATH="/home/zachary/.cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0/snapshots/f21265c1e2710b3bd2386596943f0007f55f802e/model.safetensors"
export SAVE_DIR="/home/zachary/checkpoints/s2s"
# export PL_LOGGER="wandb"
# export WANDB_CACHE_DIR="/group2/ct/zack-novack/cache/wandb"
# export WANDB_BASE_URL="https://api.wandb.ai"
# # set random port
# export MASTER_PORT=$(shuf -i 20000-30000 -n 1) 
export CKPT_PATH="/home/zachary/checkpoints/s2s/ossl2_experiments/l0hvqvix/checkpoints/epoch=352-step=810000.ckpt"
# export CKPT_PATH=""
export ENABLE_TORCH_COMPILE="0"

CUDA_VISIBLE_DEVICES=0,1 python train.py \
        --model-config $MODEL_CONFIG \
        --dataset-config $DATASET_CONFIG \
        --config-file /home/zachary/code/stable-audio-tools/defaults.ini \
        --pretrained-ckpt-path $PRETRAINED_CKPT_PATH \
        --ckpt-path $CKPT_PATH \
        --save-dir $SAVE_DIR \
        --num-workers 128 --strategy ddp_find_unused_parameters_true --accum-batches 1 \
        --batch-size 64 --checkpoint-every 5000 --precision 16-mixed --name "ossl2_experiments"


