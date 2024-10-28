#!/bin/bash

# conda activate GLM4

NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=1

PHASE="GLM4-TASK-tuning"
LOG_PATH="Logs/${PHASE}/"

mkdir -p "$LOG_PATH"

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune.py \
../datas/cs_http \
../models--THUDM--glm-4-9b-chat \
configs/lora.yaml
> "${LOG_PATH}/test.log" 2>&1

# configs/ptuning_v2.yaml