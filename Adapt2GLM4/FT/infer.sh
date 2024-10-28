#!/bin/bash

# conda activate GLM4

PHASE="TASK_NAME" # The task name for traffic analysis
LOG_PATH="Logs/${PHASE}/"

mkdir -p "$LOG_PATH"

CUDA_VISIBLE_DEVICES=4 python inference.py ../output/checkpoint-3000 ../datas/test.jsonl ../datas/test_with_label.jsonl ../datas/label.json ../output/evaluation_metric_result
> "${LOG_PATH}/test.log" 2>&1
