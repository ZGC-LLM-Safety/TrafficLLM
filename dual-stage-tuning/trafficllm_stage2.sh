PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json \
    --validation_file ../datasets/ustc-tfc-2016/ustc-tfc-2016_detection_packet_train.json \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir /cache \
    --model_name_or_path ../models/chatglm2/chatglm2-6b \
    --output_dir ../models/chatglm2/peft/ustc-tfc-2016-detection-packet \
    --overwrite_output_dir \
    --max_source_length 1024 \
    --max_target_length 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 20000 \
    --logging_steps 10 \
    --save_steps 4000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN