import fire
import os


def stage1_tuning(model_name, instruction_data):
    cmd = "torchrun --standalone --nnodes=1 --nproc-per-node=1 ../dual-stage-tuning/main.py \
    --do_train \
    --train_file " + instruction_data + " \
    --validation_file " + instruction_data + " \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir /cache \
    --model_name_or_path " + model_name + " \
    --output_dir ../models/chatglm2/peft/instruction \
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
    --learning_rate 2e-2 \
    --pre_seq_len 128"

    os.system(cmd)


def stage2_tuning(model_name, traffic_data, task_name):
    cmd = "torchrun --standalone --nnodes=1 --nproc-per-node=1 ../dual-stage-tuning/main.py \
    --do_train \
    --train_file " + traffic_data + " \
    --validation_file " + traffic_data + " \
    --preprocessing_num_workers 10 \
    --prompt_column instruction \
    --response_column output \
    --overwrite_cache \
    --cache_dir /cache \
    --model_name_or_path " + model_name + " \
    --output_dir ../models/chatglm2/peft/" + task_name + " \
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
    --learning_rate 2e-2 \
    --pre_seq_len 128"

    os.system(cmd)


def model_update(model_name, traffic_data, task_name):
    assert task_name not in os.listdir("../models/chatglm2/peft/")
    stage2_tuning(model_name, traffic_data, task_name)


def model_insert(model_name, traffic_data, task_name):
    assert task_name in os.listdir("../models/chatglm2/peft/")
    os.mkdir(os.path.join("../models/chatglm2/peft/", task_name))
    stage2_tuning(model_name, traffic_data, task_name)


def main(model_name,
         tuning_data: str = None,
         adaptation_task: str = None,
         task_name: str = None,
         **kwargs):
    with open(os.path.join(tuning_data, "instructions/instruction.json"), "r", encoding="utf-8") as fin:
        instruction_data = fin.readlines()

    with open(os.path.join(tuning_data, "traffic/traffic.json"), "r", encoding="utf-8") as fin:
        traffic_data = fin.readlines()

    stage1_tuning(model_name, instruction_data)

    if adaptation_task == "update":
        model_update(model_name, traffic_data, task_name)
    elif adaptation_task == "register":
        model_insert(model_name, traffic_data, task_name)


if __name__ == "__main__":
    fire.Fire(main)
