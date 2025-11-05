# llama2-7b

## 原始训练预测版本见llama-recipe（https://github.com/meta-llama/llama-recipes）；
## 最近更新利用llama2的分类头取代自回归头，使用交叉殇替代原损失函数，并使用lora进行模型微调；

1. 数据预处理
python preprocess/preprocess_dataset.py --input /mnt/data/cty/data/traffic_data/ustc-tfc-2016 --dataset_name ustc-tfc-2016 --traffic_task detection --granularity packet --output_path preprocess/build_datasets/detection/ustc-tfc-2016/ --output_name ustc-tfc-2016

2. 模型训练及验证集评估
python training_script.py --data_path=/mnt/data/cty/works/chatglm2-6b/preprocess/build_datasets/detection/ustc-tfc-2016-new --eval_batch_size=64 --lora_alpha=64 --lora_bias=none --lora_dropout=0.07398992286835075 --lora_rank=16 --lr=0.02 --max_length=512 --model_name=/mnt/data/cty/models/llama2/models_hf/7B --num_epochs=20 --output_path=llama-2-7b-hf-lora-token-classification --train_batch_size=16 --weight_decay=0.005808018858604934 --set_pad_id