
## llama2-7B

* The original version: llama-recipe (https://github.com/meta-llama/llama-recipes).
* Recently, we update the classification head of llama2 to replace the autoregressive head. The original loss function has been replaced with cross decay. Additionally, Lora has been used for model fine-tuning.

### 1. Data Preprocessing
```
python preprocess/preprocess_dataset.py --input /mnt/data/cty/data/traffic_data/ustc-tfc-2016 --dataset_name ustc-tfc-2016 --traffic_task detection --granularity packet --output_path preprocess/build_datasets/detection/ustc-tfc-2016/ --output_name ustc-tfc-2016
```

### 2. Model Training and Evaluation
```
python training_script.py --data_path=/mnt/data/cty/works/chatglm2-6b/preprocess/build_datasets/detection/ustc-tfc-2016-new --eval_batch_size=64 --lora_alpha=64 --lora_bias=none --lora_dropout=0.07398992286835075 --lora_rank=16 --lr=0.02 --max_length=512 --model_name=/mnt/data/cty/models/llama2/models_hf/7B --num_epochs=20 --output_path=llama-2-7b-hf-lora-token-classification --train_batch_size=16 --weight_decay=0.005808018858604934 --set_pad_id
```

## deepseek-R1

* Download model and install requirements based on the original version: deepseek-r1 (https://github.com/deepseek-ai/DeepSeek-R1).

### 1. Model Training
```
python deepseek-r1.py
```

