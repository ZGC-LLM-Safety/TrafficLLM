from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, classification_report
import json
import time
import os
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import datetime
import torch
import shutil



device = "cuda" # or "cpu"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_path = "/mnt/data/cty/models/deepseek-r1-1.5b"
output_dir = "/mnt/data/cty/works/mistral-finetune/base_model/finetuned_model/deepseek_1.5b"
# adapter_model_name = ""

data_path = '/mnt/data/cty/works/mistral-finetune/route_data/'
dataset_name = "ustc-tfc-2016"
expert_name = "RP"



peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_token"],
    task_type="CAUSAL_LM"
)


def evaluation(predict_responses, target_responses, label_dict):
    preds = []
    labels = []
    for predict_response, target_response in zip(predict_responses, target_responses):
        if "。" in predict_response and "。" in target_response:
            predict_response = predict_response[:-1]
            target_response = target_response[:-1]
        if ' ' not in predict_response:
            if predict_response not in label_dict.keys():
                preds.append(len(label_dict.keys()))
                print("generated mistake labels:", predict_response)
            else:
                preds.append(label_dict[predict_response])
            labels.append(label_dict[target_response])
        else:
            if predict_response.split(" ")[-1] not in label_dict.keys():
                preds.append(len(label_dict.keys()))
                print("generated mistake labels:", predict_response.split(" ")[-1])
            else:
                preds.append(label_dict[predict_response.split(" ")[-1]])
            labels.append(label_dict[target_response.split(" ")[-1]])

    print("acc:", accuracy_score(labels, preds))
    print("precision:", precision_score(labels, preds, average='weighted'))
    print("recall:", recall_score(labels, preds, average='weighted'))
    print("f1:", f1_score(labels, preds, average='weighted'))
    print("confusion matrix:\n", confusion_matrix(labels, preds))
    print("classification report:\n", classification_report(labels, preds))



class EvaluationCallback(TrainerCallback):
    def __init__(self, test_dataset, tokenizer):
        # ... existing code ...
        self.test_dataset = test_dataset['train']
        self.label_dict = data_path + dataset_name + '/' + expert_name + '/label.jsonl'
        self.tokenizer = tokenizer
        self.epoch = 0
        # Add storage for metrics
        self.eval_losses = []
        self.train_losses = []
        self.epochs = []
        self.predicts = []
        self.labels = []


    def on_epoch_end(self, args, state, control, model, **kwargs):
        print(f"\nEvaluating model after epoch... {self.epoch}")

        # Store the current training loss
        if state.log_history:
            latest_loss = state.log_history[-1].get('loss')
            if latest_loss is not None:
                self.train_losses.append(latest_loss)
                self.epochs.append(self.epoch)

        model.eval()
        total_eval_loss = 0
        num_eval_samples = 0

        with torch.no_grad():
            # for i in range(min(500, len(self.test_dataset))):
            for i in range(len(self.test_dataset)):
                user_input = self.test_dataset[i]['messages'][0]['content']
                label = self.test_dataset[i]['messages'][1]['content']
                # system_prompt = self.test_dataset[i]['messages'][0]['content']

                # Prepare the input for the model
                messages = [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
                chat_template_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )

                model_inputs = self.tokenizer([chat_template_text],
                                              return_tensors="pt",
                                              ).to(model.device)

                # Generate response
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    # temperature=0.7,
                    # top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=False
                )

                # Calculate loss
                outputs = model(**model_inputs, labels=model_inputs.input_ids)
                loss = outputs.loss.item()
                total_eval_loss += loss
                num_eval_samples += 1

                # Decode and print response
                generated_text = self.tokenizer.batch_decode(
                    [output_ids[len(input_ids):] for input_ids, output_ids in
                     zip(model_inputs.input_ids, generated_ids)],
                    skip_special_tokens=True
                )[0].split(">")[-1]

                # print(f"\nTest sample {i + 1}:")
                # print(f"Input: {user_input}")
                # print(f"Output: {generated_text}")
                # print(f"Loss: {loss}")
                # print("-" * 50)

                self.predicts.append(generated_text)
                self.labels.append(label)

        with open(self.label_dict, "r") as fin:
            label_dict = json.load(fin)
        evaluation(self.predicts, self.labels, label_dict)

        # Calculate average evaluation loss
        avg_eval_loss = total_eval_loss / num_eval_samples if num_eval_samples > 0 else 0
        self.eval_losses.append(avg_eval_loss)

        # Save metrics
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'current_epoch': self.epoch,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Create directory if it doesn't exist
        os.makedirs('losses', exist_ok=True)

        # Save metrics to JSON
        with open('losses/training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Create and save plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(len(self.eval_losses)), self.eval_losses, 'r-', label='Evaluation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Evaluation Loss Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig('/mnt/data/cty/works/mistral-finetune/images/training_progress.png')
        plt.close()

        self.epoch += 1
        model.train()


# def evaluation():
#     print("\n\n ----------------test model after finetuning ----------------")
#
#     # Free up GPU memory
#
#     model_finetuned = AutoModelForCausalLM.from_pretrained(model_name).to(device)
#     model_finetuned = PeftModel.from_pretrained(model_finetuned, adapter_model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#
#     # Load the JSON data from the file
#     with open('data/phone_log_fake.json', 'r') as file:
#         data = json.load(file)
#
#     # Iterate over the first 10 user inputs
#     for i, entry in enumerate(data[:10]):
#         user_input = entry['messages'][0]['content']  # Extract the user input message
#         print(f"Testing input {i + 1}: {user_input}")
#
#         # Prepare the input for the model
#         messages = [
#             {"role": "system", "content": "You are helpful assistant to analysis cisco ip phone log.\
#                     given the error pattern and type, you need to find the root cause."},
#             {"role": "user", "content": user_input}
#         ]
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#         model_inputs = tokenizer([text], return_tensors="pt").to(model_finetuned.device)
#         generated_ids = model_finetuned.generate(
#             **model_inputs,
#             max_new_tokens=256
#         )
#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]
#         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#         print(f"Output: {response}\n")
#         print("----------------------------------------")
#
#     print("\n\n ----------------test model after finetuning end ----------------")



if __name__ == "__main__":

    print("Loading dataset...")

    # Load dataset from JSON file
    dataset = load_dataset('json', data_files=data_path + dataset_name + '/' + expert_name + '/train.jsonl', split='train')
    test_dataset = load_dataset('json', data_files=data_path + dataset_name + '/' + expert_name + '/test.jsonl')

    print(type(dataset))
    print(dataset)
    # print(json.dumps(dataset[0], indent=2))
    print(f"\ndataset[0].keys(): {dataset[0].keys()}")
    print(f"\ndataset[0]['messages'][0]['content']:\n {dataset[0]['messages'][1]['content']}")

    # Split your dataset into train and test
    # train_test_dataset = dataset.train_test_split(test_size=0.02)
    # print(f"\ntrain_test_dataset['test']:\n {train_test_dataset['test']}")
    # print(json.dumps(train_test_dataset['test'][1], indent=2))

    print(json.dumps(dataset[0], indent=2))


    time_start = time.time()
    if not os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  trust_remote_code=True,
                                                  padding_side='right')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  trust_remote_code=True,
                                                  padding_side='right')
        model = AutoModelForCausalLM.from_pretrained(model_path)

    time_end = time.time()
    print(f"加载模型时间: {time_end - time_start} 秒")

    # print(train_test_dataset['test'][0]['messages'][1]['content'])

    print(output_dir)

    # Remove output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Removed existing output directory: {output_dir}")

    # Create fresh output directory
    os.makedirs(output_dir)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=test_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=1,  # Reduced from 5
            per_device_train_batch_size=2,  # Add small batch size
            gradient_accumulation_steps=4,  # Add gradient accumulation
            learning_rate=2e-4,  # Add explicit learning rate
            weight_decay=0.01,  # Add weight decay for regularization
            logging_steps=1,  # More frequent logging for small dataset
            save_steps=5,  # More frequent saving
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_train_loss",
            greater_is_better=False,
            warmup_steps=10,  # Add warmup steps
        ),
        peft_config=peft_config,
        callbacks=[EvaluationCallback(test_dataset, tokenizer)]
    )

    # After creating the trainer (after in[9] and before in[13])

    # Calculate and print trainable parameters
    trainable_params = 0
    all_params = 0

    for _, param in trainer.model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"All parameters: {all_params:,}")
    print(f"Percentage of parameters being trained: {100 * trainable_params / all_params:.2f}%")

    train_output = trainer.train()

    # evaluation()

