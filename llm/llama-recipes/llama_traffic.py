from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, Trainer, TrainerCallback
from copy import deepcopy
import evaluate
import numpy as np
import os
import torch
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROJECT"] = "peft_tweets" # log to your project
os.environ["WANDB_LOG_MODEL"] = "all" # log your models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MAX_LEN = 512
llama_checkpoint = "/mnt/data/cty/models/llama2/models_hf/7B"
# dataset = load_dataset("mehdiiraqui/twitter_disaster")
dataset = load_from_disk("traffic_preprocess/datasets/twitter_disaster")

data = dataset['train'].train_test_split(train_size=0.8, seed=42)
data['val'] = data.pop("test")
data['test'] = dataset['test']

data['train'].to_pandas().info()
data['test'].to_pandas().info()

pos_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().target.value_counts()[1])
neg_weights = len(data['train'].to_pandas()) / (2 * data['train'].to_pandas().target.value_counts()[0])
POS_WEIGHT, NEG_WEIGHT = (1.1637114032405993, 0.8766697374481806)

max_char = data['train'].to_pandas()['text'].str.len().max()
max_words = data['train'].to_pandas()['text'].str.split().str.len().max()


llama_tokenizer = AutoTokenizer.from_pretrained(llama_checkpoint, add_prefix_space=True)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id
llama_tokenizer.pad_token = llama_tokenizer.eos_token

def llama_preprocessing_function(examples):
    return llama_tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

col_to_delete = ['id', 'keyword','location', 'text']
llama_tokenized_datasets = data.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
llama_tokenized_datasets = llama_tokenized_datasets.rename_column("target", "label")
llama_tokenized_datasets.set_format("torch")

llama_tokenized_datasets_train = llama_tokenized_datasets["train"]
llama_tokenized_datasets_val = llama_tokenized_datasets["val"]

# Data collator for padding a batch of examples to the maximum length seen in the batch
llama_data_collator = DataCollatorWithPadding(tokenizer=llama_tokenizer)

llama_model =  AutoModelForSequenceClassification.from_pretrained(
  pretrained_model_name_or_path=llama_checkpoint,
  num_labels=2,
  device_map="auto",
  offload_folder="offload",
  trust_remote_code=True
)

llama_model.config.pad_token_id = llama_model.config.eos_token_id

llama_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=16, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)

llama_model = get_peft_model(llama_model, llama_peft_config)
llama_model.print_trainable_parameters()

llama_model.to(device)

# if llama_model.device.type != 'cuda':
#     llama_model = llama_model.to('cuda')

def compute_metrics(eval_pred):
    # All metrics are already predefined in the HF `evaluate` package
    precision_metric = evaluate.load("metric/precision.py") # precision
    recall_metric = evaluate.load("metric/recall.py")
    f1_metric= evaluate.load("metric/f1.py")
    accuracy_metric = evaluate.load("metric/accuracy.py")

    logits, labels = eval_pred # eval_pred is the tuple of predictions and labels returned by the model
    predictions = np.argmax(logits, axis=-1)
    precision = precision_metric.compute(predictions=predictions, references=labels)["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels)["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels)["f1"]
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    # The trainer is expecting a dictionary where the keys are the metrics names and the values are the scores.
    return {"precision": precision, "recall": recall, "f1-score": f1, 'accuracy': accuracy}

class WeightedCELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        # Get model's predictions
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weights, pos_weights], device=model.device, dtype=logits.dtype))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

llama_model = llama_model.cuda()

lr = 1e-4
batch_size = 8
num_epochs = 5
training_args = TrainingArguments(
    output_dir="llama-lora-token-classification",
    learning_rate=lr,
    lr_scheduler_type= "constant",
    warmup_ratio= 0.1,
    max_grad_norm= 0.3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.001,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    fp16=True,
    gradient_checkpointing=True,
)

llama_trainer = WeightedCELossTrainer(
    model=llama_model,
    args=training_args,
    train_dataset=llama_tokenized_datasets_train,
    eval_dataset=llama_tokenized_datasets_val,
    data_collator=llama_data_collator,
    compute_metrics=compute_metrics
)

llama_trainer.add_callback(CustomCallback(llama_trainer))
llama_trainer.train()
