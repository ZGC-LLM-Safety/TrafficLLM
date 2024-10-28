# @Author  : ZGCLLM

from pathlib import Path
from typing import Annotated, Union
import typer
from peft import PeftModelForCausalLM
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from PIL import Image
import torch
import pandas as pd
import json
import fire
from datetime import datetime
from tqdm import tqdm
import os
import re
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, classification_report
# import swanlab
# from swanlab.integration.huggingface import SwanLabCallback

app = typer.Typer(pretty_exceptions_show_locals=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
):
    model_dir = Path(model_dir).expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        with open(model_dir / 'adapter_config.json', 'r', encoding='utf-8') as file:
            config = json.load(file)
        model = AutoModel.from_pretrained(
            config.get('base_model_name_or_path'),
            trust_remote_code=trust_remote_code,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        model = PeftModelForCausalLM.from_pretrained(
            model=model,
            model_id=model_dir,
            trust_remote_code=trust_remote_code,
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=trust_remote_code,
        encode_special_tokens=True,
        use_fast=False
    )

    #model = model.to(device)
    return model, tokenizer

def td_evaluation(predict_responses, target_responses, label_file, evaluation_result_file):
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    preds = []
    labels = []

    error_response = []

    for index, (predict_response, target_response) in enumerate(zip(predict_responses, target_responses)):
        # response = predict_response.split(" ")[-1]
        response = predict_response
        
        # if ' ' not in predict_response:
        if not predict_response.isspace():
            # print(1)
            if response not in label_dict.keys():
                preds.append(len(label_dict.keys()))
                print(f"generated mistake labels: {response}, Index: {index}")
                error_response.append([response, index])
            else:
                preds.append(label_dict[response])
            labels.append(label_dict[target_response])
        else:
            if response not in label_dict.keys():
                preds.append(len(label_dict.keys()))
                print(f"generated mistake labels: {response}, Index: {index}")
                error_response.append([response, index])
            else:
                preds.append(label_dict[response])
            # labels.append(label_dict[target_response.split(" ")[-1]])
            labels.append(label_dict[target_response])
    
    possible_class = set(labels)
    pred_fixed = []
    for i in range(len(preds)):
        if preds[i] == len(label_dict.keys()):
            incorrect_class = possible_class - {labels[i]}
            replacement_class = next(iter(incorrect_class))
            pred_fixed.append(replacement_class)
        else:
            pred_fixed.append(preds[i])
    
    with open(evaluation_result_file, 'w', encoding="utf-8") as f:
        # print("Date Time ------ %s ------" % str(datetime.now()),file = f)
        if error_response:
            print("Error Responses: \n", file=f)
            print(*error_response, sep="\n", file=f)
        print("acc: %s" %  accuracy_score(labels, pred_fixed), file=f)
        print("precision: %s" % precision_score(labels, pred_fixed, average='weighted'), file=f)
        print("recall: %s" % recall_score(labels, pred_fixed, average='weighted'), file=f)
        print("f1: %s" % f1_score(labels, pred_fixed, average='weighted'), file=f)
        print("confusion matrix:\n %s" % confusion_matrix(labels, pred_fixed), file=f)
        print("classification report:\n %s" % classification_report(labels, pred_fixed, digits=4), file=f)
    with open(evaluation_result_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            print(line)

@app.command()
def main(
        model_dir: Annotated[str, typer.Argument(help='')],
        test_file: Annotated[str, typer.Argument(help='')],
        target_path: Annotated[str, typer.Argument(help='')],
        label_file: Annotated[str, typer.Argument(help='')],
        evaluation_result_file: Annotated[str, typer.Argument(help='')],
):

    model, tokenizer = load_model_and_tokenizer(model_dir)
    generate_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
        "eos_token_id": model.config.eos_token_id,
    }

    # Load TEST data by read file
    if test_file is not None:
        assert os.path.exists(test_file), f"Provided Test file does not exist {test_file}"
        with open(test_file, "r", encoding="utf-8") as fin:
            test_set = fin.readlines()
    else:
        print("No Test file provided. Exiting.")
        sys.exit(1)

    target_responses = [] # label
    label_df = pd.read_json(target_path, lines=True)[:100]
    for index, row in label_df.iterrows():
        target_responses.append(row['messages'][2]['content'])
    # print(target_responses)
    
    test_prompts = []
    for test_data in test_set:
        test_prompts.append(json.loads(test_data)["messages"])
    test_prompts = test_prompts[:100]

    predict_responses  = []

    for prompt_index in tqdm(range(len(test_prompts))):
        
        messages = test_prompts[prompt_index]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(model.device)
        
        generated_ids = model.generate(
                        inputs.input_ids,
                        max_new_tokens=1024
                                    )
        generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                            ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print("response: ",response)
        print("label: ",target_responses[prompt_index])
        predict_responses.append(response)
         
    td_evaluation(predict_responses, target_responses, label_file, evaluation_result_file)

if __name__ == '__main__':
    app()
