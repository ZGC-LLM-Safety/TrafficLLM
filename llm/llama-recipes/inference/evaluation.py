import fire
import torch
import os
import sys
import time
import json
from tqdm import tqdm
from typing import List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import LlamaTokenizer
from model_utils import load_model, load_peft_model, load_llama_from_config


def test_set_to_prompt(test_set):

    test_prompts = []
    target_responses = []

    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. " \
                   "Write a response that appropriately completes the request.\n\n" \
                   "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    for test_data in test_set:
        test_prompts.append(prompt.format_map(test_data))
        target_responses.append(test_data["output"])

    return test_prompts, target_responses


def td_evaluation(predict_responses, target_responses, label_file):
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    preds = []
    labels = []
    for predict_response, target_response in zip(predict_responses, target_responses):
        if predict_response.split(" ")[-1][:-1] not in label_dict.keys():
            preds.append(len(label_dict.keys()))
            print("generated mistake labels:", predict_response.split(" ")[-1][:-1])
        else:
            preds.append(label_dict[predict_response.split(" ")[-1][:-1]])
        labels.append(label_dict[target_response.split(" ")[-1][:-1]])

    print("acc:", accuracy_score(labels, preds))
    print("precision:", precision_score(labels, preds, average='weighted'))
    print("recall:", recall_score(labels, preds, average='weighted'))
    print("f1:", f1_score(labels, preds, average='weighted'))


def tg_evaluation(predict_responses, target_responses):
    for i, (predict_response, target_response) in enumerate(zip(predict_responses, target_responses)):
        print("Q" + str(i) + ":\n")
        print("predict response:\n", predict_response + "\n")
        print("target response:\n", target_response + "\n")


def tu_evaluation():
    pass


def main(
    model_name,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    test_file: str=None,
    label_file: str=None,
    traffic_task: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_salesforce_content_safety: bool=True, # Enable safety check with Salesforce safety flan t5
    max_padding_length: int=None, # the max padding length to be used with tokenizer padding the prompts.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    **kwargs
):
    if test_file is not None:
        assert os.path.exists(test_file), f"Provided Test file does not exist {test_file}"
        with open(test_file, "r", encoding="utf-8") as fin:
            test_set = json.load(fin)
    else:
        print("No Test file provided. Exiting.")
        sys.exit(1)

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {

            "pad_token": "<PAD>",
        }
    )

    model.resize_token_embeddings(model.config.vocab_size + 1)

    test_prompts, target_responses = test_set_to_prompt(test_set)
    test_prompts = test_prompts[-200:]
    target_responses = target_responses[-200:]

    predict_responses = []

    for test_prompt in tqdm(test_prompts):

        batch = tokenizer(test_prompt, padding='max_length', truncation=True, max_length=max_padding_length,
                          return_tensors="pt")

        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
        e2e_inference_time = (time.perf_counter() - start) * 1000
        # print(f"the inference time is {e2e_inference_time} ms")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # print(f"Model output:\n{output_text}")
        predict_responses.append(output_text)

    if traffic_task == "detection":
        td_evaluation(predict_responses, target_responses, label_file)

    elif traffic_task == "generation":
        tg_evaluation(predict_responses, target_responses)

    elif traffic_task == "understanding":
        tu_evaluation()


if __name__ == "__main__":
    fire.Fire(main)
