from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,  confusion_matrix, classification_report
from tqdm import tqdm
import fire
import os
import torch
from transformers import AutoConfig
import sys
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_set_to_prompt(test_set):

    test_prompts = []
    target_responses = []

    for test_data in test_set:
        test_prompts.append(json.loads(test_data)["instruction"])
        target_responses.append(json.loads(test_data)["output"])

    return test_prompts, target_responses


def td_evaluation(predict_responses, target_responses, label_file):
    with open(label_file, "r", encoding="utf-8") as fin:
        label_dict = json.load(fin)

    preds = []
    labels = []
    for predict_response, target_response in zip(predict_responses, target_responses):
        print(predict_response)
        if ' ' not in predict_response:
            print(1)
            if predict_response not in label_dict.keys():
                preds.append(len(label_dict.keys()))
                print("generated mistake labels:", predict_response)
            else:
                preds.append(label_dict[predict_response])
            labels.append(label_dict[target_response])
        else:
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
    print("confusion matrix:\n", confusion_matrix(labels, preds))
    print("classification report:\n", classification_report(labels, preds))


def tg_evaluation(predict_responses, target_responses, test_prompts):
    write_path = "generation.json"
    dataset = {}
    for i, (predict_response, target_response, test_prompt) in enumerate(zip(predict_responses, target_responses, test_prompts)):
        # print("Q" + str(i) + ":\n")
        # print("predict response:\n", predict_response + "\n")
        # print("target response:\n", target_response + "\n")
        label = test_prompt.split(" ")[-2]
        if label not in dataset.keys():
            dataset[label] = []
        dataset[label].append(predict_response)
    with open(write_path, "w", encoding="utf-8") as fin:
        json.dump(dataset, fin, indent=4, separators=(',', ': '))


def main(model_name,
         test_file: str = None,
         label_file: str = None,
         traffic_task: str = None,
         ptuning_path: str = None,
         **kwargs):

    if test_file is not None:
        assert os.path.exists(test_file), f"Provided Test file does not exist {test_file}"
        with open(test_file, "r", encoding="utf-8") as fin:
            test_set = fin.readlines()
            # test_set = json.load(fin)
    else:
        print("No Test file provided. Exiting.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if ptuning_path is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float()

    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()

    model = model.eval()

    test_prompts, target_responses = test_set_to_prompt(test_set)
    test_prompts = test_prompts[:1000]
    target_responses = target_responses[:1000]

    predict_responses = []

    for test_prompt in tqdm(test_prompts):
        if traffic_task == "detection":
            response, history = model.chat(tokenizer, test_prompt, history=[], top_p=0.85, temperature=0.1)
        elif traffic_task == "generation":
            response, history = model.chat(tokenizer, test_prompt, history=[])
        else:
            response = None
        print("response:", response)
        predict_responses.append(response)

    if traffic_task == "detection":
        td_evaluation(predict_responses, target_responses, label_file)
    elif traffic_task == "generation":
        tg_evaluation(predict_responses, target_responses, test_prompts)


if __name__ == "__main__":
    fire.Fire(main)
