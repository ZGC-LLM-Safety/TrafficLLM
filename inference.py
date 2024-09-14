from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import fire
import os
import torch
from transformers import AutoConfig
import sys
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(model_name,
         prompt: str = None,
         ptuning_path: str = None,
         **kwargs):

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

    test_prompts = [prompt]

    for test_prompt in tqdm(test_prompts):

        response, history = model.chat(tokenizer, test_prompt, history=[])
        print(response)


if __name__ == "__main__":
    fire.Fire(main)
