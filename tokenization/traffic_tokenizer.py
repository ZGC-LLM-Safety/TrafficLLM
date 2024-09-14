from transformers import AutoTokenizer
import sentencepiece as spm
import json
from tqdm import tqdm


model_name = "/Your/ChatGLM2/MODEL_PATH/"


def build_training_data():
    data_path = "/Your/ORIGINAL/DATA_PATH/"
    write_path = "dataset.txt"
    dataset = []
    with open(data_path, "r", encoding="utf-8") as fin:
        for line in fin.readlines():
            dataset.append(json.loads(line)["instruction"])
            dataset.append(json.loads(line)["output"])

    with open(write_path, "w", encoding="utf-8") as fin:
        for data in dataset:
            fin.write(data + "\n")


def train():
    spm.SentencePieceTrainer.Train(
        input="dataset.txt",
        model_prefix="tokenizer",
        vocab_size=64794,
        user_defined_symbols=['foo', 'bar'],
        character_coverage=1.0,
        model_type="bpe",
    )


def tokenizer_comparing():
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_ids)
    print(tokenizer.special_tokens_map)
    with open("dataset.txt", "r", encoding="utf-8") as fin:
        dataset = fin.readlines()
    count = 0
    len_chatglm2 = 0
    len_trafficllm = 0
    for text in tqdm(dataset[:100]):
        if len(text) < 10:
            continue
        count += 1
        len_chatglm2 += len(tokenizer.tokenize(text))
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load("./tokenizer.model")
        len_trafficllm += len(sp_model.EncodeAsPieces(text))

    print("chatglm2 token len", len_chatglm2 / count)
    print("trafficllm token len", len_trafficllm / count)


if __name__ == "__main__":
    build_training_data()
    train()
    # tokenizer_comparing()