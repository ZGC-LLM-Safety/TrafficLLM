from flask import Flask, request, jsonify
import threading
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
import os
from flask_cors import CORS
import random
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '2'
app = Flask(__name__)
CORS(app)

with open("config.json", "r", encoding="utf-8") as fin:
    config = json.load(fin)

tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)

# create lock and counter
lock = threading.Lock()
counter = 0
MAX_CONCURRENT_REQUESTS = 10

def load_model(model, ptuning_path):
    if ptuning_path is not None:
        prefix_state_dict = torch.load(
            os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        model = model.half().cuda()
        model.transformer.prefix_encoder.float()

    return model


def preprompt(task, traffic_data):
    """Preprompts in LLMs for downstream traffic pattern learning"""
    prepromt_set = {
        "MTD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine which application "
               "category the encrypted beign or malicious traffic belongs to. The categories include 'BitTorrent, "
               "FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft,Cridex, Geodo, Htbot, Miuref, "
               "Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'.\n",
        "BND": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the BOTNET DETECTION TASK to determine which type of network the "
               "traffic belongs to. The categories include 'IRC, Neris, RBot, Virut, normal'.\n",
        "WAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "AAD": "Classify the given HTTP request into normal and abnormal categories. Each HTTP request will consist "
               "of three parts: method, URL, and body, presented in JSON format. If a web attack is detected in an "
               "HTTP request, please output an 'exception'. Only output 'abnormal' or 'normal', no additional output "
               "is required. The given HTTP request is as follows:\n",
        "EVD": "Given the following traffic data <packet> that contains protocol fields, traffic features, "
               "and payloads. Please conduct the encrypted VPN detection task to determine which behavior or "
               "application category the VPN encrypted traffic belongs to. The categories include 'aim, bittorrent, "
               "email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, vimeo, voipbuster, youtube'.\n",
        "TBD": "Given the following traffic data <packet> that contains protocol fields, traffic features, and "
               "payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine which behavior or application "
               "category the traffic belongs to under the Tor network. The categories include 'audio, browsing, chat, "
               "file, mail, p2p, video, voip'.\n"
    }

    prompt = prepromt_set[task] + traffic_data

    return prompt


def dual_stage_inference(human_instruction, traffic_data, model):

    # Stage 1: task understanding
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"]["NLP"])
    model_nlp = load_model(model, ptuning_path)

    model_nlp = model_nlp.eval()

    task_response, history = model_nlp.chat(tokenizer, human_instruction, history=[])
    print("Downstream task: " + task_response)

    # Stage 2: task-specific traffic learning
    task = config["tasks"][task_response]
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task])
    model_downstream = load_model(model, ptuning_path)

    model_downstream = model_downstream.eval()

    traffic_prompt = preprompt(task, traffic_data)
    final_response, history = model_downstream.chat(tokenizer, traffic_prompt, history=[])
    print("Predicted result: " + final_response)

    return task_response, final_response


@app.route('/api/conversation', methods=['POST'])
def conversation():
    global counter

    if counter >= MAX_CONCURRENT_REQUESTS:
        response = {'msg': '请求过载，请稍后再试', 'status': 'fail'}
        return jsonify(response)

    with lock:
        counter += 1

    try:
        human_instruction = request.json['human_instruction']
        traffic_data = request.json['traffic_data']
        task_response, final_response = dual_stage_inference(human_instruction, traffic_data, model)

        # inference_results = beam_search.generate([{'question': question, '<ans>': ''}], max_length=500, repetition_penalty=1.5) # 1.1
        # ans = inference_results[0]['<ans>']

        response = {'msg': '请求成功', 'status': 'success', 'task_response': task_response, 'final_response': final_response}
        return jsonify(response)

    finally:
        with lock:
            counter -= 1



if __name__ == '__main__':
    print("TrafficLLM service has been started")
    app.run(host="0.0.0.0", port=8877)

