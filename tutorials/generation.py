from scapy.all import *
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import json
import os
import fire

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def remove_tmp_files(path):
    files = os.listdir(path)
    for file in files:
        tf = os.path.join(path, file)
        os.remove(tf)


def build_header(header, hex_data):
    ip_pkt_len = 16 + int(len(hex_data) / 2)
    if header["proto"] == 6:
        pkt = Ether() / IP(src=header["src"], dst=header["dst"], len=ip_pkt_len) / TCP(sport=header["sport"], dport=header["dport"])
    else:
        pkt = Ether() / IP(src=header["src"], dst=header["dst"], len=ip_pkt_len) / UDP(sport=header["sport"], dport=header["dport"])
    hexdump(pkt)
    pkt.show()
    header = str(bytes_hex(pkt))[2:-1]
    return header


def packet_pcap_generation(header, hex_data, output_path):
    remove_tmp_files("tmp/packet_generation")

    header = build_header(header, hex_data)

    # Insert a fake Ether and IP layers in the packet
    with open("tmp/packet_generation/hexfile", "w") as fin:
        fin.write(header + hex_data)

    os.system("xxd -r -p tmp/packet_generation/hexfile tmp/packet_generation/binaryfile")
    os.system("od -Ax -tx1 -v tmp/packet_generation/binaryfile > tmp/packet_generation/hexdump")
    os.system("text2pcap -d tmp/packet_generation/hexdump " + os.path.join(output_path, "synthetic_packet.pcap"))


def flow_pcap_generation(header, hex_data, output_path):
    remove_tmp_files("tmp/packet_generation")
    remove_tmp_files("tmp/flow_generation")

    packets = hex_data[5:].split("<pck>")
    for i, packet in enumerate(packets):
        remove_tmp_files("tmp/packet_generation")

        header = build_header(header, packet)

        with open("tmp/packet_generation/hexfile", "w") as fin:
            fin.write(header + packet)

        os.system("xxd -r -p tmp/packet_generation/hexfile tmp/packet_generation/binaryfile")
        os.system("od -Ax -tx1 -v tmp/packet_generation/binaryfile > tmp/packet_generation/hexdump")
        os.system("text2pcap -d tmp/packet_generation/hexdump " + "tmp/flow_generation/" + str(i) + ".pcap")

    files = os.listdir("tmp/flow_generation/")
    paths = [os.path.join("tmp/flow_generation/", file) for file in files]

    os.system("mergecap -a " + " ".join(paths) + " -w " + os.path.join(output_path, "synthetic_flow.pcap") + " -F pcap")


def generation(header, payload, output_path):
    header = header.replace("'", "\"")
    # print(header)
    header = json.loads(header)
    packet_pcap_generation(header, payload, output_path)


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


def main(config, prompt: str = None, **kwargs):

    with open(config, "r", encoding="utf-8") as fin:
        config = json.load(fin)

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(config["model_path"], trust_remote_code=True, pre_seq_len=128)
    model = AutoModel.from_pretrained(config["model_path"], config=model_config, trust_remote_code=True)

    # Stage 1: task understanding
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"]["NLP"])
    model_nlp = load_model(model, ptuning_path)

    model_nlp = model_nlp.eval()

    response, history = model_nlp.chat(tokenizer, prompt, history=[])
    print(response)

    # Stage 2: task-specific traffic learning
    task = config["tasks"][response]
    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task][0])
    model_downstream = load_model(model, ptuning_path)
    model_downstream = model_downstream.eval()
    header, history = model_downstream.chat(tokenizer, prompt, history=[])
    print(header)

    ptuning_path = os.path.join(config["peft_path"], config["peft_set"][task][1])
    model_downstream = load_model(model, ptuning_path)
    model_downstream = model_downstream.eval()
    payload, history = model_downstream.chat(tokenizer, prompt, history=[])
    print(payload)

    generation(header, payload, output_path="")


if __name__ == "__main__":
    fire.Fire(main)
