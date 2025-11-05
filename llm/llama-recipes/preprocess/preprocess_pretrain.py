from packet_data_preprocess import build_packet_data
from tqdm import tqdm
import argparse
import random
import json
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="raw dataset path", required=True)
    parser.add_argument("--output_path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output_name", type=str, help="output dataset name", required=True)

    args = parser.parse_args()
    return args


def build_dataset(path):
    build_data = []
    pcaps = os.listdir(path)
    for pcap in tqdm(pcaps):
        pcap_data = build_packet_data(os.path.join(path, pcap))
        build_data.extend(pcap_data)

    return build_data


def main():
    args = get_args()
    build_data = build_dataset(args.input)

    dataset = []
    for data in build_data:
        index = random.sample(range(0, len(data)), 1)[0]
        instruction = "Below is the first half of a traffic packet, please infer the second half of the traffic: "
        dataset.append(
            {
                "instruction": instruction + data[:index],
                "output": data[index:]
            }
        )
    random.shuffle(dataset)
    with open(os.path.join(args.output_path, args.output_name), "w", encoding="utf-8") as fin:
        for data in dataset:
            json.dump(data, fin)
            fin.write("\n")


if __name__ == "__main__":
    main()