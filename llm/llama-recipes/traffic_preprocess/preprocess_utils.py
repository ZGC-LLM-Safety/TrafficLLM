from flow_data_preprocess import build_flow_data
from packet_data_preprocess import build_packet_data
import random
import json
import os


MAX_SAMPLING_NUMBER = 5000
TRAINING_SAMPLE_RATIO = 0.8


def split_dataset(build_data, sampling=True):
    random.shuffle(build_data)
    if sampling is True:
        train_nb = int(min(MAX_SAMPLING_NUMBER, len(build_data)) * TRAINING_SAMPLE_RATIO)
        test_nb = int(min(MAX_SAMPLING_NUMBER, len(build_data)) * (1 - TRAINING_SAMPLE_RATIO))
    else:
        train_nb = int(len(build_data) * TRAINING_SAMPLE_RATIO)
        test_nb = int(len(build_data) * (1 - TRAINING_SAMPLE_RATIO))
    train_data = build_data[:train_nb]
    test_data = build_data[train_nb:train_nb + test_nb]

    return train_data, test_data


def write_dataset(dataset, output_path):
    random.shuffle(dataset)
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(dataset, fin, indent=4, separators=(',', ': '))


def write_labels(labels, output_path):
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(label_dict, fin, indent=4, separators=(',', ': '))


def build_dataset(args, path, file):
    build_data = []
    files_path = os.path.join(path, file)
    pcaps = os.listdir(files_path)
    for pcap in pcaps:
        if args.granularity == "flow":
            pcap_data = build_flow_data(os.path.join(files_path, pcap))
        else:
            pcap_data = build_packet_data(os.path.join(files_path, pcap))
        build_data.extend(pcap_data)

    train_data, test_data = split_dataset(build_data)
    return train_data, test_data


def save_dataset(args, train_dataset, test_dataset):
    if args.granularity == "flow":
        write_dataset(train_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_flow_train.json"))
        write_dataset(test_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_flow_test.json"))
    else:
        write_dataset(train_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_packet_train.json"))
        write_dataset(test_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_packet_test.json"))


def build_td_text_dataset(traffic_data, first_label=None, second_label=None, task_name=None, granularity=None):
    """Building the text datasets of traffic detection task"""
    if task_name == "EMD":
        instruction = "Below is a traffic " + granularity +  ". Please conduct the encrypted malware detection task."

        output = "This might be a " + first_label + \
                 " traffic " + granularity + ". The category is likely to be recognized as " + second_label + "."

    elif task_name == "EAC":
        instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted App classification task."

        output = "The traffic category is likely to be recognized as " + second_label + "."

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction,
                "input": data,
                "output": output
            }
        )

    return dataset


def build_tg_text_dataset(traffic_data, traffic_label, granularity):
    """Building the text datasets of traffic generation task"""
    instruction = "Please generate a " + granularity + " of " + traffic_label + " traffic."

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction,
                "input": "",
                "output": data
            }
        )

    return dataset


def build_tu_text_dataset():
    """Building the text datasets of traffic understanding task"""
    pass
