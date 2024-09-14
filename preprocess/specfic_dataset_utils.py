from preprocess_utils import build_td_text_dataset, write_labels, build_dataset, save_dataset
from tqdm import tqdm
import os


def ustc_tfc2016_preprocess(args, detection_task="EMD"):
    """USTC-TFC2016 for the encrypted malware detection (EMD) task"""

    train_dataset = []
    test_dataset = []
    labels = []

    benign_class_path = os.path.join(args.input, "Benign")
    malware_class_path = os.path.join(args.input, "Malware")

    files = os.listdir(benign_class_path)
    labels.extend(files)

    for file in tqdm(files):
        class_train_data, class_test_data = build_dataset(args, benign_class_path, file)

        train_text_data = build_td_text_dataset(class_train_data, first_label="benign", second_label=file, task_name=detection_task, granularity=args.granularity)
        test_text_data = build_td_text_dataset(class_test_data, first_label="benign", second_label=file, task_name=detection_task, granularity=args.granularity)

        train_dataset.extend(train_text_data)
        test_dataset.extend(test_text_data)

    files = os.listdir(malware_class_path)
    labels.extend(files)

    for file in tqdm(files):
        class_train_data, class_test_data = build_dataset(args, malware_class_path, file)

        train_text_data = build_td_text_dataset(class_train_data, first_label="malware", second_label=file, task_name=detection_task, granularity=args.granularity)
        test_text_data = build_td_text_dataset(class_test_data, first_label="malware", second_label=file, task_name=detection_task, granularity=args.granularity)

        train_dataset.extend(train_text_data)
        test_dataset.extend(test_text_data)

    save_dataset(args, train_dataset, test_dataset)

    write_labels(labels, os.path.join(args.output_path, args.output_name + "_label.json"))
