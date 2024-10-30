# -*- coding: utf-8 -*-
# @Author  : ZGCLLM

import json
import os
from tqdm import tqdm

import pandas as pd

def file_w(output_file, parsed_data):
	with open(output_file, 'w', encoding="utf-8") as f:
		for data in parsed_data:
			f.write(json.dumps(data, ensure_ascii=False) + "\n")
	return 0

def test_and_label(GLM4_test_label_json, message):
	GLM4_test_json  = GLM4_test_label_json.replace("_with_label","")

	for item in message:
		item["messages"][2]["content"] = '-'

	output_file = GLM4_test_json
	file_w(output_file, message)
	return 0

def Transfer_file(root_dir, file_name):
	data_df = pd.read_json(root_dir + file_name, lines=True)
	data_jsonl = []
	for index, row in data_df.iterrows():
		instruction = row['instruction'].split("\n<packet>:")[0]
		input_value = "<packet>:" + row['instruction'].split("\n<packet>:")[1].split(", tcp.payload:")[0] + "."
		output_value = row['output']

		messages = [
			{"role": "system", "content": f"{instruction}"},
			{"role": "user", "content": f"{input_value}"},
			{"role": "assistant", "content": f"{output_value}"}
		]

		data_jsonl.append({"messages":messages})
	
	# jsonl for GLM4
	file_name = file_name.replace(".json",".jsonl")
	output_file = root_dir + "GLM4_" + file_name # "new_train_glm.jsonl"

	if "test" in file_name:
		temp = output_file.replace(".jsonl","_with_label.jsonl")
		output_file = temp

	file_w(output_file, data_jsonl)

	# no label version
	if "test" in file_name:
		test_and_label(output_file, data_jsonl)

	return 0

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
	# data files' path
	src_dataset = "./datas"
	for r,d,f in os.walk(src_dataset):
		for item in tqdm(range(len(f))):
			if "train" in f[item] or "test" in f[item]:
				Transfer_file(r, f[item])
		break