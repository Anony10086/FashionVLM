import os
from tqdm import tqdm
import json
import time
import argparse
import tarfile

import ollama

from dataset import FashionRecDatasetBase

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='basic_recommendation',
                    help='task name, basic_recommendation | personalized_recommendation | alternative_recommendation')
parser.add_argument('--split', type=str, default='test', help='split of dataset, test | valid')
parser.add_argument('--method', type=str, default='llama32', help='model name: llama3.2-vision | llava:v1.6')
args = parser.parse_args()



#######################################
##############Loading Data#############
#######################################
data_set_root = f"./datasets/FashionRec/data"
dataset = FashionRecDatasetBase(
    tar_files=f"{data_set_root}/{args.task}/test/000.tar",
    num_examples=500
)
# Extract and save images for ollama
temp_dir = f"{data_set_root}/{args.task}/test/temp" 
if os.path.exists(temp_dir) and os.listdir(temp_dir):
    print(f"Directory {temp_dir} already exists and is not empty. Skipping extraction.")

os.makedirs(temp_dir, exist_ok=True)
print(f"Extracting {f"{data_set_root}/{args.task}/test/000.tar"} to {temp_dir}...")
try:
    with tarfile.open(f"{data_set_root}/{args.task}/test/000.tar", 'r') as tar:
        tar.extractall(path=temp_dir)
    print("Extraction complete.")
except Exception as e:
    print(f"Error during extraction: {e}")
    import shutil
    shutil.rmtree(temp_dir)
    raise



#######################################
#########Strating Inference############
#######################################
results = {}
start_time = time.perf_counter()  # 记录总时间
for idx, (image, json_data) in enumerate(tqdm(dataset)):
    index = json_data['key']
    conversation = json_data['conversation']
    messages = [
        {
            'role': 'system',
            'content': 'You are a fashion stylist. Recommend an item according to user`s query and uploaded image. Recommend only one item for each query.'
        },
        {
            'role': 'user',
            'content': json_data['conversation'][0]['value'],
            'images': [f'./datasets/FashionRec/data/basic_recommendation/test/temp/{index}.jpg']
        }
    ]
    for conv in conversation[1:-1]:
        if conv['from'] == 'human':
            messages.append(
                {
                    'role': 'user',
                    'content': conv['value']
                }
            )
        elif conv['from'] == 'gpt':
            messages.append(
                {
                    'role': 'assistant',
                    'content': conv['value']
                }
            )
    response = ollama.chat(
        model=args.method,
        messages=messages
    )
    text = response['message']['content']
    results[index] = text

end_time = time.perf_counter()  # 记录总结束时间
total_time = end_time - start_time  # 总时间（包括后处理）
print(f"Total Cost Time: {total_time:.2f} seconds")
print(f"Average Batch Latency: {total_time / (len(dataset) + 1):.2f} seconds")

if args.method == 'llama3.2-vision':
    output_dir = 'llama_32'
elif args.method == 'llava:v1.6':
    output_dir = 'llava_16'
else:
    raise ValueError

save_dir = f"./output/{output_dir}"
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/{args.task}_results.json", "w") as f:
    json.dump(results, f, indent=2)
