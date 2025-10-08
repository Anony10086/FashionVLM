import os
from tqdm import tqdm
import json
import time
import base64
import io
import argparse

from PIL import Image

from dataset import FashionRecDatasetBase

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='basic_recommendation',
                    help='task name, basic_recommendation | personalized_recommendation | alternative_recommendation')
parser.add_argument('--model', type=str, default='gpt-4o', help='gpt model choice: o4-mini | o3-mini | gpt-4o | gpt-4.1')
parser.add_argument('--split', type=str, default='test', help='split of dataset, test | valid')
args = parser.parse_args()



def pil_to_base64(image, format="PNG"):
    """
    将 PIL Image 转换为 Base64 编码的字符串。

    参数：
        image: PIL Image 对象
        format: 图像格式（默认 "PNG"，也可以是 "JPEG" 等）

    返回：
        Base64 编码的字符串
    """
    # 确保输入是 PIL Image 对象
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image object, but got {type(image)}")
    # 创建一个字节流缓冲区
    buffer = io.BytesIO()
    # 将 PIL Image 保存到字节流中
    image.save(buffer, format=format)
    # 获取字节流的内容
    img_bytes = buffer.getvalue()
    # 将字节流编码为 Base64
    base64_bytes = base64.b64encode(img_bytes)
    # 将 Base64 字节转换为字符串
    base64_string = base64_bytes.decode("utf-8")

    return base64_string


#######################################
##############Loading Data#############
#######################################
data_set_root = f"./datasets/FashionRec/data"
dataset = FashionRecDatasetBase(
    tar_files=f"{data_set_root}/{args.task}/test/000.tar",
    num_examples=500
)


#######################################
#########Strating Inference############
#######################################
batch_samples = []
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
            'content': [
                {
                    "type": "text",
                    "text": json_data['conversation'][0]['value']
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{pil_to_base64(image)}"
                    }
                }
            ]
        }
    ]
    for conv in conversation[1:-1]:
        if conv['from'] == 'human':
            messages.append(
                {
                    'role': 'user',
                    'content': [
                        {
                            "type": "text",
                            "text": conv['value']
                        }
                    ]
                }
            )
        elif conv['from'] == 'gpt':
            messages.append(
                {
                    'role': 'assistant',
                    'content': [
                        {
                            "type": "text",
                            "text": conv['value']
                        }
                    ]
                }
            )

    request_sample = {
        "custom_id": f"request-{index}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": args.model,
            "messages": messages,
            "temperature": 1.0,
            "max_tokens": 350,
        }
    }
    batch_samples.append(request_sample)

save_dir = f"./output/{args.model.replace('-', '_')}"
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/{args.task}_requests.jsonl", "w") as f:
    for request_sample in batch_samples:
        f.write(json.dumps(request_sample) + "\n")
