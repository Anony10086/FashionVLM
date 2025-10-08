import os
from tqdm import tqdm
import json
import time
import argparse
import sys
import warnings
import copy

from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from dataset import FashionRecDatasetBase

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='basic_recommendation',
                    help='task name, basic_recommendation | personalized_recommendation | alternative_recommendation')
parser.add_argument('--split', type=str, default='test', help='split of dataset, test | valid')
parser.add_argument('--model', type=str, default='0.5b-si', help='model name: llava_onevision_05b_si | llava_onevision_7b_ov | llava_onevision_7b_ov_chat')
args = parser.parse_args()



#######################################
##############Loading Data#############
#######################################
data_set_root = "./datasets/FashionRec/data"
dataset = FashionRecDatasetBase(
    tar_files=f"{data_set_root}/{args.task}/test/000.tar",
    num_examples=500
)


#######################################
#############Loading Model#############
#######################################
warnings.filterwarnings("ignore")
if args.model == 'llava_onevision_7b_ov':
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
elif args.model == 'llava_onevision_7b_ov_chat':
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov-chat"
elif args.model == 'llava_onevision_05b_si':
    pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
else:
    raise ValueError("Model not supported.")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "attn_implementation": None,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
model.config.tokenizer_padding_side = 'left'  # Use left padding for batch processing
model.eval()


#######################################
#########Strating Inference############
#######################################
results = {}
start_time = time.perf_counter()  # 记录总时间

# Initialize lists to store batch data
batch_size = 3
batched_input_ids = []
batched_images = []
batched_image_sizes = []
batched_indices = []

for idx, (image, json_data) in enumerate(tqdm(dataset)):
    index = json_data['key']
    conversation = json_data['conversation']
    img_path = f'./datasets/FashionRec/data/basic_recommendation/test/temp/{index}.jpg'
    image = Image.open(img_path)
    
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    question = DEFAULT_IMAGE_TOKEN + json_data['conversation'][0]['value']

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.system = 'You are a fashion stylist. Recommend an item according to user`s query and uploaded image. Recommend only one item for each query.'
    conv.append_message(conv.roles[0], question)
    for x in conversation[1:-1]:
        if x['from'] == 'human':
            conv.append_message(conv.roles[0], x['value'])
        elif x['from'] == 'gpt':
            conv.append_message(conv.roles[1], x['value'])

    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    # print("Prompt to model: ", prompt_question)
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").to(device)
    batched_input_ids.append(input_ids)
    batched_images.append(image)
    batched_image_sizes.append(image.size)
    batched_indices.append(index)

    if (idx + 1) % batch_size == 0 or idx == len(dataset) - 1:
        # Pad input_ids to the same length
        padded_input_ids = pad_sequence(batched_input_ids,
                                        batch_first=True,
                                        padding_value=tokenizer.pad_token_id).to(device)
        attention_mask = (padded_input_ids != tokenizer.pad_token_id).to(dtype=torch.float16)
        print(padded_input_ids.size())

        # Stack image tensors
        image_tensor = process_images(batched_images, image_processor, model.config)
        # print(image_tensor.size())
        batched_image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        modalities = ["image" for _ in image_tensor]
        
        # --- 3. Generate outputs ---
        cont = model.generate(
            padded_input_ids,
            images=batched_image_tensors,
            image_sizes=batched_image_sizes,
            modalities=modalities,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print("Model outputs: ", text_outputs)
        
        # --- 4. Post-process and clear batch lists ---
        for i, output_text in enumerate(text_outputs):
            current_index = batched_indices[i]
            results[current_index] = output_text
        
        # Clear lists for the next batch
        batched_input_ids = []
        batched_images = []
        batched_image_sizes = []
        batched_indices = []
        torch.cuda.empty_cache()
        print(f"Batch {idx // batch_size + 1} processed.")

end_time = time.perf_counter()  # 记录总结束时间
total_time = end_time - start_time  # 总时间（包括后处理）
print(f"Total Cost Time: {total_time:.2f} seconds")
print(f"Average Batch Latency: {total_time / (len(dataset) + 1):.2f} seconds")

output_dir = args.model

save_dir = f"./output/{output_dir}"
os.makedirs(save_dir, exist_ok=True)
with open(f"{save_dir}/{args.task}_results.json", "w") as f:
    json.dump(results, f, indent=2)
