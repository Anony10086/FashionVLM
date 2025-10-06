import os
import json
from tqdm import tqdm

import ollama
import torch
from PIL import Image

from metric import CLIPScore


clip_metric = CLIPScore()

root = '/mnt/d/PostDoc/fifth paper/code/FashionVLM/output/fashion_vlm'
target_root = '/mnt/d/PostDoc/fifth paper/code/FashionVLM/datasets/FashionRec/data/a100/temp'
with open(os.path.join(root, 'a100_results.json'), 'r') as f:
    results = json.load(f)


lat = 0
lats = 0.0
for key, value in tqdm(results.items()):
    gen_image_path = os.path.join(root, 'a100', f'{key}_gen.jpg')

    messages = [
        {
            'role': 'system',
            'content': 'You need to extract the recommended item description from the user`s message. '
                       'If there is no recommended item, just output None.'
                       'Summarize the description of the recommended item. You must classify what is provided and what is recommended.'
                       'We only need the description of the recommended items.'
                       'IMPORTANT: Reply in One scentence. Use two examples I provided',
        },
        {
            'role': 'user',
            'content': "It sounds like you're aiming for a chic and coordinated look! Based on your preference for knee-high styles with buckles, I recommend checking out some dark brown knee-high boots featuring a buckle detail at the ankle. They would beautifully complement your outfits and add a touch of sophistication. Let me know if that fits what you're looking for!",
        },
        {
            'role': 'assistant',
            'content': "dark brown knee-high boots featuring a buckle detail at the ankle"
        },
        {
            'role': 'user',
            'content': "Yes, I recommend the following sandals:",
        },
        {
            'role': 'assistant',
            'content': "Sandal"
        },
        {
            'role': 'user',
            'content': value,
        }
    ]
    response = ollama.chat(
        model='phi4',
        messages=messages
    )
    text = response['message']['content']

    index = key.split('_')[1]
    candidate_item_paths = [os.path.join(target_root, f'{index}_{i}.jpg') for i in range(1, 6)]
    candidate_images = [Image.open(x).convert('RGB') for x in candidate_item_paths]
    candidate_embeddings = clip_metric.encode_image(candidate_images)

    pred_embeddings = clip_metric.encode_text([text])
    sim_scores = 100 * torch.nn.functional.cosine_similarity(candidate_embeddings, pred_embeddings)
    pred_index = torch.argmax(sim_scores).item()

    with open(os.path.join(target_root, f'{index}.json'), 'r') as f:
        temp = json.load(f)
    gt = int(temp['gt']) - 1
    if gt == pred_index:
        lat += 1
    lats += temp['gt_distribution'][pred_index]

print(f'lat: {lat}\nlats: {lats}')
