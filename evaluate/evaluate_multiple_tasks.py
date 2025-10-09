import os
import argparse
import json
from tqdm import tqdm
import pandas as pd
import pickle
import numpy as np
import tarfile

import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_pil_image, InterpolationMode

from metric import CLIPScore, SentenceBertScore, CompatibilityScore, ResizeSquareImage, ConvertRGB


parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='fashion_vlm',
                    help='method name, fashion_vlm | show_o | gpt_4o | llama_32 | llava_16 | gpt_4.1')
parser.add_argument('--task', type=str, default='basic_recommendation',
                    help='task name, basic_recommendation | personalized_recommendation | alternative_recommendation')
parser.add_argument('--split', type=str, default='test', help='split of dataset, test | valid')
args = parser.parse_args()


class FashionRecEvalDataset(Dataset):
    def __init__(self, dataset_dir, json_file_name, inference_dir, task, clip_transform=None):
        self.dataset_dir = dataset_dir
        self.inference_dir = inference_dir
        self.task = task
        self.clip_transform = transforms.Compose([
            # ConvertRGB(),
            # ResizeSquareImage(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.481, 0.458, 0.408], [0.269, 0.261, 0.276])
        ])

        # Loading dataset meta data such as items and users
        meta_dir = f'{dataset_dir}/../../../meta'
        # self.items = pd.read_parquet(os.path.join(meta_dir, f'items_lite.parquet'))
        # self.users = pd.read_parquet(os.path.join(meta_dir, f'users_lite.parquet'))
        # subcategories = self.items.subcategory.unique()
        #
        # self.history_embedding = {uid: {x: None for x in subcategories} for uid in self.users.user_id.unique()}

        # Loading items clip embedding
        with open(os.path.join(meta_dir, f'clip_features.pkl'), 'rb') as f:
            self.clip_embeddings = pickle.load(f)

        if json_file_name.endswith('json'):
            with open(os.path.join(inference_dir, json_file_name), 'r') as f:
                self.results = json.load(f)
                self.results = [(k, v) for k, v in self.results.items()]
        elif json_file_name.endswith('jsonl'):
            self.results = []
            with open(os.path.join(inference_dir, json_file_name), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    index = data['custom_id'].split('-')[-1]
                    response = data['response']['body']['choices'][0]['message']['content']
                    self.results.append((index, response))
        
        self._load_tar_files(os.path.join(dataset_dir, '000.tar'))

    def _load_tar_files(self, tar_path: str):
        self.jsons = {}
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    filename = os.path.basename(member.name)  # get 0009741.json
                    base_name = os.path.splitext(filename)[0]  # get 0009741
                    if member.name.endswith(".json"):
                        with tar.extractfile(member) as json_file:
                            self.jsons[base_name] = json.loads(json_file.read().decode("utf-8"))

    def __getitem__(self, idx):
        result = self.results[idx]
        index = result[0]

        # Loading Ground Truth Data
        json_data = self.jsons[index]

        pred_text = result[1]
        targ_text = json_data['conversation'][-1]['value']

        pred_img_path = os.path.join(self.inference_dir, self.task, f'{index}_gen.jpg')
        pred_image = Image.open(pred_img_path).convert('RGB')
        # pred_image = self.clip_transform(pred_image)

        if self.task == 'alternative_recommendation':
            targ_item = json_data['changeable_items'][0]  # First one is the target item, Second one the item to be changed.
        else:
            targ_item = json_data['target_items'][-1]

        targ_img_path = os.path.join('./datasets/FashionRec/data/item_images', targ_item['item_id'] + '.jpg')
        targ_image = Image.open(targ_img_path).convert('RGB')
        # targ_image = self.clip_transform(targ_image)

        history_feats = []
        if self.task == 'personalized_recommendation':
            # uid = json_data['uid']
            history_img_ids = [x['item_id'] for x in json_data['history']]
            for item_id in history_img_ids:
                image_feats = self.clip_embeddings[item_id]['image_embeds']
                history_feats.append(torch.from_numpy(image_feats))
            history_feats = torch.stack(history_feats, dim=0)
            history_feats = torch.mean(history_feats, dim=0)
        else:
            history_feats = torch.zeros(512)

        # partial_outfit = json_data['partial_outfit'] + json_data['target_items'][1:]
        # partial_outfit_images = []
        # for item in partial_outfit:
        #     image_path = item['path']
        #     image = Image.open(image_path).convert('RGB')
        #     image = self.clip_transform(image)
        #     partial_outfit_images.append(image)  # 4, 3, 224, 224
        # partial_outfit_images = torch.stack(partial_outfit_images, dim=0)  # 4, 3, 224, 224
        # partial_outfit_length = len(partial_outfit)

        return pred_text, targ_text, pred_image, targ_image, history_feats

    def __len__(self):
        return len(self.results)


def collect_fn(batch):
    pred_text = [x[0] for x in batch]
    targ_text = [x[1] for x in batch]
    pred_image = [x[2] for x in batch]
    targ_image = [x[3] for x in batch]

    history_feats = torch.stack([x[4] for x in batch], dim=0)
    return pred_text, targ_text, pred_image, targ_image, history_feats


###############################
##########LOAD CLIP############
###############################
device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_metric = CLIPScore()
bert_metric = SentenceBertScore()
# comp_metric = CompatibilityScore(model_path=f'./evaluate/compatibility_net.pth')

###############################
##########LOAD DATA############
###############################
dataset_dir = f'datasets/FashionRec/data/{args.task}/{args.split}'
if args.method in ['gpt_4o', 'o3_mini', 'o4_mini', 'gpt_4.1']:
    json_file_name = f"{args.task}_results.jsonl"
else:
    json_file_name = f"{args.task}_results.json"
inference_dir = f'output/{args.method}'

eval_dataset = FashionRecEvalDataset(
    dataset_dir=dataset_dir, json_file_name=json_file_name,
    inference_dir=inference_dir, task=args.task
)
dataloader = DataLoader(
    eval_dataset, batch_size=10, shuffle=False,
    drop_last=False, num_workers=0,
    collate_fn=collect_fn
)

clip_text_scores = []
clip_image_scores = []
sbert_scores = []
per_scores = []
grd_per_scores = []
visualize = True
for pred_text, targ_text, pred_image, targ_image, history_feats in tqdm(dataloader):
    # visualize
    if visualize:
        targ_image[0].save(f'{inference_dir}/targ_image_temp.jpg')
        pred_image[0].save(f'{inference_dir}/pred_image_temp.jpg')
        print('pred_text: ' + pred_text[0], '\n', 'targ_text: ' + targ_text[0])
        visualize = False

    # pred_text_token = clip_metric.tokenizer(pred_text).to(device)
    # targ_text_token = clip_metric.tokenizer(targ_text).to(device)

    # pred_image = pred_image.to(device)
    # targ_image = targ_image.to(device)

    # Calculate CLIP score
    targ_image_feats = clip_metric.encode_image(targ_image)
    pred_text_feats = clip_metric.encode_text(pred_text)
    clip_text_score = 100 * F.cosine_similarity(targ_image_feats, pred_text_feats)
    clip_text_scores.append(clip_text_score)

    pred_image_feats = clip_metric.encode_image(pred_image)
    clip_image_score = 100 * F.cosine_similarity(targ_image_feats, pred_image_feats)
    clip_image_scores.append(clip_image_score)

    # Calculate Sentence BERT score
    sbert_scores.append(bert_metric.calculate_sentence_bert_score(pred_text, targ_text))

    # Calculate Personalization
    if args.task == 'personalized_recommendation':
        history_feats = history_feats.to(device)
        history_feats = torch.nn.functional.normalize(history_feats, p=2, dim=1)

        per_smi = 100 * torch.nn.functional.cosine_similarity(pred_image_feats, history_feats)
        grd_per_smi = 100 * torch.nn.functional.cosine_similarity(targ_image_feats, history_feats)

        per_scores.append(per_smi)
        grd_per_scores.append(grd_per_smi)

    # Calculate compatibility score
    # concate_outfit_images = concate_outfit_images.to(device)
    # for i, length in enumerate(concate_outfit_length):
    #     start_idx = sum(concate_outfit_length[:i])
    #     end_idx = start_idx + length
    #     outfit_images = concate_outfit_images[start_idx:end_idx]
    #
    #     complete_outfit = torch.cat([outfit_images, pred_image[i].unsqueeze(0)])
    #     outfit_feats = clip_metric.model.encode_image(complete_outfit)
    #     outfit_feats = outfit_feats.unsqueeze(0)
    #     comp_score = comp_metric.evaluate(outfit_feats)
    #     comp_scores.append(comp_score)
    #
    #     complete_outfit = torch.cat([outfit_images, targ_image[i].unsqueeze(0)])
    #     outfit_feats = clip_metric.model.encode_image(complete_outfit)
    #     outfit_feats = outfit_feats.unsqueeze(0)
    #     comp_score = comp_metric.evaluate(outfit_feats)
    #     grd_comp_scores.append(comp_score)

        # for j, img in enumerate(torch.cat([outfit_images, pred_image[i].unsqueeze(0), targ_image[i].unsqueeze(0)])):
        #     img = to_pil_image(img.cpu())
        #     img.save(f'{inference_dir}/{j}.jpg')

cts = torch.cat(clip_text_scores, dim=0).cpu().numpy().mean()
cis = torch.cat(clip_image_scores, dim=0).cpu().numpy().mean()
sbert_score = torch.cat(sbert_scores, dim=0).cpu().numpy().mean()
# comp_score = torch.cat(comp_scores, dim=0).cpu().numpy().mean()
# grd_comp_score = torch.cat(grd_comp_scores, dim=0).cpu().numpy().mean()
print(f"{args.method} on {args.task} task of {args.split} split achieves "
      f"Sentence Bert score: {sbert_score:.2f}, CLIP Text score: {cts:.2f}, CLIP Image score: {cis:.2f}.")
      # f"Compatibility score: {comp_score:.2f}, Ground Truth Compatibility Score: {grd_comp_score:.2f}"


if args.task == 'personalized_recommendation':
    per_score = torch.cat(per_scores, dim=0).cpu().numpy().mean()
    grd_per_score = torch.cat(grd_per_scores, dim=0).cpu().numpy().mean()
    print(f"Personalization score is {per_score:.2f}. Ground Truth score is {grd_per_score:.2f}")
