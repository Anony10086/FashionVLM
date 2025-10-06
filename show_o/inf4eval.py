import argparse
import logging
import os
from tqdm import tqdm
import json
import math
import shutil

import ollama
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from inference_showo import ShowoModel
from training.prompting_utils import UniversalPrompting


def parse_all_args():
    parser = argparse.ArgumentParser(description="Inference script for show-o model")
    parser.add_argument(
        "--data_path",
        type=str,
        default='/data/path/',
        help="A folder containing the dataset for training and inference."
    )
    parser.add_argument(
        '--img_folder_path',
        type=str,
        default='/data/path/xxx'
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='',
        help="The name of the Dataset for training and inference."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/output/path/",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="FITB",
        help="The task for evaluation: FITB or GOR (Generative Outfit Recommendation)."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test"
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default="10",
        help=("0 = history llama, recommend show-o, generate show-o",
              "10 = history llama, recommend llama, generate show-o")
    )
    parser.add_argument(
        "--use_history",
        action="store_false",
        help="Whether to use user's history or not."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    args = parser.parse_args()
    return args


class ShowoHistorySummaryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        resolution = 512
        self.image_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', padding_side="left")
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=128,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100, cond_dropout_prob=0.1
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iid, image_path, question = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        prompt = self.uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])['input_ids'][0]
        prompt = torch.tensor(prompt)
        return iid, image, prompt


class ShowoFITBInferenceDataset(Dataset):
    def __init__(self, data, output_dir, id_cate_dict, use_history=True):
        self.data = data
        self.use_history = use_history
        resolution = 512
        self.image_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', padding_side="left")
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=128,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100, cond_dropout_prob=0.1
        )
        if use_history:
            with open(os.path.join(output_dir, 'history_summary.json')) as f:
                self.history = json.load(f)
        else:
            self.history = []
        self.id_cate_dict = id_cate_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid, oid, image_path, cid = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        category = self.id_cate_dict[cid]
        if self.use_history:
            user_preference = self.history[str(uid)].get(str(cid), "")
            question = (f"What {category} would you recommend to the user that match this partial outfit shown in image. "
                        f"Try to describe it in detail in one sentence. {user_preference}")
        else:
            question = f"I'm putting together this outfit but need some advice on {category}. What would you recommend?"

        prompt = self.uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])['input_ids'][0]
        prompt = torch.tensor(prompt)
        return uid, oid, image, prompt, cid


def main():
    # 基础配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    # 创建logger
    logger = logging.getLogger(__name__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load show-o model
    # unified_model = ShowoModel(
    #     config="configs/showo_demo_512x512.yaml",
    #     max_new_tokens=1000,
    #     temperature=0.8,
    #     top_k=1,
    # )
    unified_model = ShowoModel(
        config="outputs/show-o-tuning-stage2-512x512/config_infer.yaml",
        max_new_tokens=1000,
        temperature=0.8,
        top_k=1,
        load_from_showo=False,
    )
    args = parse_all_args()
    logger.info(f"Args: {args}")
    logger.info("Data loading......")
    data_path = os.path.join(args.data_path, args.dataset_name)

    all_item_image_path = np.load(os.path.join(data_path, 'all_item_image_paths.npy'), allow_pickle=True)
    id_cate_dict = np.load(os.path.join(data_path, 'id_cate_dict.npy'), allow_pickle=True).item()

    if args.mode == "test":
        test_fitb = np.load(os.path.join(data_path, "processed", "fitb_test.npy"), allow_pickle=True).item()
        test_grd = np.load(os.path.join(data_path, "test_grd.npy"), allow_pickle=True).item()
        test_history = np.load(os.path.join(data_path, "test_history.npy"), allow_pickle=True).item()
    else:
        test_fitb = np.load(os.path.join(data_path, "processed", "fitb_vaild.npy"), allow_pickle=True).item()
        test_grd = np.load(os.path.join(data_path, "vaild_grd.npy"), allow_pickle=True).item()
        test_history = np.load(os.path.join(data_path, "vaild_history.npy"), allow_pickle=True).item()

    fitb_merge_save_path = os.path.join(args.output_dir, "fitb_merge")
    os.makedirs(fitb_merge_save_path, exist_ok=True)
    if len(os.listdir(fitb_merge_save_path)) < len(test_fitb["oids"]):
        assert args.dataset_name in ['ifashion', 'polyvore']
        base_size = 582
        temp_background = Image.new('RGB', (base_size, base_size), 'white')
        # More outfit need to be merged
        for uid, oid, outfit in tqdm(zip(test_fitb["uids"], test_fitb["oids"], test_fitb["outfits"])):
            item_list = []
            for iid in outfit.tolist():
                # iid = 0 means blanked item
                if iid > 0:
                    image_path = os.path.join(args.img_folder_path, all_item_image_path[iid])
                    img = Image.open(image_path)
                    if args.dataset_name == "ifashion":
                        ratio = 291 / max(img.size)  # 计算缩放比例
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))  # 计算新尺寸
                        img = img.resize(new_size, Image.NEAREST)
                    item_list.append(img)
            # 粘贴图片到指定位置
            background = temp_background.copy()
            background.paste(item_list[0], (0, 0))  # 左上角
            background.paste(item_list[1], (291, 0))  # 右上角
            background.paste(item_list[2], (0, 291))  # 左下角
            # 调整最终大小为512*512
            background = background.resize((512, 512), Image.LANCZOS)
            # 保存图片
            background.save(os.path.join(fitb_merge_save_path, f"{uid}_{oid}.jpg"))

    result_output_dir = os.path.join(args.output_dir, "eval-test", f"{args.task}-checkpoint-{args.ckpt}-cate12.0-mutual5.0-hist4.0")
    os.makedirs(result_output_dir, exist_ok=True)
    # 1. summarize user history
    output_path = os.path.join(result_output_dir, 'history_summary.json')
    if not os.path.exists(output_path) and args.use_history:
        # we only need those user-item interaction within fitb test
        user_category_interaction = {}
        for idx, uid in enumerate(test_fitb["uids"]):
            if uid not in user_category_interaction.keys():
                user_category_interaction[uid] = set()
            outfit = test_fitb["outfits"][idx]
            zero_indices = torch.where(outfit == 0)[0].item()
            category_id = test_fitb["category"][idx].tolist()[zero_indices]
            user_category_interaction[uid].add(category_id)

        item_for_process = []
        history_merge_save_path = os.path.join(args.output_dir, "history_merge")
        os.makedirs(history_merge_save_path, exist_ok=True)
        history_merge_list = os.listdir(history_merge_save_path)
        for uid, value in tqdm(test_history.items()):
            for category_id, items in value.items():
                if category_id not in user_category_interaction[uid]:
                    continue
                category = id_cate_dict[category_id]
                save_path = os.path.join(history_merge_save_path, f"{uid}_{category_id}.jpg")
                if f"{uid}_{category_id}.jpg" not in history_merge_list:
                    merge_images_and_save(
                        [os.path.join(args.img_folder_path, all_item_image_path[iid]) for iid in items], save_path)
                item_for_process.append((f"{uid}_{category_id}", save_path,
                                         f"Analysis these fashion products and summary user's preference towards {category} in one sentence. "
                                         f"You should start with the user seems to prefer"))

        # summarize_history_dataloader = DataLoader(
        #     ShowoHistorySummaryDataset(item_for_process),
        #     batch_size=1,
        #     shuffle=False,
        #     drop_last=False,
        #     # num_workers=args.dataloader_num_workers,
        #     num_workers=20
        # )

        with torch.no_grad():
            history_summary = {}
            # use show-o to inference
            # for (uid_cid, image, prompt) in tqdm(summarize_history_dataloader):
            #     # use show-o to inference
            #     image = image.to(device)
            #     prompt = prompt.to(device)
            #     results = unified_model.mmu_infer_tensor(image, prompt)
            #     for idx, description in zip(uid_cid, results):
            #         uid, cid = idx.split("_")
            #         if uid not in history_summary:
            #             history_summary[uid] = {}
            #         history_summary[uid][cid] = description

            # use llama to inference
            for uid_cid, image_path, prompt in tqdm(item_for_process):
                response = ollama.chat(
                    model='llama3.2-vision',
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }]
                )
                uid, cid = uid_cid.split("_")
                if uid not in history_summary:
                    history_summary[uid] = {}
                history_summary[uid][cid] = response["message"]["content"]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(history_summary, f, ensure_ascii=False, indent=2)

    # 2. generate recommend item description, FITB test
    output_path = os.path.join(result_output_dir, 'fitb_recommend_item.json')
    if not os.path.exists(output_path):
        outfit_to_be_processed = []
        for uid, oid, category_list, outfit in tqdm(zip(test_fitb["uids"], test_fitb["oids"], test_fitb["category"], test_fitb["outfits"])):
            partial_outfit_path = os.path.join(fitb_merge_save_path, f"{uid}_{oid}.jpg")
            cid = 0
            for idx, iid in enumerate(outfit.tolist()):
                if iid == 0:
                    cid = category_list[idx].item()
                    continue
            outfit_to_be_processed.append((uid, oid, partial_outfit_path, cid))

        recommend_item = {}

        # use show-o to recommend item
        with torch.no_grad():
            fitb_dataloader = DataLoader(
                ShowoFITBInferenceDataset(outfit_to_be_processed, result_output_dir, id_cate_dict, use_history=args.use_history),
                batch_size=1,
                shuffle=False,
                drop_last=False,
                # num_workers=args.dataloader_num_workers,
                num_workers=0
            )
            for (uids, oids, image, prompt, cids) in tqdm(fitb_dataloader):
                # use show-o to inference
                image = image.to(device)
                prompt = prompt.to(device)
                # use showo to generate recommendations and reasons
                results = unified_model.mmu_infer_tensor(image, prompt)

                for uid, oid, result, cid in zip(uids, oids, results, cids):
                    # use phi4 to extract recommended item descriptions
                    target_cate_str = id_cate_dict[cid.item()]
                    description = extract_item_description_via_phi4(result, target_cate_str)
                    if uid.item() not in recommend_item.keys():
                        recommend_item[uid.item()] = {}
                    recommend_item[uid.item()][oid.item()] = str(description)
                    print(f"For incomplete outfit {uid}_{oid}, we recommend {description}. Original showo output is: {result}.")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(recommend_item, f, ensure_ascii=False, indent=2)

        # use llama to recommend item
        # with open(os.path.join(result_output_dir, 'history_summary.json')) as f:
        #     history = json.load(f)
        # for (uid, oid, partial_outfit_path, cid) in tqdm(outfit_to_be_processed):
        #     user_preference = history[str(uid)].get(str(cid), "")
        #     category = id_cate_dict[cid]
        #     prompt = (
        #         f"Generate a brief visual description of a {category} in one sentence that matches this partial outfit. "
        #         f"User preference: {user_preference}. "
        #         "Only describe the item's appearance using attributes like color, pattern, style, material, and design details. "
        #         "DO NOT include any explanations, recommendations, or other text. "
        #     )
        #     response = ollama.chat(
        #         model='llama3.2-vision',
        #         messages=[{
        #             'role': 'user',
        #             'content': prompt,
        #             'images': [partial_outfit_path]
        #         }]
        #     )
        #     if uid not in recommend_item.keys():
        #         recommend_item[uid] = {}
        #     recommend_item[int(uid)][int(oid)] = response["message"]["content"]
        #
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     json.dump(recommend_item, f, ensure_ascii=False, indent=2)

    # generate recommended item image based on description
    output_path = os.path.join(args.output_dir, "eval-test", f"{args.task}-checkpoint-{args.ckpt}-cate12.0-mutual5.0-hist4.0", "images")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    recommend_item_path = os.path.join(result_output_dir, "fitb_recommend_item.json")

    gen_file = {}
    gen_file_save_path = os.path.join(args.output_dir, "eval-test", f"{args.task}-checkpoint-{args.ckpt}-cate12.0-mutual5.0-hist4.0.npy")
    grd_file = {}
    grd_file_save_path = os.path.join(args.output_dir, "eval-test", "FITB-grd-new.npy")
    with open(recommend_item_path, 'r', encoding='utf-8') as f:
        recommend_item = json.load(f)
    prompt_list, save_path_list = [], []
    for uid, value in recommend_item.items():
        if uid not in gen_file:
            gen_file[uid] = {}
            grd_file[uid] = {}
        for oid, description in value.items():
            outfit_index = test_fitb["oids"].index(int(oid))
            category_id = test_fitb['category'][outfit_index][test_fitb['outfits'][outfit_index].tolist().index(0)].item()
            category = id_cate_dict[category_id]
            # use show-o to generate image
            prompt = f"{description}, product image, high quality"

            image_path = os.path.join(output_path, f"{uid}_{oid}_pred.jpg")

            if not os.path.exists(image_path):
                prompt_list.append(prompt)
                save_path_list.append(image_path)

            outfit_result = {'cates': [torch.tensor(category_id, dtype=torch.int64)],
                             'full_cate': test_fitb['category'][outfit_index],
                             'outfits': test_fitb['outfits'][outfit_index],
                             'image_paths': [image_path]}
            gen_file[uid][oid] = outfit_result

            replaced_item_index = test_fitb['outfits'][outfit_index].tolist().index(0)
            replaced_item_id = test_grd[int(oid)]['outfits'][replaced_item_index]
            grd_file[uid][oid] = {
                'outfits': test_grd[int(oid)]['outfits'],
                'image_paths': [os.path.join(args.img_folder_path, all_item_image_path[replaced_item_id])],
            }

    if len(prompt_list) > 0:
        for generated_image, save_path in tqdm(zip(unified_model.t2i_infer_without_saving(prompt_list), save_path_list), total=len(prompt_list)):
            image = Image.fromarray(generated_image)
            image.save(save_path)

            filename_without_ext = os.path.splitext(os.path.basename(save_path))[0]
            uid, oid, _ = filename_without_ext.split("_")
            incomplete_outfit_path = os.path.join(fitb_merge_save_path, f"{uid}_{oid}.jpg")
            target_item_path = grd_file[uid][oid]["image_paths"][0]
            shutil.copy(incomplete_outfit_path, os.path.join(output_path, f"{uid}_{oid}_outfit.jpg"))
            shutil.copy(target_item_path, os.path.join(output_path, f"{uid}_{oid}_grd.jpg"))

    np.save(gen_file_save_path, gen_file)
    np.save(grd_file_save_path, grd_file)


def merge_images_and_save(images, path):
    images = [Image.open(f) for f in images if isinstance(f, str)]
    # max image number is 9
    images = images[:9]

    def resize_image(img):
        # 获取原始尺寸
        w, h = img.size
        # 计算缩放比例
        ratio = 291.0 / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # 使用LANCZOS重采样方法进行缩放
        resized_img = img.resize((new_w, new_h), Image.LANCZOS)
        return resized_img

    # 计算网格布局
    cols = math.ceil(math.sqrt(len(images)))
    resized_images = [resize_image(img) for img in images]

    # 计算合并图像的尺寸
    max_width = max(img.width for img in resized_images)
    max_height = max(img.height for img in resized_images)
    total_width = max_width * cols
    total_height = max_height * cols

    # 创建白色背景的合并图像
    merged_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

    # 粘贴图像
    for i in range(len(resized_images)):
        row = i // cols
        col = i % cols
        x = col * max_width
        y = row * max_height
        # 在网格中居中放置图像
        x_offset = (max_width - resized_images[i].width) // 2
        y_offset = (max_height - resized_images[i].height) // 2
        merged_image.paste(resized_images[i], (x + x_offset, y + y_offset))

    # 最后将整个图像缩放到512x512
    final_image = merged_image.resize((512, 512), Image.LANCZOS)
    final_image.save(path)


def extract_item_description_via_phi4(pred_answer, target_cate_str):
    description = ollama.chat(
        model="phi4",
        messages=[
            {'role': 'system',
             'content': "You are a fashion item extractor. Extract ONLY the specific recommended item from the "
                        "provided category (target_cate_str). Create a simple description starting with 'A' or "
                        "'An', focusing on key features (color, style, material, design elements). Ignore styling "
                        "suggestions and combinations. Always end with ', white background'. If no description can "
                        "be extracted, return A {provided category}, white background.",
             },
            {
                'role': 'user',
                'content': "Extract description of women's boot from the following sentences:\nTo complement your "
                           "striking black and white striped blouse paired with the eye-catching yellow "
                           "leather-like pants, I suggest opting for sleek ankle boots that echo some elements "
                           "from the rest of your ensemble. Consider black leather boots to keep it classic and "
                           "chic, or maybe something with metallic details like gold studs to tie in nicely with "
                           "your current booties. This would not only match well with both the blazer and pants "
                           "but also add a cohesive touch while allowing those yellow pants to remain the standout "
                           "piece.",
            },
            {
                'role': 'assistant',
                'content': "A black leather ankle boots, white background",
            },
            {
                'role': 'user',
                'content': "Extract description of dress from the following sentences:\n  To switch things up for "
                           "another occasion, I would recommend adding a statement piece to the outfit, such as a "
                           "statement necklace, statement earrings, or a bold statement bracelet. These "
                           "accessories can help elevate the look and make it more eye-catching and fashionable. "
                           "Additionally, you can also consider adding a pop of color to the outfit by wearing "
                           "bright or contrasting colors, like a bright pink coat or a bold pink shoe. This will help create",
            },
            {
                'role': 'assistant',
                'content': "A dress, white background",
            },
            {
                'role': 'user',
                'content': f"Extract description of {target_cate_str} from the following sentences:\n{pred_answer}"
            }
        ]
    )["message"]["content"]
    return description

if __name__ == "__main__":
    main()
