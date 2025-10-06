from typing import List
import os
import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from models import Showo, MAGVITv2, CLIPVisionTower, get_mask_chedule
from training.prompting_utils import (UniversalPrompting,
                                      create_attention_mask_for_mmu,
                                      create_attention_mask_predict_next)
from training.utils import image_transform
from transformers import AutoTokenizer, CLIPImageProcessor


class ShowoModel:
    def __init__(self, config, temperature, top_k, max_new_tokens):
        self.config = OmegaConf.load(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        self.top_k = top_k  # retain only the top_k most likely tokens, clamp others to have 0 probability
        self.max_new_tokens = max_new_tokens

        #param for t2i
        self.save_dir = "generated_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.config.training.batch_size = 2
        self.config.training.guidance_scale = 5
        self.config.training.generation_timesteps = 50
        self._init_models()

    def _init_models(self):
        # 初始化Universal Prompting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.showo.llm_model_path,
            padding_side="left"
        )
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=self.config.dataset.preprocessing.max_seq_length,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
            ),
            ignore_id=-100, cond_dropout_prob=self.config.training.cond_dropout_prob
        )

        # 初始化VQ模型
        self.vq_model = MAGVITv2.from_pretrained(self.config.model.vq_model.vq_model_name).to(self.device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()

        # 初始化Vision Tower
        vision_tower_name = "openai/clip-vit-large-patch14-336"
        self.vision_tower = CLIPVisionTower(vision_tower_name).to(self.device)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)

        # 初始化Showo模型
        self.model = Showo.from_pretrained(self.config.model.showo.pretrained_model_path).to(self.device)
        self.model.eval()
        self.mask_token_id = self.model.config.mask_token_id

    def mmu_infer(self, image_path, question):
        image_ori = Image.open(image_path).convert("RGB")
        image = image_transform(image_ori, resolution=self.config.dataset.params.resolution).to(self.device)
        image = image.unsqueeze(0)

        # pixel_values = self.clip_image_processor.preprocess(image_ori, return_tensors="pt")["pixel_values"][0]
        image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)

        # without w_clip_vit
        input_ids = self.uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])['input_ids']
        input_ids = torch.tensor(input_ids).to(self.device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device),
            (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(
            input_ids.to(self.device),
            eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>'])
        )

        cont_toks_list = self.model.mmu_generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens, top_k=self.top_k,
            eot_token=self.uni_prompting.sptids_dict['<|eot|>']
        )

        cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

        text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)
        return text

    def t2i_infer(self, prompts: List[str]):
        output_path = []
        for step in tqdm(range(0, len(prompts), self.config.training.batch_size)):
            batch_prompt = prompts[step:step + self.config.training.batch_size]
            image_tokens = torch.ones((len(batch_prompt), self.config.model.showo.num_vq_tokens),
                                      dtype=torch.long, device=self.device) * self.mask_token_id
            input_ids, _ = self.uni_prompting((batch_prompt, image_tokens), 't2i_gen')
            if self.config.training.guidance_scale > 0:
                uncond_input_ids, _ = self.uni_prompting(([''] * len(batch_prompt), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(
                    torch.cat([input_ids, uncond_input_ids], dim=0),
                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True
                )
            else:
                uncond_input_ids = None
                attention_mask = create_attention_mask_predict_next(
                    input_ids,
                    pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True
                )
            if self.config.get("mask_schedule", None) is not None:
                schedule = self.config.mask_schedule.schedule
                args = self.config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = self.model.t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=self.config.training.guidance_scale,
                    temperature=self.config.training.get("generation_temperature", 1.0),
                    timesteps=self.config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=self.config.training.get("noise_type", "mask"),
                    seq_len=self.config.model.showo.num_vq_tokens,
                    uni_prompting=self.uni_prompting,
                    config=self.config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
            images = self.vq_model.decode_code(gen_token_ids)
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

            # 保存图片
            for idx, image in enumerate(images, start=1):
                image = Image.fromarray(image)
                # 使用时间戳和索引创建唯一的文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{timestamp}_{step + idx}.jpg"
                image_path = os.path.join(self.save_dir, image_filename)
                image.save(image_path)
                output_path.append(image_path)

        return output_path
