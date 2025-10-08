from typing import List

import PIL
import torch
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import pad
from PIL import Image
# import open_clip
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from compatibility_net import FashionEvaluator


class ResizeSquareImage(torch.nn.Module):
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR, fill=(255, 255, 255), padding_mode="constant"):
        super().__init__()
        self.size = size  # 目标最长边
        self.interpolation = interpolation
        self.fill = fill  # 填充值，白色 (255, 255, 255)
        self.padding_mode = padding_mode  # 填充模式，默认 "constant"

    def __call__(self, img):
        # 确保输入是 PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image, but got {type(img)}")

        # 步骤 1：调整大小（最长边为 self.size）
        width, height = img.size
        max_side = max(width, height)
        scale = self.size / max_side
        new_width = round(width * scale)
        new_height = round(height * scale)
        resize_transform = transforms.Resize((new_height, new_width), interpolation=self.interpolation)
        img = resize_transform(img)

        # 步骤 2：填充到正方形
        # 重新获取调整后的宽高
        width, height = img.size
        max_side = max(width, height)  # 应该是 self.size
        if width == height:
            return img

        # 计算填充量
        pad_width = max_side - width
        pad_height = max_side - height
        padding = [
            pad_width // 2,  # 左
            pad_height // 2,  # 上
            pad_width - (pad_width // 2),  # 右
            pad_height - (pad_height // 2)  # 下
        ]
        img = pad(img, padding, self.fill, self.padding_mode)
        return img


class ConvertRGB(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        # 确保输入是 PIL Image
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image, but got {type(img)}")

        # 如果图像是 RGBA 模式，填充透明区域为白色
        if img.mode == 'RGBA':
            background = Image.new('RGBA', img.size, (255, 255, 255, 255))  # 白色背景
            img = Image.alpha_composite(background, img)
            img = img.convert('RGB')  # 转换为 RGB 模式

        # 如果图像不是 RGBA，但也不是 RGB，转换为 RGB
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CLIPScore:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: List) -> Tensor:
        inputs = self.processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device=self.device)
        img_features = self.model.get_image_features(**inputs)
        img_features = F.normalize(img_features, p=2, dim=1)
        return img_features

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> Tensor:
        inputs = self.processor.tokenizer(
            texts,
            truncation=True,
            max_length=77,
            padding=True,
            return_tensors="pt"
        ).to(device=self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=1)
        return text_features


class SentenceBertScore:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map=device)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_sentence(self, sentences: List[str]):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input.to(self.model.device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=-1)
        return sentence_embeddings

    def calculate_sentence_bert_score(self, sentences1: List[str], sentences2: List[str]):
        sentence_embeddings1 = self.embed_sentence(sentences1)
        sentence_embeddings2 = self.embed_sentence(sentences2)

        sim_score = 100 * F.cosine_similarity(sentence_embeddings1, sentence_embeddings2)
        return sim_score


class CompatibilityScore:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.evaluator = FashionEvaluator(cnn_feat_dim=1024).to(device)
        state_dict = torch.load(model_path, map_location=device)
        self.evaluator.load_state_dict(state_dict)
        self.evaluator.eval()

    @torch.no_grad()
    def evaluate(self, outfit_feats: Tensor):
        prediction = self.evaluator(outfit_feats)
        scores = torch.nn.Sigmoid()(prediction)
        return scores
