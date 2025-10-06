import os

from omegaconf import OmegaConf
import torch
from show_o.models import Showo


config = OmegaConf.load("/mnt/d/PostDoc/fifth paper/code/FashionVLM/show_o/outputs/FashionVLM-2025-03-30/config_infer.yaml")

fashion_vlm = Showo(**config.model.showo)
path = os.path.join(config.experiment.output_dir, "pytorch_model.bin")
print(f"Resuming from checkpoint {path}")
state_dict = torch.load(path)
fashion_vlm.load_state_dict(state_dict, strict=False)

output_path = "/mnt/d/PostDoc/fifth paper/code/FashionVLM/show_o/outputs/fashion_vlm_huggingface"
fashion_vlm.save_pretrained(
    save_directory=output_path,
    safe_serialization=True,  # 使用 safetensors 格式
    push_to_hub=False  # 不上传到 Hugging Face，如果需要上传设置为 True
)
