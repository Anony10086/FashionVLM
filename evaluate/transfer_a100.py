import os
import json
from PIL import Image
from typing import List
from shutil import copy


def create_image_grid(image_paths: List[str], output_path: str, grid_size: int = 2) -> None:
    """
    将多个图片合并为 4 宫格图片，少于 4 个则留空，确保透明背景被填充为白色，并实现等比缩放。

    参数:
    - image_paths: 输入图片路径列表
    - output_path: 输出 4 宫格图片路径
    - grid_size: 网格大小（默认 2x2）
    """
    images = []
    target_size = (256, 256)  # 每个格子的目标尺寸

    for path in image_paths[:4]:
        try:
            # 加载图片，保留透明性
            img = Image.open(path).convert('RGBA')

            # 如果图片有透明通道，将透明区域填充为白色
            if img.mode == 'RGBA':
                background = Image.new('RGBA', img.size, (255, 255, 255, 255))  # 白色背景
                img = Image.alpha_composite(background, img)

            # 转换为 RGB 模式
            img = img.convert('RGB')
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            if original_width >= original_height:
                # 宽度是长边，调整宽度到 256
                new_width = 256
                new_height = int(256 / aspect_ratio)
            else:
                # 高度是长边，调整高度到 256
                new_height = 256
                new_width = int(256 * aspect_ratio)

            # 等比缩放
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            # img.thumbnail(target_size, Image.Resampling.LANCZOS)  # 使用高质量缩放算法

            # 创建 256x256 的空白画布（白色背景）
            canvas = Image.new('RGB', target_size, (255, 255, 255))

            # 计算居中位置
            offset_x = (target_size[0] - img.size[0]) // 2
            offset_y = (target_size[1] - img.size[1]) // 2

            # 将缩放后的图片居中贴到画布上
            canvas.paste(img, (offset_x, offset_y))

            images.append(canvas)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            images.append(None)

    # 如果图片不足 4 张，补空
    while len(images) < 4:
        images.append(None)

    # 创建空白画布（512x512，白色背景）
    grid_image = Image.new('RGB', (512, 512), (255, 255, 255))  # 白色背景

    # 按 2x2 排列贴图
    for idx, img in enumerate(images):
        if img is not None:
            x = (idx % 2) * 256
            y = (idx // 2) * 256
            grid_image.paste(img, (x, y))

    # 保存图片为 JPG 格式
    grid_image.save(output_path, quality=95)  # 设置质量为 95，避免过度压缩


output_dir = '/mnt/d/PostDoc/fifth paper/code/FashionVLM/datasets/FashionRec/data/a100/temp'

# A100
root = '/mnt/d/PhD_Study/second paper/OutfitMatcher/data/A100'
with open(os.path.join(root, 'fill_in_the_blank.json'), 'r') as f:
    fitb_data = json.load(f)

for i, dict_data in enumerate(fitb_data):
    # process question
    partial_outfit = dict_data['question']
    partial_outfit_path = [os.path.join(root, 'image', x.split('_')[1] + '.jpg') for x in partial_outfit]

    output_path = os.path.join(output_dir, f'{i}_0.jpg')
    create_image_grid(partial_outfit_path[:4], output_path)

    # Process answers
    for j, candidate in enumerate(dict_data['answers'], start=1):
        candidate_path = os.path.join(root, 'image', candidate.split('_')[1] + '.jpg')
        copy(candidate_path, os.path.join(output_dir, f'{i}_{j}.jpg'))
    answer_type = dict_data['answers'][0].split('_')[0].lower()

    # process query
    output_json = {
        'key': f'a100_{i}',
        'gt': dict_data['gt'],
        'gt_distribution': dict_data['gt_distribution'],
        'conversation': [
            {
                'from': 'human',
                'value': f'I uploaded a picture of my outfit. What {answer_type} would you recommend to match them?'
            }
        ]
    }
    with open(os.path.join(output_dir, f'{i}.json'), 'w') as f:
        json.dump(output_json, f)


# A100_pro
# root = '/mnt/d/PhD_Study/second paper/OutfitMatcher/data/A100_Pro'
# with open(os.path.join(root, 'fill_in_the_blank.json'), 'r') as f:
#     fitb_data = json.load(f)
#
# for i, dict_data in enumerate(fitb_data):
#     # process question
#     partial_outfit = dict_data['question']
#     partial_outfit_path = [os.path.join(root, 'image', x.split('_')[1] + '.jpg') for x in partial_outfit]
#
#     output_path = os.path.join(output_dir, f'{i}_0.jpg')
#     create_image_grid(partial_outfit_path[:4], output_path)
#
#     # Process answers
#     for j, candidate in enumerate(dict_data['answers'], start=1):
#         candidate_path = os.path.join(root, 'image', candidate.split('_')[1] + '.jpg')
#         copy(candidate_path, os.path.join(output_dir, f'{i}_{j}.jpg'))
#     answer_type = dict_data['answers'][0].split('_')[0].lower()
#
#     # process query
#     output_json = {
#         'key': f'a100_{i}',
#         'gt': dict_data['gt'],
#         'gt_distribution': dict_data['gt_distribution'],
#         'conversation': [
#             {
#                 'from': 'human',
#                 'value': f'I uploaded a picture of my outfit. What {answer_type} would you recommend to match them?'
#             }
#         ]
#     }
#     with open(os.path.join(output_dir, f'{i}.json'), 'w') as f:
#         json.dump(output_json, f)
