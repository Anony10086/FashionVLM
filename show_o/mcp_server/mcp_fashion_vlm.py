import os
from dotenv import load_dotenv
import numpy as np
from itertools import combinations
from typing import Dict, Any, List, Optional
import pandas as pd

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import uvicorn
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPProcessor, CLIPModel
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.routing import Route, Mount
from starlette.applications import Starlette
from openai import AsyncOpenAI

from inference_showo import ShowoModel
from training.prompting_utils import UniversalPrompting


# Load environment variables
load_dotenv()
FASHION_DATA_ROOT = os.getenv("FASHION_DATA_ROOT", "/mnt/d/PostDoc/fifth paper/code/FashionVLM/datasets/FashionRec")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
openai = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)
VALID_CATEGORIES = [
    'Pants', 'Coats', 'Cross-body bags', 'Shirts', 'Hats & caps', 'Sneakers', 'Jeans', 'Boots', 'Dresses', 'Sandals',
    'T-shirts & vests', 'Knitwear', 'Skirts', 'Earrings', 'Hats', 'Sweaters & knitwear', 'Loafers', 'Ballet flats',
    'Espadrilles', 'Tote bags', 'Shoulder bags', 'Slides & flip flops', 'Pumps', 'Necklaces', 'Polo shirts', 'Suits',
    'Oxford shoes', 'Bracelets', 'Jackets', 'Tops', 'Rings', 'Mules', 'Luggage & holdalls', 'Brogues', 'Activewear',
    'Belts', 'Derby shoes', 'Mini bags', 'Watches', 'Backpacks', 'Denim', 'Laptop bags & briefcases', 'Clutch bags',
    'Clutches', 'Lingerie & Nightwear', 'Skiwear', 'Sunglasses', 'Ties & bow ties', 'Shorts', 'Scarves', 'Messenger bags'
]


###################################
#########Loading Data##############
###################################
# Load item metadata
items_df = pd.read_parquet(f"{FASHION_DATA_ROOT}/meta/items_lite.parquet").set_index("item_id")
outfits_df = pd.read_parquet(f"{FASHION_DATA_ROOT}/meta/outfits_lite.parquet").set_index("outfit_id")
users_df = pd.read_parquet(f"{FASHION_DATA_ROOT}/meta/users_lite.parquet").set_index("user_id")
image_paths = items_df["path"].to_dict()

###################################
#########Loading Model#############
###################################
# Load CLIP model and processor
print("Loading CLIP Model")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
clip_model.eval()

print("Loading Fashion VLM params")
fashion_vlm = ShowoModel(
    config="/mnt/d/PostDoc/fifth paper/code/FashionVLM/show_o/outputs/FashionVLM-2025-03-30/config_infer.yaml",
    max_new_tokens=1000,
    temperature=0.8,
    top_k=1,
    load_from_showo=False,
    save_dir="/mnt/d/PostDoc/fifth paper/code/FashionM3/generated_images"
)

resolution = 512
image_transform = transforms.Compose([
    transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop((resolution, resolution)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
])

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', padding_side="left")
uni_prompting = UniversalPrompting(
    tokenizer,
    max_text_len=128,
    special_tokens=(
        "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
    ),
    ignore_id=-100, cond_dropout_prob=0.1
)


class InteractionDataManager:
    def __init__(self, users_df, outfits_df, items_df):
        """
        初始化类，加载数据并设置基本参数

        参数:
        - users_file: 用户数据文件路径 (parquet)
        - outfits_file: Outfit 数据文件路径 (parquet)
        - items_file: 单品数据文件路径 (parquet)
        """
        self.users_df = users_df
        self.outfits_df = outfits_df
        self.items_df = items_df

        # 创建映射
        self.item_id_to_index = {item_id: index for index, item_id in enumerate(self.items_df.index)}
        self.index_to_item_id = {index: item_id for index, item_id in enumerate(self.items_df.index)}
        self.user_id_to_index = {user_id: index for index, user_id in enumerate(self.users_df.index)}
        self.index_to_user_id = {index: user_id for index, user_id in enumerate(self.users_df.index)}
        self.outfit_ids_dict = self.outfits_df['item_ids'].to_dict()  # get outfit's item ids from outfit id
        self.item_category_dict = self.items_df['category'].to_dict()  # get item's category from item id
        self.item_subcategory_dict = self.items_df['subcategory'].to_dict()  # get item's subcategory from item id
        self.n_items = len(self.items_df)
        self.n_users = len(self.users_df)

        self.user_outfit_pairs = []
        outfit_set = set(self.outfits_df.index)
        for uid, user in self.users_df.iterrows():
            oids = user.outfit_ids.split(",")
            self.user_outfit_pairs.extend([(uid, oid) for oid in oids if oid in outfit_set])

        # 预处理类别到物品ID的映射（使用groupby）
        self.subcategory_to_items = self.items_df.groupby('subcategory').apply(lambda x: set(x.index)).to_dict()

        # 预处理类别到物品索引的映射（优化查找效率）
        self.subcategory_to_indices = {}
        for subcategory, item_ids in self.subcategory_to_items.items():
            self.subcategory_to_indices[subcategory] = set([self.item_id_to_index[item_id]
                                                            for item_id in item_ids
                                                            if item_id in self.item_id_to_index])

        item_interaction_matrix_path = f'{FASHION_DATA_ROOT}/data/personalized_recommendation/temp_matrix/item_matrix.npz'
        try:
            self.load_matrix('item', item_interaction_matrix_path)
        except FileNotFoundError:
            self.build_item_interaction_matrix()
            self.save_matrix('item', item_interaction_matrix_path)

        user_item_interaction_matrix_path = f'{FASHION_DATA_ROOT}/data/personalized_recommendation/temp_matrix/user_item_matrix.npz'
        try:
            self.load_matrix('user_item', user_item_interaction_matrix_path)
        except FileNotFoundError:
            self.build_user_item_interaction_matrix()
            self.save_matrix('user_item', user_item_interaction_matrix_path)

        # 加载item clip features
        with open(f"{FASHION_DATA_ROOT}/meta/clip_features.pkl", "rb") as f:
            print("Loading Fashion Features...")
            self.clip_features = pickle.load(f)
            print("Loading Fashion Features Successfully")

        # Prepare embeddings and item IDs
        self.item_ids = list(self.clip_features.keys())
        self.image_embeddings = np.array([self.clip_features[item_id]["image_embeds"] for item_id in self.item_ids])

    def save_matrix(self, matrix_type, filepath):
        """
        保存矩阵到文件

        参数:
        - matrix_type: 'item' 或 'user_item'，指定保存的矩阵类型
        - filepath: 保存路径 (例如 'temp/item_matrix.npz')
        """
        if matrix_type == 'item':
            matrix = self.item_interaction_matrix
        elif matrix_type == 'user_item':
            matrix = self.user_item_interaction_matrix
        else:
            raise ValueError("matrix_type must be 'item' or 'user_item'")

        if matrix is None:
            raise ValueError(f"{matrix_type} matrix has not been built yet.")

        sparse.save_npz(filepath, matrix)
        print(f"Saved {matrix_type} matrix to {filepath}")

    def load_matrix(self, matrix_type, filepath):
        """
        从文件加载矩阵

        参数:
        - matrix_type: 'item' 或 'user_item'，指定加载的矩阵类型
        - filepath: 加载路径 (例如 'temp/item_matrix.npz')
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")

        matrix = sparse.load_npz(filepath)
        if matrix_type == 'item':
            self.item_interaction_matrix = matrix
        elif matrix_type == 'user_item':
            self.user_item_interaction_matrix = matrix
        else:
            raise ValueError("matrix_type must be 'item' or 'user_item'")

        print(f"Loaded {matrix_type} matrix from {filepath}")
        return matrix

    def build_item_interaction_matrix(self):
        """构建 Item-Item 交互矩阵"""
        # 初始化单品交互矩阵
        self.item_interaction_matrix = sparse.lil_matrix((self.n_items, self.n_items), dtype=int)

        for index, outfit in self.outfits_df.iterrows():
            item_ids = outfit['item_ids'].split(',')
            # 记录 item 对的共现
            for item_id1, item_id2 in combinations(item_ids, r=2):
                if item_id1 in self.item_id_to_index and item_id2 in self.item_id_to_index:
                    idx1 = self.item_id_to_index[item_id1]
                    idx2 = self.item_id_to_index[item_id2]
                    self.item_interaction_matrix[idx1, idx2] += 1
                    self.item_interaction_matrix[idx2, idx1] += 1  # 无序对称

        # 转换为 CSR 格式
        self.item_interaction_matrix = self.item_interaction_matrix.tocsr()
        return self.item_interaction_matrix

    def build_user_item_interaction_matrix(self):
        """构建 User-Item 交互矩阵"""
        # 初始化用户-单品交互矩阵
        self.user_item_interaction_matrix = sparse.lil_matrix((self.n_users, self.n_items), dtype=int)

        for uid, user in self.users_df.iterrows():
            oids = user["outfit_ids"].split(",")
            outfits = self.outfits_df.loc[self.outfits_df.index.isin(oids)]
            for oid, outfit in outfits.iterrows():
                item_ids = outfit['item_ids'].split(',')
                # 记录 user-item 对的出现
                for iid in item_ids:
                    if iid in self.item_id_to_index:
                        uidx = self.user_id_to_index[uid]
                        iidx = self.item_id_to_index[iid]
                        self.user_item_interaction_matrix[uidx, iidx] += 1

        # 转换为 CSR 格式
        self.user_item_interaction_matrix = self.user_item_interaction_matrix.tocsr()
        return self.user_item_interaction_matrix

    def _process_interactions_for_category(
            self,
            matrix,
            given_id,
            category_indices,
            id_to_index
    ):
        """
        处理单个实体与目标类别的交互

        参数:
        - matrix: 交互矩阵
        - given_id: 给定的实体ID（用户或物品）
        - category_indices: 目标类别的物品索引集合

        返回:
        - 交互列表，每个元素为一个包含item_id、interaction_count和score的字典
        """
        interactions = []

        given_index = id_to_index[given_id]
        row = matrix[given_index]

        # 提取该行的非零元素
        row_start = row.indptr[0]
        row_end = row.indptr[1]
        col_indices = row.indices[row_start:row_end]
        data_values = row.data[row_start:row_end]

        # 筛选出属于目标类别的物品
        for col_idx, value in zip(col_indices, data_values):
            # 检查是否为目标类别的物品
            if col_idx in category_indices:
                # 获取物品ID
                output_id = self.index_to_item_id[col_idx]
                interactions.append({
                    'item_id': output_id,
                    'interaction_count': int(value),
                    'score': 0.0
                })

        return interactions

    def get_item_category_interactions(
        self,
        target_category: str,
        given_ids: List[str],
        query_type='item',  # item or user
        top_k=None,
    ):
        """
        获取指定实体（用户或单品）与目标类别的所有交互情况

        参数:
        - target_category: 待查询的subcategory
        - given_ids: List of 目标类别
        - query_type: 查询的类别， item或user
        - top_k: 返回交互次数最多的前k个物品, 如果是None直接全部返回

        返回:
        - 列表，包含与目标类别的交互统计信息，按交互次数排序
        """
        if query_type == 'item':
            matrix = self.item_interaction_matrix
            id_to_index = self.item_id_to_index
        elif query_type == 'user':
            matrix = self.user_item_interaction_matrix
            id_to_index = self.user_id_to_index
        else:
            print(f'query_type must be either item or user but got {query_type}')
            return []

        # 收集所有交互记录
        all_interactions = []
        category = target_category
        category_indices = self.subcategory_to_indices.get(category, set())  # 获取该类别的所有物品索引

        # 获取该实体的所有交互
        for given_id in given_ids:
            interactions = self._process_interactions_for_category(
                matrix, given_id, category_indices, id_to_index
            )
            # 将交互添加到结果列表
            all_interactions.extend(interactions)

        # 合并相同物品的交互次数
        item_interactions = {}
        for interaction in all_interactions:
            item_id = interaction['item_id']
            count = interaction['interaction_count']

            if item_id in item_interactions:
                item_interactions[item_id] += count
            else:
                item_interactions[item_id] = count

        # 转换为结果格式
        merged_interactions = [
            {'item_id': item_id, 'interaction_count': count, 'score': 0.0}
            for item_id, count in item_interactions.items()
        ]

        # 排序
        if merged_interactions:
            merged_interactions.sort(key=lambda x: x['interaction_count'], reverse=True)

        # 截取top-k
        if top_k and merged_interactions:
            merged_interactions = merged_interactions[:top_k]

        # 存储结果
        return merged_interactions

    def rank_by_similarity(self, item_interactions, user_interactions, beta=2.0):
        """
        计算用户交互项与商品交互项的相似度并排序
        """
        def get_combined_features(feature_dict):
            return (feature_dict['image_embeds'] + feature_dict['text_embeds']) / 2

        if not item_interactions:
            return user_interactions

        item_feature_list = []
        for item in item_interactions:
            item_id = item['item_id']
            if item_id not in self.clip_features:
                raise ValueError(f"Didn't find clip feature of item with id: {item_id}")

            item_features = get_combined_features(self.clip_features[item_id])
            item_feature_list.append(item_features)

        weights = np.array([x['interaction_count'] for x in item_interactions], dtype=np.float32)
        weights = weights / np.sum(weights)
        item_feature = np.sum(np.stack(item_feature_list, axis=0) * weights[:, np.newaxis], axis=0).reshape(1, -1)

        max_count = max((user_item.get('interaction_count', 1) for user_item in user_interactions), default=1)
        for user_item in user_interactions:
            user_item_id = user_item['item_id']
            if user_item_id not in self.clip_features:
                raise ValueError(f"Didn't find clip feature of item with id: {user_item_id}")

            user_item_features = get_combined_features(self.clip_features[user_item_id]).reshape(1, -1)
            similarity = cosine_similarity(user_item_features, item_feature).item()
            interaction_count = user_item['interaction_count']
            count_factor = (interaction_count / max_count) * beta + 1
            user_item['score'] = float(similarity) * count_factor

        user_interactions.sort(key=lambda x: x.get('score', 0), reverse=True)
        return user_interactions


data_manager = InteractionDataManager(users_df, outfits_df, items_df)
mcp = FastMCP('fashion-vlm-server')


async def compute_text_embedding(text: str) -> np.ndarray:
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs).numpy()
    return text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)


async def find_most_similar_image(text_embedding: np.ndarray) -> Dict[str, Any]:
    similarities = np.dot(data_manager.image_embeddings, text_embedding.T).flatten()
    most_similar_idx = np.argmax(similarities)
    most_similar_item_id = data_manager.item_ids[most_similar_idx]
    return {
        "image_path": image_paths[most_similar_item_id],
        "similarity": float(similarities[most_similar_idx])
    }


@mcp.tool()
async def retrieve_image(text: str) -> Dict[str, Any]:
    """Search for the most similar fashion image based on a text description.

    Args:
        text (str): Text description of the fashion item to search.
    """
    print(f"Searching for {text}")
    text_embedding = await compute_text_embedding(text)
    return await find_most_similar_image(text_embedding)


def get_recommendation(query, image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)
    prompt = uni_prompting.text_tokenizer(['USER: \n' + query + ' ASSISTANT:'])['input_ids'][0]
    prompt = torch.tensor(prompt).unsqueeze(0)
    results = fashion_vlm.mmu_infer_tensor(image, prompt)
    response = results[0]
    return response


@mcp.tool()
async def fashion_recommend(query: str, image_path: str, target_category: str, user_id: Optional[str], list_of_items: List[str]) -> Dict[str, str]:
    """Generate fashion recommendations based on a user's query and uploaded image.

    This function processes the recommendation in the following steps:
    1. Retrieves the user's interaction history for the specified target category using user_id, target_category, and list_of_items.
    2. Summarizes the user's preferences for the target category by analyzing descriptions of previously interacted fashion items via a language model.
    3. Appends the summarized preference (as a single sentence) to the query and processes it with the uploaded image using a Fashion Vision-Language Model (VLM).
    4. Returns the personalized recommendation along with the derived user preference.

    The target_category is inferred from the query (e.g., "I want a skirt ..." implies "Skirts") and must belong to a predefined list of valid categories.

    Args:
        query (str): A complete sentence explicitly stating the user's desired fashion item (e.g., "I want a skirt for summer"). Must be in English.
        image_path (str): File path to the user-uploaded image, provided via the prompt.
        target_category (str): The specific fashion category of interest, derived from the query (e.g., "Skirts"). Must be in valid categories.
        user_id (str): Unique identifier for the user, provided via the prompt.
        list_of_items (List[str]): List of item IDs used to filter the user's interaction history, provided via the prompt.

    Returns:
        Dict[str, str]: A dictionary containing:
            - "recommendation": The personalized fashion recommendation text.
            - "user_preference": The summarized user preference sentence.

    Valid Categories:
        ['Pants', 'Coats', 'Cross-body bags', 'Shirts', 'Hats & caps', 'Sneakers', 'Jeans', 'Boots', 'Dresses', 'Sandals',
         'T-shirts & vests', 'Knitwear', 'Skirts', 'Earrings', 'Hats', 'Sweaters & knitwear', 'Loafers', 'Ballet flats',
         'Espadrilles', 'Tote bags', 'Shoulder bags', 'Slides & flip flops', 'Pumps', 'Necklaces', 'Polo shirts', 'Suits',
         'Oxford shoes', 'Bracelets', 'Jackets', 'Tops', 'Rings', 'Mules', 'Luggage & holdalls', 'Brogues', 'Activewear',
         'Belts', 'Derby shoes', 'Mini bags', 'Watches', 'Backpacks', 'Denim', 'Laptop bags & briefcases', 'Clutch bags',
         'Clutches', 'Lingerie & Nightwear', 'Skiwear', 'Sunglasses', 'Ties & bow ties', 'Shorts', 'Scarves', 'Messenger bags']
    """
    def get_item(item_id: str) -> pd.Series:
        return data_manager.items_df.loc[item_id]

    # If no image uploaded, we should use fashion_recommend_without_image
    if image_path == "":
        recommendation = await fashion_recommend_without_image(query)
        return {
            "recommendation": recommendation,
            "user_preference": ""
        }

    # If no user_id provided or user_id not found in database
    if not user_id or user_id not in data_manager.user_id_to_index.keys():
        return {
            "recommendation": get_recommendation(query, image_path),
            "user_preference": ""
        }

    user_preference = ""
    if target_category in VALID_CATEGORIES:
        user_interaction_result = data_manager.get_item_category_interactions(
            target_category, [user_id], query_type='user'
        )

        if len(list_of_items) != 0:
            item_interaction_result = data_manager.get_item_category_interactions(
                target_category, list_of_items, query_type='item'
            )
        else:
            item_interaction_result = []

        descriptions_for_summary = []
        historical_image_path = []

        if len(user_interaction_result) >= 0:
            user_interaction_result = data_manager.rank_by_similarity(
                item_interaction_result,
                user_interaction_result
            )
            for x in user_interaction_result[:5]:
                item = get_item(x['item_id'])
                descriptions_for_summary.append(item['gen_description'])
                historical_image_path.append(os.path.abspath(item['path']))

        if descriptions_for_summary:
            user_message = f"Summary user's preference of {target_category} based on following descriptions of fashion items that user brought previously:"
            for x in descriptions_for_summary:
                user_message += f"\n{x}"
            # Get summary using OpenAI API call
            response = await openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"You are a user preference summary assistant. Your response is limited in one sentence, staring at 'I prefer ...'"},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1000,
            )
            user_preference = response.choices[0].message.content
            query += user_preference

    return {
        "recommendation": get_recommendation(query, image_path),
        "user_preference": user_preference
    }


@mcp.tool()
async def fashion_recommend_without_image(query: str) -> str:
    """Recommend fashion items sorely based on user's query.
    Output texts of fashion recommendation from model.

    Args:
        query (str): User's fashion related query including their recommendation request.
    """
    response = await openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a fashion stylist. You should answer user's fashion-related question, especially about fashion recommendation."},
            {"role": "user", "content": query}
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content


@mcp.tool()
async def image_generate(text: str) -> str:
    """"Generate image based on description. Output is path that saves generated image.

    Args:
        text (str): Descriptive text from user. Used for fashion image generation. English ONLY!
    """
    output_path = fashion_vlm.t2i_infer([text])[0]
    output_path = os.path.abspath(output_path)
    print(f"Generated image saved at {output_path}")
    return output_path


# 获取内部 Server 对象
mcp_server = mcp._mcp_server
sse_transport = SseServerTransport("/messages/")


# 处理 SSE 连接
async def handle_sse(request):
    print("Handling SSE connection")
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        read_stream, write_stream = streams
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )

# 定义路由
routes = [
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse_transport.handle_post_message),
]

# 创建 Starlette 应用
starlette_app = Starlette(routes=routes)


if __name__ == "__main__":
    print("Starting Fashion VLM server with HTTP and SSE...")
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000)

