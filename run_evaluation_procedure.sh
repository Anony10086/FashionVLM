#!/bin/bash
# 取消代理设置
unset http_proxy
unset https_proxy

# 定义模型方法变量
export METHOD_NAME='llava_onevision_05b_si'

echo "=================================================="

# 1. basic_recommendation 任务
echo "1.1. Using $METHOD_NAME basic_recommendation to recommend"
python3 evaluate/infer_llava_next.py --model $METHOD_NAME --task basic_recommendation
echo "1.2. Generating $METHOD_NAME basic_recommendation Image"
python3 evaluate/generate_image_fashionvlm.py --method $METHOD_NAME --task basic_recommendation
echo "1.3. Evaluating $METHOD_NAME on basic_recommendation"
python3 evaluate/evaluate_multiple_tasks.py --method $METHOD_NAME --task basic_recommendation

echo "--------------------------------------------------"

# 2. personalized_recommendation 任务
echo "2.1. Using $METHOD_NAME personalized_recommendation to recommend"
python3 evaluate/infer_llava_next.py --model $METHOD_NAME --task personalized_recommendation
echo "2.2. Generating $METHOD_NAME personalized_recommendation Image"
python3 evaluate/generate_image_fashionvlm.py --method $METHOD_NAME --task personalized_recommendation
echo "2.3. Evaluating $METHOD_NAME on personalized_recommendation"
python3 evaluate/evaluate_multiple_tasks.py --method $METHOD_NAME --task personalized_recommendation

echo "--------------------------------------------------"

# 3. alternative_recommendation 任务
echo "3.1. Using $METHOD_NAME alternative_recommendation to recommend"
python3 evaluate/infer_llava_next.py --model $METHOD_NAME --task alternative_recommendation
echo "3.2. Generating $METHOD_NAME alternative_recommendation Image"
python3 evaluate/generate_image_fashionvlm.py --method $METHOD_NAME --task alternative_recommendation
echo "3.3. Evaluating $METHOD_NAME on alternative_recommendation"
python3 evaluate/evaluate_multiple_tasks.py --method $METHOD_NAME --task alternative_recommendation

echo "=================================================="