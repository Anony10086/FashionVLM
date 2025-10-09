#!/bin/bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
# 定义模型方法变量
export METHOD_NAME='llava_onevision_7b_ov_chat_finetune'  # "llava_onevision_7b_ov_chat_finetune" | "llava_onevision_05b_si_finetune"

TASKS=("basic_recommendation" "personalized_recommendation" "alternative_recommendation")

echo "=================================================="
i=1
for TASK_NAME in "${TASKS[@]}"; do
    echo "--- Starting Task: $TASK_NAME ---"
    
    # 1. Infer/Recommend
    echo "${i}.1. Using $METHOD_NAME $TASK_NAME to recommend"
    python3 evaluate/infer_llava_next.py --method "$METHOD_NAME" --task "$TASK_NAME"
    
    # 2. Generate Image
    echo "${i}.2. Generating $METHOD_NAME $TASK_NAME Image"
    python3 evaluate/generate_image_fashionvlm.py --method "$METHOD_NAME" --task "$TASK_NAME"
    
    # 3. Evaluate
    echo "${i}.3. Evaluating $METHOD_NAME on $TASK_NAME"
    python3 evaluate/evaluate_multiple_tasks.py --method "$METHOD_NAME" --task "$TASK_NAME"

    echo "--- Task $TASK_NAME Complete ---"
    echo "--------------------------------------------------"
    i=$((i + 1))
done