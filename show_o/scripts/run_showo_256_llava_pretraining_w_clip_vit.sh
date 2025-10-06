#!/bin/bash

nohup accelerate launch \
    --config_file accelerate_configs/1_gpu.yaml \
    --main_process_port=8888 \
    training/train_w_clip_vit.py \
    config=configs_docker/showo_instruction_tuning_1_w_clip_vit.yaml \
    > logs/run_showo_256_llava_pretraining_w_clip_vit.txt 2>&1 &

echo "Process started with PID: $!"