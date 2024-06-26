#!/bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3

base_dir="/home/wangguisen/projects/Langchain-Chatchat"
pripath="$base_dir/rag_evaluation"
eval_data_path="$base_dir/kimi_response_for_51docs_on_13questions_20240523.json"
output_path="$pripath/output"
emb_model_path="/home/wangguisen/models/bge-large-zh-v1.5"
generate_model_path="/home/wangguisen/models/Qwen1.5-14B-Chat"

cd $base_dir

python -m rag_evaluation \
    --eval_data_path $eval_data_path \
    --knowledge_base test_eval \
    --output_path $output_path \
    --emb_model_path $emb_model_path \
    --device cuda \
    --kb_top_k 5 \
    --score_threshold 0.8

# nohup sh ./rag_evaluation/run.sh > ./rag_evaluation/output/run.log 2>&1 &

# --generate_model_path $generate_model_path \