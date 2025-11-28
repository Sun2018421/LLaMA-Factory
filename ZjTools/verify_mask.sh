#!/bin/bash
# 验证mask索引是否与SFT训练数据对应的脚本
# 用法: ./verify_mask.sh <sample_idx>
# 例如: ./verify_mask.sh 0

SAMPLE_IDX="5 10 20 30"

echo "=========================================="
echo "验证样本 $SAMPLE_IDX 的Mask索引对应关系"
echo "=========================================="

# 设置路径
MASK_DIR="saves/qwen3-4b/full/attr_temp1.0/masks"
MODEL_PATH="/mnt/zj-gpfs/home/whs/model/Qwen3-4B-Instruct-2507"
DATASET="math_cot"
TEMPLATE="qwen"
CUTOFF_LEN=18000

# 运行验证脚本
python verify_mask_index.py \
    --batch_samples $SAMPLE_IDX \
    --mask_dir $MASK_DIR \
    --model_path $MODEL_PATH \
    --dataset $DATASET \
    --template $TEMPLATE \
    --cutoff_len $CUTOFF_LEN \

