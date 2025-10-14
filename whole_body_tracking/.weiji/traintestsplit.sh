#!/bin/bash

SOURCE_DIR="./artifacts/Data10k"
TRAIN_DIR="./artifacts/Data10k-train"
TEST_DIR="./artifacts/Data10k-test"

# 创建目标目录
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

# 获取所有文件并随机排序
files=($(ls "$SOURCE_DIR" | shuf))

# 计算数量
total=${#files[@]}
train_count=$((total * 9 / 10))

echo "Total files: $total"
echo "Train files: $train_count"
echo "Test files: $((total - train_count))"

echo "Building train set..."
# 复制训练集
for ((i=0; i<train_count; i++)); do
    # echo "Copying ${files[i]} to $TRAIN_DIR"
    cp -r "$SOURCE_DIR/${files[i]}" "$TRAIN_DIR/"
done

echo "Building test set..."
# 复制测试集
for ((i=train_count; i<total; i++)); do
    # echo "Copying ${files[i]} to $TEST_DIR"
    cp -r "$SOURCE_DIR/${files[i]}" "$TEST_DIR/"
done

echo "完成！训练集: $train_count 个文件，测试集: $((total - train_count)) 个文件"