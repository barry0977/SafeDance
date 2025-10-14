#!/bin/bash
# 确保目标文件夹 data 存在
mkdir -p artifacts/PBHC_data   # 确保 a/b 存在
for dir in artifacts/*/; do
    subdir=$(basename "$dir")
    if [ "$subdir" != "PBHC_data" ]; then
        mv "$dir" "artifacts/PBHC_data/"
    fi
done

