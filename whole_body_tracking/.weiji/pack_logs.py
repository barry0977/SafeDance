"""
适配新的log结构：logs/rsl_rl/{task_name}/{timestamp_experiment_name}/
直接记录需要保存的文件路径，然后直接打包得到tar文件
"""

import argparse
import glob
import math
import os
import re
import tarfile
import time


def extract_number(filename):
    """从文件名中提取数字部分并返回整数."""
    match = re.search(r"model_(\d+).pt", filename)
    if match:
        return int(match.group(1))
    else:
        return -1  # 如果没有匹配到数字，返回-1


def is_file_older_than_n_days(file_path, n_days):
    """检查文件的最后修改时间是否早于n天前."""
    if n_days == 0:  # 如果n_days为0，表示不跳过任何文件
        return False
    file_mtime = os.path.getmtime(file_path)
    current_time = time.time()
    time_diff = current_time - file_mtime
    return time_diff > (n_days * 86400)  # 86400秒 = 1天


def format_file_size(size_bytes):
    """将字节数转换为人类可读的文件大小格式."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def get_file_size(file_path):
    """获取文件大小（字节）."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0


def main(n_days, max_id, min_id, output_file_dir, auto_clear, copy_event):
    print("Starting script...")
    logs_dir = "logs/rsl_rl"
    output_dir = "output"

    # 检查 logs 目录是否存在
    if not os.path.isdir(logs_dir):
        print(f"Error: {logs_dir} is not a directory.")
        return

    # 初始化摘要字典和待添加文件列表
    summary = {}
    files_to_add = []
    total_copied = 0  # 记录复制的模型文件数量
    total_size = 0  # 记录总文件大小

    # 遍历 logs/rsl_rl 目录下的所有 task_name 目录
    print(f"Processing task directories in {logs_dir}...")
    for task_name in os.listdir(logs_dir):
        task_path = os.path.join(logs_dir, task_name)
        if not os.path.isdir(task_path):
            print(f"Skipping {task_path} as it is not a directory.")
            continue

        print(f"Processing task directory: {task_path}")
        if task_name not in summary:
            summary[task_name] = {}

        # 遍历 task_name 目录下的所有 experiment 目录
        print(f"Processing experiment directories in {task_path}...")
        for experiment_name in os.listdir(task_path):
            experiment_path = os.path.join(task_path, experiment_name)
            if not os.path.isdir(experiment_path):
                print(f"Skipping {experiment_path} as it is not a directory.")
                continue

            print(f"Processing experiment directory: {experiment_path}")

            # 查找所有 model_***.pt 文件
            print(f"Searching for model_*.pt files in {experiment_path}...")
            files = [f for f in os.listdir(experiment_path) if f.startswith("model_") and f.endswith(".pt")]

            if not files:
                print(f"No model_*.pt files found in {experiment_path}. Skipping.")
                continue

            # 提取所有文件的数字并过滤范围
            numbers = [extract_number(f) for f in files]
            numbers = [n for n in numbers if min_id <= n <= max_id]
            if not numbers:
                print(f"No valid model_*.pt files in {experiment_path}")
                continue

            max_number = max(numbers)
            if max_number == -1:
                print(f"Warning: No valid model_*.pt files in {experiment_path}")
                continue

            target_file = f"model_{max_number}.pt"
            src_file = os.path.join(experiment_path, target_file)

            # 检查文件是否在n天前创建
            if is_file_older_than_n_days(src_file, n_days):
                print(f"Skipping {target_file} as it was created more than {n_days} days ago.")
                continue

            # 添加模型文件到打包列表
            model_arcname = os.path.join(output_dir, task_name, experiment_name, target_file)
            files_to_add.append((src_file, model_arcname))
            summary[task_name][experiment_name] = target_file
            total_copied += 1

            # 处理事件文件
            if copy_event:
                target_event = "events.out.tfevents*"
                src_events = glob.glob(os.path.join(experiment_path, target_event))
                for event_src in src_events:
                    event_arcname = os.path.join(output_dir, task_name, experiment_name, os.path.basename(event_src))
                    files_to_add.append((event_src, event_arcname))

            # 处理git文件夹
            git_dir = "git"
            src_git = os.path.join(experiment_path, git_dir)
            if os.path.exists(src_git) and os.path.isdir(src_git):
                git_arcname = os.path.join(output_dir, task_name, experiment_name, git_dir)
                files_to_add.append((src_git, git_arcname))

            # 处理params文件夹
            params_dir = "params"
            src_params = os.path.join(experiment_path, params_dir)
            if os.path.exists(src_params) and os.path.isdir(src_params):
                params_arcname = os.path.join(output_dir, task_name, experiment_name, params_dir)
                files_to_add.append((src_params, params_arcname))

    # 计算总文件大小
    print(f"\nCalculating total file size...")
    for src_path, arcname in files_to_add:
        if os.path.isfile(src_path):
            file_size = get_file_size(src_path)
            total_size += file_size
        elif os.path.isdir(src_path):
            # 对于目录，递归计算所有文件的大小
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = get_file_size(file_path)
                    total_size += file_size

    # 打印摘要
    print("\nSummary of files to be archived:")
    for task_name, experiment_dict in summary.items():
        print(f"\nTask: {task_name}")
        for experiment_name, file in experiment_dict.items():
            print(f"    Experiment: {experiment_name}, Model: {file}")
    print(f"\nTotal model files archived: {total_copied}")
    print(f"Total files to archive: {len(files_to_add)}")
    print(f"Total size: {format_file_size(total_size)}")

    # 创建tar.gz压缩包
    if files_to_add:
        print(f"\nCreating {output_file_dir}.tar.gz...")
        with tarfile.open(f"{output_file_dir}.tar.gz", "w:gz") as tar:
            for src_path, arcname in files_to_add:
                tar.add(src_path, arcname=arcname)
        print(f"Successfully created {output_file_dir}.tar.gz with {len(files_to_add)} files.")

        # 显示压缩包大小
        archive_size = get_file_size(f"{output_file_dir}.tar.gz")
        print(f"Archive size: {format_file_size(archive_size)}")
        if total_size > 0:
            compression_ratio = (1 - archive_size / total_size) * 100
            print(f"Compression ratio: {compression_ratio:.1f}%")
    else:
        print("\nNo files to archive.")

    print("Finished script.")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="直接打包模型文件并跳过n天前的文件")
    parser.add_argument("--n_days", type=float, default=0, help="跳过n天前的文件（支持浮点数，如0.1天）。默认0（不跳过）")
    parser.add_argument("--max_id", type=int, default=math.inf, help="过滤模型编号最大值。默认无限制")
    parser.add_argument("--min_id", type=int, default=100, help="过滤模型编号最小值。默认100")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="输出压缩包名称（不含扩展名）。默认'output'",
    )
    parser.add_argument("-y", action="store_true", help="保留参数（原用于自动清空目录，现无作用）")
    parser.add_argument("--copy-event", type=int, default=1, help="是否打包事件文件。默认1（打包）")
    args = parser.parse_args()

    # 调用主函数
    main(
        args.n_days,
        args.max_id,
        args.min_id,
        args.output_dir,
        args.y,
        bool(args.copy_event),
    )
