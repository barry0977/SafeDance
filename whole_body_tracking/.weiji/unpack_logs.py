import argparse
import os
import tarfile


def unpack_to_logs(tar_gz_file):
    # 确保 tar.gz 文件存在
    if not os.path.exists(tar_gz_file):
        print(f"Error: {tar_gz_file} 不存在。")
        return

    # 定义目标目录
    target_dir = "logs/rsl_rl"

    # 如果目标目录不存在，则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标目录: {target_dir}")

    # 解压 tar.gz 文件并去掉 output/ 这一层目录
    print(f"解压 {tar_gz_file} 到 {target_dir}，并去掉 output/ 目录层...")
    with tarfile.open(tar_gz_file, "r:gz") as tar:
        # 获取 tar 文件中的所有成员
        members = tar.getmembers()
        for member in members:
            # 去掉成员路径中的 output/ 前缀
            if member.name.startswith("output/"):
                member.name = member.name[len("output/") :]  # 去掉 output/
            else:
                continue  # 如果路径不是 output/ 开头，跳过

            # 解压到目标目录
            if member.isfile():
                print(f"解压文件: {member.name} -> {target_dir}")
            elif member.isdir():
                print(f"创建目录: {member.name} -> {target_dir}")
            tar.extract(member, path=target_dir)

    print("解压完成。")
    print(f"所有文件已解压到 {target_dir} 目录，并去掉了 output/ 目录层。")


if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="解压 tar.gz")
    parser.add_argument("--input", type=str, help="要解压的 tar.gz 文件路径")
    args = parser.parse_args()

    # 调用解压函数
    unpack_to_logs(args.input)
