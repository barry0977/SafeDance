#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse

# 配置
USER = "xieweiji180"
PASSWORD = "aef417d0b26566c15598c4237cc00e64"
BASE_URL = "https://gz01-srdart.srdcloud.cn/generic/p24hqasyf0004/p24hqasyf0004-embodiedai-release-generic-local//wjx"


def execute_command(cmd, description):
    """执行命令并处理结果 - 实时输出子进程内容"""
    print(f"\n--- {description} ---")
    print(f"Executing: {cmd}")
    print("-" * 50)

    try:
        # 使用 Popen 实现实时输出
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # 将 stderr 重定向到 stdout
            text=True, 
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )
        
        # 实时读取并输出子进程的输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.rstrip())  # 实时输出，去掉末尾换行符
        
        # 等待进程完成
        return_code = process.wait()
        
        if return_code == 0:
            print("✅ Success!")
            return True
        else:
            print(f"❌ Error! Return code: {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False


def generate_commands(rel_path, delete=False):
    """根据路径字符串生成上传和下载命令，统一使用zip格式"""
    # 把相对路径里的 "/" 转成 "_"，保证唯一
    safe_name = rel_path.replace("/", "_")
    zip_name = f"{safe_name}.zip"

    # 统一使用zip格式：先打包，再上传/下载，最后解压
    upload_cmd = f'zip -r {zip_name} {rel_path} && curl -u {USER}:{PASSWORD} -T ./{zip_name} "{BASE_URL}/{zip_name}"'
    download_cmd = f'wget -O {zip_name} --user={USER} --password={PASSWORD} "{BASE_URL}/{zip_name}" --no-check-certificate && unzip -o {zip_name}'

    delete_cmd = f"rm {zip_name}"
    if delete:
        upload_cmd += f" && {delete_cmd}"
        download_cmd += f" && {delete_cmd}"

    return upload_cmd, download_cmd


def main():
    parser = argparse.ArgumentParser(description="Sync files to/from remote server")
    parser.add_argument("relative_path", help="Relative path to file or directory")
    parser.add_argument("-U", "--upload", action="store_true", help="Execute upload command directly")
    parser.add_argument("-D", "--download", action="store_true", help="Execute download command directly")
    parser.add_argument("-d", "--delete", action="store_true", help="Delete the generated zip file after operation")

    args = parser.parse_args()

    rel_path = args.relative_path.rstrip("/")  # 保留用户传入的相对路径

    # 生成命令（不检查本地文件是否存在）
    upload_cmd, download_cmd = generate_commands(rel_path, args.delete)

    # 根据参数决定执行操作
    if args.upload:
        success = execute_command(upload_cmd, "Upload Command")
        if not success:
            sys.exit(1)
    elif args.download:
        success = execute_command(download_cmd, "Download Command")
        if not success:
            sys.exit(1)
    else:
        # 默认行为：只显示命令
        print("\n--- Upload Command ---")
        print(upload_cmd)
        print("\n--- Download Command ---")
        print(download_cmd)


if __name__ == "__main__":
    main()
