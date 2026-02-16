import os
from datetime import datetime
import shutil
from pathlib import Path
import time
import json
import requests
from datetime import datetime

def get_basic_info(task_record_path):
        
    with open(task_record_path, 'r', encoding='utf-8') as f:
        task_record = json.load(f)

    task_key = next(iter(task_record))
    task_info = task_record[task_key]
    task_id = task_info["task_id"]
    file_count = task_info["file_count"]
    page_count = task_info["page_count"]
    basic_info = [task_id, file_count, page_count]
    
    return basic_info

def check_file_exists(folder_path, file_name, timeout=300, check_interval=1):
    """
    检查文件夹下是否存在某文件，直到超时。
    
    :param folder_path: 文件夹路径
    :param file_name: 要检查的文件名
    :param timeout: 最大等待时间（秒），默认300秒
    :param check_interval: 检测频率（秒），默认1秒
    :return: 如果文件存在，返回1；如果超时后文件仍然不存在，返回0。
    """
    if file_name in os.listdir(folder_path):
        return 1

    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if file_name in os.listdir(folder_path):
            return 1
        time.sleep(check_interval)
    
    return 0

def send_message_to_phone(msg ,title , PUSHDEER_URL, PUSHDEER_KEY):
        
    try:
        for keys in PUSHDEER_KEY: 
            requests.get(PUSHDEER_URL, params={
                "pushkey": keys,
                "text": title,
                "desp": msg
            })
            print(f"已发送推送消息：{msg}")
    except Exception as e:
        print(f"发送推送失败：{e}")

def copy_folder_to_database(folder_path, base_path):
    """
    将指定文件夹复制到指定的数据库文件夹中，文件夹名称将包含当前时间戳。

    参数:
    folder_path (str): 需要复制的源文件夹路径。
    base_path (str): 存储目标文件夹的根目录路径（例如“database”）。

    返回:
    None
    """
    try:
        # 确保目标数据库文件夹存在
        database_dir = os.path.join(base_path, "database")
        os.makedirs(database_dir, exist_ok=True)

        # 生成带时间戳的目标文件夹名称
        timestamp = datetime.now().strftime("report_%Y_%m_%d_%H_%M_%S")
        folder_name = os.path.basename(os.path.normpath(folder_path))
        dest_folder_name = f"{timestamp}_{folder_name}"
        dest_folder = os.path.join(database_dir, dest_folder_name)

        # 复制文件夹到目标目录
        shutil.copytree(folder_path, dest_folder)

        print(f"已将本进程中产生的所有文件{folder_path}备份")
        print(f"备份到到数据库{database_dir}")
        print(f"具体在数据库中位置为{dest_folder}\n")
    except Exception as e:
        print(f"复制文件夹失败：{e}")

def copy_file(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)  # 不保留文件属性

def merge_txt_files(file1_path, file2_path, output_path):

    try:

        with open(file1_path, 'r', encoding='utf-8') as f1:
            content1 = f1.read()

        with open(file2_path, 'r', encoding='utf-8') as f2:
            content2 = f2.read()

        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(content1)
            out.write('\n')  # 可选：添加换行符分隔两个文件内容
            out.write(content2)
    
    except Exception as e:
        print(f"合并文件时发生错误: {e}")


import json

def show_file(file_path):
    """
    读取文件并输出内容到控制台
    - .json: 解析后压缩成一行输出
    - .jsonl: 一行一行读取，一行一行输出（每行压缩为一行 JSON）
    - .txt: 直接输出内容（去掉换行）
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(json.dumps(data, separators=(',', ':'), ensure_ascii=False))

        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        print(json.dumps(obj, separators=(',', ':'), ensure_ascii=False))
                    except json.JSONDecodeError as e:
                        print(f"第{line_no}行JSON解析失败: {e}")

        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f.read().replace('\n', '').replace('\r', ''))

        else:
            print("不支持的文件类型，仅支持 .json、.jsonl 和 .txt 文件。")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def trim_log_if_too_large(log_path: str, max_bytes: int = 10 * 1024 * 1024):
    """
    若日志文件超过 max_bytes，则删除其前一半内容（尽量从换行处截断，避免截断到半行）。
    采用二进制处理，最后按 utf-8 写回；若截断点落在多字节字符中，会用 'ignore' 跳过残缺字符。
    """
    try:
        if not os.path.exists(log_path):
            return

        size = os.path.getsize(log_path)
        if size <= max_bytes:
            return

        with open(log_path, "rb") as f:
            data = f.read()

        # 取后一半
        start = len(data) // 2

        # 尽量从下一行开始，避免半行开头（找下一个 \n）
        nl = data.find(b"\n", start)
        if nl != -1 and nl + 1 < len(data):
            start = nl + 1

        tail = data[start:]

        # 写回（用 ignore 避免 utf-8 半字符导致异常）
        text = tail.decode("utf-8", errors="ignore")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"[log] mon.log 大小超过 {max_bytes / (1024 * 1024)} MB, 删去前一半.")
    except Exception as e:
        # 不要让日志裁剪影响主流程
        print(f"[log] trim_log_if_too_large 运行失败: {e}")














