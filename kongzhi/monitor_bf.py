import time
import os
import sys
import json
from multiprocessing import Process
from datetime import datetime
from zoneinfo import ZoneInfo
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from path import base_path
from path import watch_paths
from kongzhi_keywords import PUSHDEER_URL
from kongzhi_keywords import PUSHDEER_KEY
from monitor_bf_subprocess import run_main_mdocr
from monitor_bf_subprocess import run_d_tree
from monitor_bf_subprocess import run_md_llm_api
from monitor_bf_subprocess import run_eval_core
from kongzhi_utils import check_file_exists
from kongzhi_utils import show_file
from kongzhi_utils import get_basic_info
from kongzhi_utils import copy_file
from kongzhi_utils import merge_txt_files
from kongzhi_utils import trim_log_if_too_large

# ========== 新增：将所有终端输出追加到 logs.txt ==========
class Tee:
    """同时输出到终端和文件"""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


# 自定义事件处理器：监控 sandbox 目录中新建的子文件夹
class FolderCreationHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            new_folder = event.src_path
            print(f"检测到新建文件夹: {new_folder}，开启新进程监控...")
            process = Process(target=main_kongzhi, args=(new_folder,))
            process.start()
            print(f"已为新建文件夹 {new_folder} 启动处理进程")


def check_point(folder_path, file_name, start_time, timeout=3600, is_show=0):
    if is_show == -1:
        return False

    file_path = os.path.join(folder_path, file_name)

    is_file_exist = check_file_exists(folder_path, file_name, timeout)

    if is_file_exist == 1:
        now_time = time.time()
        print(f"检测到{file_name},总用时{round(now_time - start_time, 1)}秒\n")
        if is_show == 1:
            print(f"{file_name}内容为:")
            show_file(file_path)
            print("\n")
        return True
    else:
        # copy_folder_to_database(folder_path, base_path)
        return False


def main_kongzhi(folder_path):

    #=============== 其它文件目录====================
    temp_path = os.path.join(base_path, "kongzhi/test/database_test/temp_file") #测试用路径
    #port_path = folder_path.replace("sandbox2", "sandbox") #接口用路径
    #===============================================

    start_time = time.time()

    if not check_point(folder_path, "task_record.json", start_time, is_show=1):
        print("未生成task_record.json-接口")
        return  # 文件不存在，终止处理流程

    task_record_path = os.path.join(folder_path, "task_record.json")
    basic_info = get_basic_info(task_record_path)
    temp_report_info_path = os.path.join(folder_path, "temp_report_info.json")
    print(f"开始处理新请求{basic_info[0]}, 总文件数:{basic_info[1]}, 总图片数:{basic_info[2]}")
    
    time.sleep(0.2) #等待文件稳定
    #提取程序
    run_main_mdocr(folder_path)

    if not check_point(folder_path, "temp_report_info.json", start_time, is_show=1):
        print("未生成 temp_report_info.json-数据提取")
        return  # 文件不存在，终止处理流程
    
    if not check_point(folder_path, "train_samples.jsonl", start_time, timeout = 0, is_show=1):
        print("微调数据提取模式关闭，未生成 train_samples.jsonl-数据提取")
    
    else:
        # ====== 追加汇总 train_samples.jsonl ======
        try:
            src_train_samples = os.path.join(folder_path, "train_samples.jsonl")
            dst_dir = os.path.join(base_path, "logs")
            dst_train_samples = os.path.join(dst_dir, "train_samples.jsonl")

            # 确保目标目录存在
            os.makedirs(dst_dir, exist_ok=True)

            # 逐行追加（jsonl 正确姿势）
            with (
                open(src_train_samples, "r", encoding="utf-8") as src_f,
                open(dst_train_samples, "a", encoding="utf-8") as dst_f
            ):
                for line in src_f:
                    line = line.strip()
                    if line:  # 跳过空行
                        dst_f.write(line + "\n")

            print(f"已将 {src_train_samples} 追加写入 {dst_train_samples}")

        except Exception as e:
            print(f"追加 train_samples.jsonl 失败: {e}")

    # ==================== yolo_result 合并逻辑 ====================
    yolo_result_path = os.path.join(folder_path, "yolo_result")
    # 1. 监测 folder_path 中是否有 yolo_result 文件夹
    if os.path.exists(yolo_result_path) and os.path.isdir(yolo_result_path):
        end_file_path = os.path.join(yolo_result_path, "end.txt")
        max_wait = 15
        if not check_point(yolo_result_path, "end.txt", start_time, timeout=max_wait):
            print(f"等待超时 ({max_wait}s)，未发现 'end.txt' 文件。默认泰国b超部分提取完毕,准备开始合并...")


        print(f"开始扫描 {yolo_result_path} 下的子文件夹内容并合并...")
    
        # --- A. 读取主 temp_report_info.json ---
        # 此时如果是泰国超声，这里读取到的就是刚才生成的空Json
        main_json_data = {}
        try:
            with open(temp_report_info_path, 'r', encoding='utf-8') as f:
                main_json_data = json.load(f)
        except Exception as e:
            print(f"读取主 temp_report_info.json 失败: {e}")

        
        sub_items = sorted(os.listdir(yolo_result_path))                
        # 3. 依次扫描子文件夹，合并数据
        for item in sub_items:
            sub_folder = os.path.join(yolo_result_path, item)
            if os.path.isdir(sub_folder):
                sub_json_path = os.path.join(sub_folder, "report_info.json")
                if os.path.exists(sub_json_path):
                    try:
                        with open(sub_json_path, 'r', encoding='utf-8') as f:
                            sub_data = json.load(f)
                            
                            # --- B. 合并逻辑 ---
                            if "info" in sub_data and isinstance(sub_data["info"], list):
                                if "info" not in main_json_data:
                                    main_json_data["info"] = []
                                
                                main_json_data["info"].extend(sub_data["info"])
                                
                                print(f"已合并: {item}/report_info.json")
                                
                    except Exception as e:
                        print(f"处理 {item} 下的 json 失败: {e}")
            
            # 4.report_info.json数据合并更新
            try:
                with open(temp_report_info_path, 'w', encoding='utf-8') as f:
                    json.dump(main_json_data, f, ensure_ascii=False, indent=4)
                print("主 temp_report_info.json 数据合并更新完成。")
            except Exception as e:
                print(f"保存更新后的 temp_report_info.json 失败: {e}")
    
    # 生成最终的report_info.json
    repot_info_name = "report_info.json"
    copy_file(temp_report_info_path, os.path.join(folder_path, repot_info_name))

    if not check_point(folder_path, "report_info.json", start_time, is_show=1):
        print("未生成 report_info.json-数据提取")
        return  # 文件不存在，终止处理流程

    time.sleep(0.2) #等待文件稳定
    # 决策树
    run_d_tree(folder_path)  # 新决策树
    if not check_point(folder_path, "report_abnormal.txt", start_time):
        print("未生成report_abnormal.txt-决策树")
        return  # 文件不存在，终止处理流程
    
    if not check_point(folder_path, "report_jianyi.txt", start_time):
        print("未生成report_jianyi.txt-决策树")
        return  # 文件不存在，终止处理流程
    
    if not check_point(folder_path, "processed_report_info.json", start_time, is_show=1):
        print("processed_report_info.json-决策树")
        return  # 文件不存在，终止处理流程
    
    time.sleep(0.2) #等待文件稳定
    # 综合分析
    run_eval_core(folder_path)

    # 大语言模型润色 api
    model_type = 2
    polish_type = 0 #默认润色线
    run_md_llm_api(folder_path, polish_type, model_type)

    if not check_point(folder_path, "eval_info.txt", start_time, is_show=1):
        print("未生成eval_info.txt", "后端错误报告-综合评估！")
        return  # 文件不存在，终止处理流程

    if not check_point(folder_path, "temp_report_polished_merge.txt", start_time, is_show=1):
        print("未生成temp_report_polished_merge.txt", "后端错误报告-大模型润色！")
        copy_file(os.path.join(folder_path, "report_jianyi.txt"),os.path.join(folder_path, "report_polished_jianyi.txt"))
        merge_txt_files(os.path.join(folder_path, "report_abnormal.txt"),os.path.join(folder_path, "report_polished_jianyi.txt"),os.path.join(folder_path, "temp_report_polished_merge.txt"))
        # 无法正常润色时，直接使用初版建议输出
    
    merge_txt_files(os.path.join(folder_path, "eval_info.txt"),os.path.join(folder_path, "temp_report_polished_merge.txt"),os.path.join(folder_path, "report_polished_merge.txt"))

    if not check_point(folder_path, "report_polished_merge.txt", start_time, is_show=1):
        print("未生成report_polished_merge.txt", "后端错误报告-综合评估与润色结果融合！")
        return

    # 确保当前进程的日志已落盘（可选，但推荐）
    try:
        log_file.flush()
    except Exception:
        pass
    # 裁剪日志文件，防止过大
    MAX_LOG_SIZE = 10  # 最大日志尺寸10 MB
    trim_log_if_too_large(os.path.join(base_path, "logs/mon.log"), max_bytes= MAX_LOG_SIZE * 1024 * 1024)

    print(f"进程正常结束，已销毁")
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    print(now.strftime("%Y-%m-%d %H:%M:%S")) 



# ================= 主程序入口（必须加这个） =================
if __name__ == "__main__":

    # Windows 必须加这个
    from multiprocessing import freeze_support
    freeze_support()

    log_dir = os.path.join(base_path, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, "mon.log")
    log_file = open(log_path, "a", encoding="utf-8")

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    observers = []

    for path in watch_paths:
        if os.path.exists(path) and os.path.isdir(path):
            handler = FolderCreationHandler()
            observer = Observer()
            observer.schedule(handler, path=path, recursive=False)
            observer.start()
            observers.append(observer)
            print(f"正在监控文件夹：{path}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("停止监控...")

        for obs in observers:
            obs.stop()

    for obs in observers:
        obs.join()

    print("程序退出")




