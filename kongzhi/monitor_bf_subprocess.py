import os
import subprocess
from path import md_ocr_path
from path import d_tree_path
from path import eval_core_path
from path import md_llm_api_path

# 运行 main_mdocr.py 脚本并传入参数
def run_main_mdocr(folder_path):
    script_name = "main_mdocr.py"
    task_record_path = os.path.join(folder_path, "task_record.json")
    save_result_path = os.path.join(folder_path, "ocr_result.json")
    output_train_info_path = os.path.join(folder_path, "train_samples.jsonl")
    output_report_info_path = os.path.join(folder_path, "temp_report_info.json")

    try:
        print(f"启动 {script_name} 处理：{folder_path}")
        result = subprocess.run([
            "python", md_ocr_path,
            "--task_record_path", task_record_path,
            "--save_result_path", save_result_path,
            "--output_train_info_path", output_train_info_path,
            "--output_report_info_path", output_report_info_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f" {script_name} 处理完成：{folder_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 出错：{e}")
        print(f"[STDERR]:\n{e.stderr}")

# 运行新决策树脚本并传入参数
def run_d_tree(folder_path):
    script_name = "advice_generate.py"
    input_path = os.path.join(folder_path, "report_info.json")
    abnormal_path = os.path.join(folder_path, "report_abnormal.txt")
    indicator_analysis_path = os.path.join(folder_path, "indicator_analysis_text.txt")
    character_analysis_path = os.path.join(folder_path, "character_analysis_text.txt")
    output_jianyi_path = os.path.join(folder_path, "report_jianyi.txt")

    try:
        print(f"启动 {script_name} 处理：{folder_path}")
        result = subprocess.run([
            "python", d_tree_path,
            "--input_path", input_path,
            "--abnormal_path", abnormal_path,
            "--indicator_analysis_path", indicator_analysis_path,
            "--character_analysis_path", character_analysis_path,
            "--output_jianyi_path", output_jianyi_path,
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f" {script_name} 处理完成：{folder_path}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 出错：{e}")
        print(f"[STDERR]:\n{e.stderr}")

# 运行 main_eval.py 脚本并传入参数
def run_eval_core(folder_path):
    script_name = "main_eval.py"
    processed_report_info_path = os.path.join(folder_path, "processed_report_info.json")
    eval_info_path = os.path.join(folder_path, "eval_info.txt")

    try:
        print(f"启动 {script_name} 处理：{folder_path}")
        result = subprocess.run([
            "python", eval_core_path,
            "--processed_report_info_path", processed_report_info_path,
            "--eval_info_path", eval_info_path,
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f"{script_name} 处理完成：{folder_path}")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 出错：{e}")
        print(f"[STDERR]:\n{e.stderr}")

# 运行 md_llm_api.py 脚本并传入参数
def run_md_llm_api(folder_path, polish_type, model_type):
    script_name = "md_llm_api.py"
    report_info_path = os.path.join(folder_path, "report_info.json")
    input_file_path_1 = os.path.join(folder_path, "report_jianyi.txt")
    input_file_path_2 = os.path.join(folder_path, "report_abnormal.txt")
    output_file_path_1 = os.path.join(folder_path, "report_polished_jianyi.txt")
    output_file_path_2 = os.path.join(folder_path, "temp_report_polished_merge.txt")

    try:
        print(f"启动 {script_name} 处理：{folder_path}")
        result = subprocess.run([
            "python", md_llm_api_path,
            "--polish_type", str(polish_type),
            "--model_type", str(model_type),
            "--report_info_path", report_info_path,
            "--input_file_path_1", input_file_path_1,
            "--input_file_path_2", input_file_path_2,
            "--output_file_path_1", output_file_path_1,
            "--output_file_path_2", output_file_path_2
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print(f"{script_name} 处理完成：{folder_path}")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 出错：{e}")
        print(f"[STDERR]:\n{e.stderr}")



