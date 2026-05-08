import os
import json

def create_json_files(base_name: str, save_dir: str,
                      task_record_list: list,
                      report_info_list: list):
    """
    在指定目录生成：
    - base_name_task_record.json
    - base_name_report_info.json
    并写入指定列表为 JSON 内容。
    支持嵌套列表。
    """
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 文件路径
    task_record_path = os.path.join(save_dir, f"{base_name}_task_record.json")
    report_info_path = os.path.join(save_dir, f"{base_name}_report_info.json")

    # 写入 JSON
    # ensure_ascii=False 保证中文正常显示
    # indent=4 保证生成的 JSON 格式美观，带缩进
    with open(task_record_path, "w", encoding="utf-8") as f:
        json.dump(task_record_list, f, ensure_ascii=False, indent=4)

    with open(report_info_path, "w", encoding="utf-8") as f:
        json.dump(report_info_list, f, ensure_ascii=False, indent=4)

    print(f"已创建并写入: {task_record_path}")
    print(f"已创建并写入: {report_info_path}")


# ====== 使用示例 ======
if __name__ == "__main__":
    base_name = "amh_003"             # 图片名称
    save_dir = "./database_test/img_ans"  # 生成文件目录

    # 1. task_record 数据 (通常是一层列表)
    task_list = ["amh"] 

    # 2. report_info 数据 (二层嵌套列表)
    # 这里的结构是 [ [候选答案1], [候选答案2] ]
    # 即使只有一个正确答案，也需要包一层外面的列表，使其变成 [[...]]
    report_list = [
        # 第一组正确答案候选
        [5.04]
    
        
        # 如果有第二组允许的正确答案（例如OCR容错），可以加在这里：
        # , [3.1, 30, 7.2, -1, 57.8, 179.0, -1, 56, -1, -1, -1] 
    ]

    create_json_files(base_name, save_dir, task_list, report_list)
