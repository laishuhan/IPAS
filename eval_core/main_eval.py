import json
import argparse
# 引入 LEVEL_THRESHOLDS 用于在主流程判断等级
from config_eval import status_map, URGENCY_CONFIG 
from evaluator import PatientEvaluator
from utils_eval import flatten_list
from utils_eval import get_need_info_from_all_info
from utils_eval import calculate_urgency
from utils_eval import extract_global_user_info

from medical_logic import adjust_total_score_global 

parser = argparse.ArgumentParser()
parser.add_argument("--processed_report_info_path", type=str, default="temp.json")
parser.add_argument("--eval_info_path", type=str, default="eval_result.json")
args = parser.parse_args()


if __name__ == '__main__':
    # args = parser.parse_args()
    
    # 请根据实际环境修改路径
    PROCESSED_REPORT_INFO_PATH = args.processed_report_info_path
    EVAL_INFO_PATH = args.eval_info_path
    
    try:
        with open(PROCESSED_REPORT_INFO_PATH, "r", encoding="utf-8") as f:
             data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_REPORT_INFO_PATH}")
        exit(1)

    info_list = data.get("info", [])  # 防止 key 不存在报错
    # 如果 info_list 为空：直接输出一个空白 txt，然后结束程序
    if not info_list:
        try:
            with open(EVAL_INFO_PATH, "w", encoding="utf-8") as f_out:
                f_out.write("")  # 空白文件
            print(f"\n info_list 为空，已输出空白结果文件至: {EVAL_INFO_PATH}")
        except Exception as e:
            print(f"\n info_list 为空，但写入空白文件失败: {e}")
        exit(0)

    all_info = []

    # 1. 提取全局信息 (Index 0)
    user_sex, user_age, user_period_info, user_preg_info = extract_global_user_info(info_list)

    global_info = [
        user_sex,
        user_age,
        user_period_info,
        user_preg_info
    ]
    all_info.append(global_info)

    # 2. 处理子报告信息
    for item in info_list:
        report_type = item["report_type"]
        report_data = item["report_data"]
        abnormal_data_text = flatten_list(item["indicator_analysis"])
        character_data = flatten_list(item["character_analysis"])

        # --- 特殊业务逻辑处理区 ---
        if report_type in (4, 5): 
            if report_data[0] == [-1, -1]: 
                abnormal_data_text[0] = "不存在"
                abnormal_data_text[1] = "不存在" 
            if report_data[1] == 0:
                abnormal_data_text[2] = "不存在"
        else: 
            limit = min(len(report_data), len(abnormal_data_text))
            for i in range(limit):
                val = report_data[i]
                if val == -1:
                    abnormal_data_text[i] = "不存在"
                elif val in ("阴性", "阳性"):
                    abnormal_data_text[i] = val
        
        # Text -> ID
        abnormal_data_ids = [status_map.get(x, -1) for x in abnormal_data_text]

        sub_info = [
            report_type, 
            report_data, 
            abnormal_data_text, 
            abnormal_data_ids, 
            character_data
        ]
        all_info.append(sub_info)
        
    
    # 3. 提取模型所需精简信息
    need_info = get_need_info_from_all_info(all_info)
    
    # 4. 执行评估与统计
    evaluator = PatientEvaluator()
    
    # (1) 计算分数
    eval_result = evaluator.evaluate(need_info)
    
    current_score = eval_result['total_score']
    extra_score = eval_result.get('extra_score', 0)
    rule_hits = eval_result.get('rule_hits', [])
    
    # (2) 计算统计
    stats_result = evaluator.calculate_statistics(need_info)
    abnormal_count = stats_result["abnormal"]
    existing_count = stats_result["existing"]
    abnormal_ratio_percent = stats_result["ratio"] * 100

    # 5. 全局修正
    # 注意: current_score 此时已经包含了 rule 的强制加分
    # 这样年龄系数和占比系数会进一步放大这个加分 (符合高风险人群在严重问题下风险更大的逻辑)
    current_score = adjust_total_score_global(
        sex=user_sex, 
        age=user_age, 
        period_info=user_period_info, 
        preg_info=user_preg_info, 
        abnormal_ratio_percent=abnormal_ratio_percent, 
        total_score=current_score
    )
    
    # 6. 判定等级
    result_config = calculate_urgency(current_score, URGENCY_CONFIG)
    
    # 7. 输出结果
    output_lines = []

    output_lines.append(f"综合异常等级: Level {result_config['level']}/5")
    output_lines.append(f"等级定义: {result_config['desc']}")
    
    output_lines.append(f"异常指标占比: {abnormal_ratio_percent:.1f}%  ({abnormal_count}/{existing_count})")
    
    final_output_text = "\n".join(output_lines)
    
    #  打印到控制台
    # print(final_output_text)
    
    #  输出到文件 (EVAL_INFO_PATH)
    try:
        # 使用 'w' 模式覆盖写入，如果需要追加请改为 'a'
        with open(EVAL_INFO_PATH, "w", encoding="utf-8") as f_out:
            f_out.write(final_output_text)
        print(f"\n 综合评估结果已成功保存至: {EVAL_INFO_PATH}")
    except Exception as e:
        print(f"\n 综合评估结果写入文件失败: {e}")

    

    

