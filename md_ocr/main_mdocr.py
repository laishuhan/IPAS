import sys
import os
import json
import argparse
import copy
import traceback
from concurrent.futures import ThreadPoolExecutor

# 保持原有模块导入
from module_ocr_data_process import get_user_info, get_basic_info
from module_ocr_data_process import format_d_tree
from module_ocr_data_process import try_convert_unit, find_best_match_key
from module_ocr_data_process import parse_llm_json, build_data_map_for_assemble
from module_ocr_data_process import normalize_val_unit_fields, normalize_llm_output_keys
from module_ocr_data_process import apply_report_value_map  
from module_ocr_data_process import apply_report_value_keep_map
from module_ocr_data_process import clean_indicator_value
from module_ocr_data_process import convert_and_save_target_format
from module_ocr_data_process import get_uncertainty
from module_ocr_data_process import get_diagnosis_text_out
from module_ocr_data_process import get_img_meta

from module_ocr_data_find import FindMethod

from module_gen_conclusion import get_train_required_info_basic

from module_independent import get_basic_info_independent

# 关键词和特殊提取列表
from keywords import SPECIAL_EXTRACTOR_LIST, SPECIAL_EXTRACTOR_MAP
from keywords import FEMALE_ONLY_REPORT, MALE_ONLY_REPORT
from keywords import UNABLE_TO_PROCESS_REPORT_LIST
from keywords import TEMPLATE_KEYWORD_MAP 
from patch import check_template_type 

from report_structure import REPORT_REGISTRY


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llm_polish')))
from llm_keywords import ali_api_text_key_001, ali_api_vision_key_001

parser = argparse.ArgumentParser()
parser.add_argument("--task_record_path", type=str, required=True)
parser.add_argument("--save_result_path", type=str, required=True)
parser.add_argument("--output_train_info_path", type=str, required=True)
parser.add_argument("--output_report_info_path", type=str, required=True)
args = parser.parse_args()

def build_training_input_for_image( reports: list) -> str:
    """
    把一张图的所有 reports 拼成一个可喂给 LLM 的训练 Input（纯文本）。。
    """
    lines = []

    # reports
    for rep in reports:
        r_type = rep.get("report_type", -1)
        report_obj = REPORT_REGISTRY.get(r_type)  # registry 里有 report_name/index :contentReference[oaicite:7]{index=7}
        r_name = report_obj.report_name if report_obj else f"type_{r_type}"

        sex_code = rep.get("sex", -1)
        if sex_code == 0:
            sex_text = "女"
        elif sex_code == 1:
            sex_text = "男"
        else:
            sex_text = "未知"

        age = rep.get("age", -1)
        if age != -1:
            age_text = str(age)
        else:
            age_text = "未知"

        r_time = rep.get("report_time", [-1, -1, -1])

        lines.append("")
        lines.append(f"报告名称: {r_name}")
        lines.append(f"报告主人信息: 性别：{sex_text}, 年龄：{age_text}")
        lines.append(f"报告时间：{r_time}")

        data = rep.get("report_data", [])
        units = rep.get("report_unit", [])
        idx_names = report_obj.index if report_obj else [f"idx_{i}" for i in range(len(data))]

        # 逐指标输出：name: value unit（跳过 -1 / 空）
        for i, v in enumerate(data):
            if v == -1 or v == "" or v is None:
                continue
            ind_name = idx_names[i] if i < len(idx_names) else f"idx_{i}"
            unit = units[i] if i < len(units) else ""
            lines.append(f"{ind_name}= {v} {unit}".strip())

    return "\n".join(lines)

def process_and_save_train_samples(final_report_info_map, save_path):
    """
    处理 final_report_info_map 数据，生成训练样本并保存为 jsonl 格式
    """
    print("开始生成并保存训练样本...")
    train_samples = []

    for img_name, content in final_report_info_map.items():
        # 1. 构建 Input
        reports = content.get("data", [])
        sample_input = build_training_input_for_image(reports)

        # 2. 构建 Output (基于 train_required_info)
        tri = content.get("train_required_info", {})
        info_from = tri.get("info_from", "unknown")

        if info_from == "basic":
            section_found = bool(tri.get("section_found", False))
            diagnosis_text = (tri.get("diagnosis_text") or "").strip()
            
            # 计算不确定性
            uncertainty = get_uncertainty(section_found, diagnosis_text)
            # 格式化输出文本
            diagnosis_text_out = get_diagnosis_text_out(uncertainty, diagnosis_text)

            train_samples.append({
                "info_from": info_from,
                "input": sample_input,
                "output": {
                    "diagnosis_text": diagnosis_text_out,
                    "uncertainty": uncertainty
                }
            })

    # 3. 写入文件
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        print(f"训练样本已保存至: {save_path}")

    except Exception as e:
        print(f"[Error] 保存训练样本失败: {e}")
        

# --- 修改：单图单报告组装逻辑 (不再依赖全局 Pool 和 模板) ---
def assemble_report_for_image(report_type, extracted_data, finder, image_user_info, global_context, template_id=0):
    """
    针对单张图片、单个报告类型进行组装
    """
    task_id = global_context['task_id']
    user_time = global_context['user_time']
    
    # 初始化数据结构
    merged_data = [
        report_type, -1, -1, [-1, -1, -1], user_time,
        global_context['period_info'], global_context['preg_info'],
        "noname",[], [], [], []
    ]

    report_obj = REPORT_REGISTRY.get(report_type)
    report_name = report_obj.report_name
    if report_obj:
        count = len(report_obj.index)
        merged_data[8] = [-1] * count     # report_data
        merged_data[9] = [""] * count     # report_unit
        merged_data[10] = [-1] * count    # report_original_data 
        merged_data[11] = [""] * count    # report_original_unit 

    # 1. 填充性别 (优先报告类型强制指定，其次图片识别)
    if report_type in FEMALE_ONLY_REPORT:
        merged_data[1] = 0
    elif report_type in MALE_ONLY_REPORT:
        merged_data[1] = 1
    elif image_user_info.get("sex", -1) != -1:
        merged_data[1] = image_user_info["sex"]

    # 2. 填充年龄
    if image_user_info.get("age", -1) != -1:
        merged_data[2] = image_user_info["age"]
        
    # 3. 填充检测时间
    if image_user_info.get("time", [-1, -1, -1]) != [-1, -1, -1]:
        merged_data[3] = image_user_info["time"]
        
    # 4. 填充姓名
    if image_user_info.get("name", "noname") != "noname":
        merged_data[7] = image_user_info["name"]

    # 5.指标名列表
    target_name_list = []
    for target_name in report_obj.index:   
        target_name_list.append(target_name)

    # --- 数据组装逻辑 ---
    if report_type not in SPECIAL_EXTRACTOR_LIST:
        print(f"\n====== 组装报告: {report_name} (Type {report_type}) ======")

        for i, target_name in enumerate(report_obj.index):
           
            # 1. 获取基础关键词
            default_aliases = report_obj.keywords[i] if hasattr(report_obj, 'keywords') else [target_name]
            
            # [新增] 2. 获取模板特定关键词
            # 结构: Map[TemplateID][ReportTypeID][IndicatorIndex] -> List[Str]
            template_aliases = []
            if template_id in TEMPLATE_KEYWORD_MAP:
                template_aliases = TEMPLATE_KEYWORD_MAP[template_id].get(report_type, {}).get(i, [])
            
            # [修改] 合并关键词 (模板关键词优先匹配)
            target_aliases = template_aliases + default_aliases

            # 在当前图片的 extracted_data 中查找
            matched_key, matched_alias = find_best_match_key(
                target_aliases,
                extracted_data.keys(),
                report_type,
                i,
                template_id
            )

            hit = True if matched_key else False
            hit_key = matched_alias if matched_key else ""
       
            print(f"{target_name}命中词库中关键词——>{hit_key}" if hit else "未命中词库中关键词")

            if matched_key:

                item = extracted_data[matched_key]
                merged_data[8][i], merged_data[9][i] = item.get('val', -1), item.get('unit', "")
            
    # 特殊报告处理
    elif report_type in SPECIAL_EXTRACTOR_MAP:
        extractor_name = SPECIAL_EXTRACTOR_MAP[report_type]
        print(f"\n====== 组装报告: 特殊报告 (Type {report_type}) ======")
        
        special_result = None
        if finder:
            extractor = getattr(finder, extractor_name, None)
            if extractor:
                try:
                    # 仅针对单张图
                    chosen_key = ali_api_text_key_001 if report_type == 4 else ali_api_vision_key_001
                    special_result = extractor(chosen_key)
                except Exception as e:
                    print(f"[WARN] 特殊提取失败: {e}")

        if special_result is not None:
            # 单图模式下，直接赋值，不进行多页列表嵌套
            merged_data[8] = copy.deepcopy(special_result)
            merged_data[10] = copy.deepcopy(special_result)
            print(f"  - 特殊报告数据: {merged_data[8]}")

    # 备份原始数据
    merged_data[10] = copy.deepcopy(merged_data[8])
    merged_data[11] = copy.deepcopy(merged_data[9])

    #此类别总指标个数
    idx_count = len(merged_data[8])
    print(f"报告类型：{report_name}")
    print(f"指标名称列表：{target_name_list}")
    
    #1.第一次值清洗
    for i in range(idx_count):
        # 值清洗
        if isinstance(merged_data[8][i], str):
            merged_data[8][i], is_text = clean_indicator_value(
                merged_data[8][i],
                report_type=report_type,
                indicator_index=i
            )    
    print(f"P1 第一次值清洗：{merged_data[10]}/{merged_data[11]}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]

    #2.智能二次提取 值/单位
    merged_data[8], merged_data[9] = finder.smart_secondary_extraction(report_type, target_name_list, merged_data[8], merged_data[9], ali_api_text_key_001)
    print(f"P2 智能二次提取：{temp_val}/{temp_unit}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]

    #3.第二次值清洗
    for i in range(idx_count):
        # 再次值清洗
        if isinstance(merged_data[8][i], str):
            merged_data[8][i], is_text = clean_indicator_value(
                merged_data[8][i],
                report_type=report_type,
                indicator_index=i
            )
    print(f"P3 第二次值清洗：{temp_val}/{temp_unit}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]

    #4.单位换算
    for i in range(idx_count):
        merged_data[8][i], merged_data[9][i] = try_convert_unit( merged_data[8][i], merged_data[9][i], report_obj.unit_conversions[i])
    print(f"P4 单位换算：{temp_val}/{temp_unit}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]

    #5.转为决策树所需小数位
    for i in range(idx_count):
        merged_data[8][i] = format_d_tree(merged_data[8][i])
    print(f"P5 转为决策树所需小数位：{temp_val}/{temp_unit}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]

    #6.值保留，仅保留关键字段（“阴性 1.2”->“阴性”）
    merged_data[8] = apply_report_value_keep_map(report_type, merged_data[8])  
    print(f"P6 值保留：{temp_val}/{temp_unit}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]
    
    #7.值映射("未检出"->"阴性")
    merged_data[8] = apply_report_value_map(report_type, merged_data[8])  
    print(f"P7 值映射：{temp_val}/{temp_unit}-->{merged_data[8]}/{merged_data[9]}")
    temp_val = merged_data[8]
    temp_unit =merged_data[9]

    return {
        "task_id": task_id,
        "report_type": merged_data[0],
        "sex": merged_data[1],
        "age": merged_data[2],
        "report_time": merged_data[3],
        "user_time": merged_data[4],
        "period_info": merged_data[5],
        "preg_info": merged_data[6],
        "user_name": merged_data[7],
        "report_data": merged_data[8],
        "report_unit": merged_data[9],
        "report_original_data": merged_data[10],  
        "report_original_unit": merged_data[11]   
    }

def process_single_image_pipeline(item, ocr_text, global_context, is_collect_train_info=False, collect_train_info_from=None):
    
    path = item[0]
    target_report_types = item[1]
    image_name = os.path.basename(path)

    print(f"开始处理图片: {image_name}, 目标报告类型: {target_report_types}")

    template_id = check_template_type(ocr_text)
    if template_id != 0:
        print(f"[{image_name}] 命中特殊模板 ID: {template_id}")

    # 先初始化 Finder（后面的通用提取要用）
    finder = FindMethod(ocr_text, path, target_report_types, template_id)

    # 如果全都是特殊提取报告，则跳过通用提取
    is_all_special = set(target_report_types).issubset(set(SPECIAL_EXTRACTOR_LIST))

    # ---- 并发任务定义：user_info / train_required_info / 通用指标提取 ----
    def _run_get_user_info():
        name, sex, age, time = get_user_info(
            path, ocr_text,
            ali_api_vision_key_001, ali_api_text_key_001,
            vision_or_text="text"
        )
        return {"name": name, "sex": sex, "age": age, "time": time}

    def _run_get_train_required_info():
        # 注意：这里的判断虽然保留，但在外部控制提交后，实际只有 True 时才会进这个函数
        if not is_collect_train_info:
            return {}
        if collect_train_info_from == "basic":
            tri = get_train_required_info_basic(
                path, ocr_text,
                ali_api_vision_key_001, ali_api_text_key_001,
                vision_or_text="text"
            )
            tri["info_from"] = "basic"
            return tri
        return {}

    def _run_extract_general_indicators():
        # all_special 直接跳过
        if is_all_special:
            return {}, {}
        final_response = finder.extract_all_indicators_from_general_image(
            key_vision=ali_api_vision_key_001,
        )
        raw_json = parse_llm_json(final_response)
        llm_output = normalize_val_unit_fields(raw_json)
        llm_output_json = normalize_llm_output_keys(llm_output)

        # 强转 string
        for key, val_item in llm_output_json.items():
            if isinstance(val_item, dict):
                if 'val' in val_item:
                    val_item['val'] = str(val_item['val'])
                if 'unit' in val_item:
                    val_item['unit'] = str(val_item['unit'])

        extracted_data_map = build_data_map_for_assemble(llm_output_json)
        return extracted_data_map, llm_output_json

    # ---- 动态调整并发逻辑 ----
    image_user_info = {"name": "noname", "sex": -1, "age": -1, "time": [-1, -1, -1]}
    train_required_info = {}
    extracted_data_map = {}
    llm_output_json = {}

    # 根据开关决定最大线程数：如果要收集训练信息则为3，否则为2
    worker_count = 3 if is_collect_train_info else 2

    with ThreadPoolExecutor(max_workers=worker_count) as local_ex:
        # 1. 必选任务提交
        fut_user = local_ex.submit(_run_get_user_info)
        fut_extract = local_ex.submit(_run_extract_general_indicators)
        
        # 2. 可选任务提交
        fut_tri = None
        if is_collect_train_info:
            fut_tri = local_ex.submit(_run_get_train_required_info)

        # 3. 获取结果 - User Info
        try:
            image_user_info = fut_user.result()
        except Exception as e:
            print(f"[{image_name}] get_user_info 失败: {e}")

        # 4. 获取结果 - Train Info (仅当任务被提交时)
        if fut_tri:
            try:
                train_required_info = fut_tri.result()
            except Exception as e:
                print(f"[{image_name}] get_train_required_info_basic 失败: {e}")
        else:
            train_required_info = {} # 未开启收集时默认为空

        # 5. 获取结果 - Indicators
        try:
            extracted_data_map, llm_output_json = fut_extract.result()
            if is_all_special:
                print(f"[{image_name}] 仅包含特殊提取报告，跳过通用 LLM 提取。")
            else:
                print(f"[{image_name}] 通用指标提取完成，共 {len(extracted_data_map)} 条。")
        except Exception as e:
            print(f"[{image_name}] 提取失败: {e}")
            extracted_data_map, llm_output_json = {}, {}

    # 4. 组装报告（仅针对该图片要求的 types）
    image_reports = []
    for r_type in target_report_types:
        if r_type not in UNABLE_TO_PROCESS_REPORT_LIST:
            try:
                report_res = assemble_report_for_image(
                    r_type,
                    extracted_data_map,
                    finder,
                    image_user_info,
                    global_context,
                    template_id=template_id
                )
                if report_res:
                    image_reports.append(report_res)
            except Exception as e:
                print(f"[{image_name}] 组装报告 Type {r_type} 失败: {e}")
                print(traceback.format_exc())   # ✅ 加这一行

    # 计算图片大小,宽高
    img_size_str, img_w_h = get_img_meta(path)


    return image_name, image_reports, llm_output_json, img_size_str, img_w_h, train_required_info

if __name__ == '__main__':
    # 信息路径
    TASK_RECORD_INFO_PATH = args.task_record_path
    OCR_RESULT_PATH = args.save_result_path
    TRAIN_REPORT_INFO_PATH = args.output_train_info_path
    REPORT_INFO_PATH = args.output_report_info_path


    IS_INDEPENDENT_ROTATE = True #是否独立旋转
    IS_INDEPENDENT_OCR = True #是否独立OCR


    # IS_INDEPENDENT_ROTATE = False #是否独立旋转
    # IS_INDEPENDENT_OCR = False #是否独立OCR

    if IS_INDEPENDENT_ROTATE == True or IS_INDEPENDENT_OCR == True:
        if IS_INDEPENDENT_ROTATE:
            print("旋转不依赖接口模式\n")
        if IS_INDEPENDENT_OCR:
            print("OCR不依赖接口模式\n")

        # 获取基础信息
        basic_info = get_basic_info_independent(TASK_RECORD_INFO_PATH, OCR_RESULT_PATH, IS_INDEPENDENT_ROTATE, IS_INDEPENDENT_OCR, "DEFAULT", "LLM")
    
    else:
        # 获取基础信息
        print("依赖接口模式！\n")
        basic_info = get_basic_info(TASK_RECORD_INFO_PATH, OCR_RESULT_PATH)

    (
        task_id, 
        paths_all_info_list, 
        u_time, 
        period, 
        preg, 
        all_ocr_text_results,
        total_img_number,
        solvable_img_number, 
        unsolvable_img_number,
        unsolvable_img_unclassified_number,
        unsolvable_img_vig_not_zero_number,
        solvable_img_list,
        unsolvable_img_list,
        unsolvable_img_list_unclassified,
        unsolvable_img_list_vig_not_zero
    ) = basic_info

    # 构建 Global Context
    global_context = {
        "task_id": task_id,
        "user_time": u_time,
        "period_info": period,
        "preg_info": preg,
    }

    # 1. 结果容器 (恢复为原始的嵌套结构)
    final_report_info_map = {} 
    all_imgs_size_list = []
    all_imgs_w_h_list = []

    print(f"开始处理 {len(paths_all_info_list)} 张图片，采用独立管线模式...")

    # 2. 并行处理
    max_workers = 3
    IS_COLLECT_TRAIN_INFO = False  # 是否收集训练所需信息
    COLLECT_TRAIN_INFO_FROM = "basic"  
    

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, item in enumerate(paths_all_info_list):
            if  (not set(item[1]).issubset(set(UNABLE_TO_PROCESS_REPORT_LIST))) and item[3] == 0:
                ocr_res = all_ocr_text_results[i]
                futures.append(executor.submit(process_single_image_pipeline, item, ocr_res, global_context, is_collect_train_info=IS_COLLECT_TRAIN_INFO, collect_train_info_from=COLLECT_TRAIN_INFO_FROM))
            elif set(item[1]).issubset(set(UNABLE_TO_PROCESS_REPORT_LIST)):
                print(f"图片{item[0]}包含的类别均无法处理，跳过此图片。")
                img_size, img_w_h = get_img_meta(item[0])
                all_imgs_size_list.append(img_size)
                all_imgs_w_h_list.append(img_w_h)
            elif item[3] != 0:
                print(f"图片{item[0]}vig不为0，跳过此图片。")
                img_size, img_w_h = get_img_meta(item[0])
                all_imgs_size_list.append(img_size)
                all_imgs_w_h_list.append(img_w_h)

        for future in futures:
            img_name, reports, llm_json, img_size, img_w_h, train_required_info = future.result()
            
            # [原始逻辑] 聚合结果为嵌套字典
            final_report_info_map[img_name] = {
                "data": reports,        # 报告列表
                "number": len(reports), # 报告数量
                "llm_output": llm_json, # 原始 LLM 提取
                "train_required_info": train_required_info # 训练所需信息
            }
            all_imgs_size_list.append(img_size)
            all_imgs_w_h_list.append(img_w_h)

    # 3. 构建原始数据对象,可用于后续模型训练
    original_data_structure = {
        "info": final_report_info_map, 
        "total_img_number": total_img_number,
        "solvable_img_number": solvable_img_number,
        "unsolvable_img_number": unsolvable_img_number,
        "unsolvable_img_unclassified_number": unsolvable_img_unclassified_number,
        "unsolvable_img_vig_not_zero_number": unsolvable_img_vig_not_zero_number,
        "solvable_img_list": solvable_img_list,
        "unsolvable_img_list": unsolvable_img_list,
        "unsolvable_img_list_unclassified": unsolvable_img_list_unclassified,
        "unsolvable_img_list_vig_not_zero": unsolvable_img_list_vig_not_zero,
        "img_size_list": all_imgs_size_list,
        "img_w_h_list": all_imgs_w_h_list 
    }

    if IS_COLLECT_TRAIN_INFO == True:
        # 4.导出并保存训练样本
        process_and_save_train_samples(final_report_info_map, TRAIN_REPORT_INFO_PATH)

    # 5. 调用函数转换并保存为目标格式
    # 直接使用内存中的 original_data_structure，无需再次读取文件
    required_data_structure = convert_and_save_target_format(original_data=original_data_structure)

    # 6. 转换图片-文件格式
    #required_data_structure_with_img_file = add_img_file_relation(required_data_structure)
    required_data_structure_with_img_file = required_data_structure
    
    try:
        with open(REPORT_INFO_PATH, 'w', encoding='utf-8') as f:
            json.dump(required_data_structure_with_img_file, f, ensure_ascii=False, indent=4)
        print(f"目标格式结果已保存至: {REPORT_INFO_PATH}")
    except Exception as e:
        print(f"[Error] 保存目标格式失败: {e}")
        
    print("\n所有流程完成。")