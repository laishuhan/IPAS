from datetime import date
from datetime import datetime
import re
import sys
import os
import json
import copy
from PIL import Image
import ast  # 用于将字符串安全地转换为列表
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from keywords import REPORT_TYPE_LIST
from keywords import KEY_NORMALIZE_CONFIG 
from report_value_map import REPORT_VALUE_KEEP_MAP
from keywords import UNCLASSIFIED_REPORT_ID
from keywords import UNCERTAINTY_MID

from report_value_map import REPORT_VALUE_FILTER_MAP 
from report_value_map import REPORT_VALUE_MAP


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'llm_polish')))
from llm_api import ali_api_vision, ali_api_text
from llm_prompt import DEFAULT_EXTRATCT_SYSTEM_PROMPT

def get_uploade_time(date_str):  # 提交时间
    if date_str == '0000-00-00':
        return -1, -1, -1  # 或者返回你希望的默认值
    elif date_str == '0000-00-01':
        return -2, -2, -2
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.year, date_obj.month, date_obj.day
    except ValueError:
        return -3, -3, -3


def get_user_info(img_path, ocr_text, key_vison, key_text, vision_model=0, text_model=2, vision_or_text="vision"):

    """
    合并提取：姓名、性别、年龄、检查日期
    返回格式: [姓名(str), 性别(int), 年龄(int), [年, 月, 日](list)]
    性别：女=0，男=1，未知=-1
    """

    if vision_or_text == "vision":

        prompt = (
            "以下是一份医疗报告的图片，请提取以下四项基本信息：\n"
            "1. 被检查者姓名（字符串）。请严格区分“患者/被检查者”与“医师/检查者”。"
            "如果报告中出现的姓名仅为“报告医师”、“审核医师”或“检查医生”，"
            "**切勿**将其作为结果，此时应返回 'noname'。若确实不存在患者姓名，返回 'noname'。\n"
            "2. 性别（整数）：女返回 0，男返回 1，未知或不存在返回 -1。\n"
            "3. 年龄（整数），若不存在则返回 -1。\n"
            "4. 报告日期（包含[年, 月, 日]三个整数的列表），若不存在则返回 [-1, -1, -1]。\n\n"
            "请严格按照以下 Python 列表格式返回结果，不要包含 Markdown 标记或其他多余文字：\n"
            "['姓名', 性别, 年龄, [年, 月, 日]]\n"
            "示例：['张三', 1, 32, [2023, 5, 20]]"
        )

        # 调用视觉模型
        content_text = ali_api_vision(DEFAULT_EXTRATCT_SYSTEM_PROMPT, prompt, img_path, vision_model, key_vison, temperature = 0.0)
    
    else :
        prompt = (
            "以下是一份医疗报告的文本内容，请提取以下四项基本信息：\n"
            "1. 被检查者姓名（字符串）。请严格区分“患者/被检查者”与“医师/检查者”。"
            "如果报告中出现的姓名仅为“报告医师”、“审核医师”或“检查医生”，"
            "**切勿**将其作为结果，此时应返回 'noname'。若确实不存在患者姓名，返回 'noname'。\n"
            "2. 性别（整数）：女返回 0，男返回 1，未知或不存在返回 -1。\n"
            "3. 年龄（整数），若不存在则返回 -1。\n"
            "4. 报告日期（包含[年, 月, 日]三个整数的列表），若不存在则返回 [-1, -1, -1]。\n\n"
            "请严格按照以下 Python 列表格式返回结果，不要包含 Markdown 标记或其他多余文字：\n"
            "['姓名', 性别, 年龄, [年, 月, 日]]\n"
            "示例：['张三', 1, 32, [2023, 5, 20]]"
        )


        # 调用文本模型
        content_text = ali_api_text(DEFAULT_EXTRATCT_SYSTEM_PROMPT, prompt, ocr_text, text_model, key_text, temperature=0.0)

    # 清洗返回内容
    cleaned_text = (
        content_text.strip()
        .replace("```python", "")
        .replace("```json", "")
        .replace("```", "")
    )

    try:
        info_list = ast.literal_eval(cleaned_text)
    except Exception as e:
        print("解析失败：", e)
        return ['noname', -1, -1, [-1, -1, -1]]

    # 校验格式
    if (
        isinstance(info_list, list)
        and len(info_list) == 4
        and isinstance(info_list[3], list)
        and len(info_list[3]) == 3
    ):
        return info_list
    else:
        print("解析格式不符，返回默认错误值")
        return ['noname', -1, -1, [-1, -1, -1]]


def get_uncertainty(section_found: bool, diagnosis_text: str) -> str:
    """
    uncertainty 三档规则（可解释、可复现）：
    - high: 没抽到诊断栏目 / 诊断为空 => 信息不足（应拒答）
    - mid : 抽到了诊断，但诊断表述包含不确定/建议进一步检查等措辞
    - low : 抽到了明确诊断且无明显不确定措辞
    """
    text = (diagnosis_text or "").strip()

    if (not section_found) or (not text):
        return "high"

    # mid: 诊断文本含不确定提示词（用正则更稳）    
    pattern = r"|".join(map(re.escape, UNCERTAINTY_MID))
    if re.search(pattern, text):
        return "mid"

    return "low"

def get_diagnosis_text_out(uncertainty, diagnosis_text):

    if uncertainty == "high":
        diagnosis_text_out = "依据当前检验结果无法给出明确临床诊断，建议复核或补充检查。"
    else:
        diagnosis_text_out = diagnosis_text

    return diagnosis_text_out

def get_ocr_text_from_paddleocr2(result):
    """
    对paddleOCR2.x结果进行文本提取。
    """
    if not result or not result[0]:
        return ""
    
    # 获取该图片中所有的文本内容
    all_text = [line[1][0] for line in result[0]]
    combined_text = "".join(all_text)
    
    return combined_text

def get_ocr_text_from_paddleocr3(result):
    """
    对paddleOCR3.x结果进行文本提取。
    """
    if not result:
        return ""
    
    combined_text = "".join(result.get("rec_texts", []))
       
    return combined_text

#获取图片全信息方法
def get_paths_all_info(pages):
    """
    获取图片全信息列表。

    参数:
        pages (list)

    返回:
        list - [ 
                  [ "图片路径A", ["类型A", "类型B"] , "国家A", vig ] ,
                  [ "图片路径B", ["类型A", "类型C"] , "国家B", vig ] ,
               ]
    """
    result = []

    for page in pages:
        image_path = page.get("imagePath")
        categories = page.get("matchedCategories", [])
        country = page.get("country", "")
        vig = page.get("vig", 0)

        # 只在 image_path 存在时加入
        if image_path:
            result.append([image_path, categories, country, vig])
    return result

def sort_paths_list_by_image_ocr_order(paths_list, image_ocr_order):
    # 按照 image_ocr_order 对 paths_list 进行重排序
    if image_ocr_order:
        # 建立 {图片路径: 索引} 的映射，将查找复杂度从 O(N) 降为 O(1)
        order_map = {path: index for index, path in enumerate(image_ocr_order)}
        
        # 对 paths_list 进行排序
        # x[0] 是 paths_list 中的图片路径
        # order_map.get(..., float('inf')) 确保如果有图片不在 order 列表中，会被排到最后，而不是报错
        paths_list.sort(key=lambda x: order_map.get(x[0], float('inf')))
        print(f"依据image_ocr_order重排序的paths_list:{paths_list}")
        return paths_list
    else:
        return paths_list


def get_basic_info(task_record_path, ocr_result_path):
    """
    从 task_record.json 和ocr_result.json中读取所需的基本信息,再进行后处理得到所需样式的信息
    """
    try:

        # task_record.json 部分——读取##################################################################################
        with open(task_record_path, 'r', encoding='utf-8') as f:
            task_record = json.load(f)

        task_key = next(iter(task_record))
        task_info = task_record[task_key]

        task_id = task_info["task_id"]
        file_count = task_info["file_count"]
        page_count = task_info["page_count"]
        create_time = task_info["create_time"]
        callback_url = task_info["callback_url"]
        pages = task_info["pages"]

        ###############################################################################################################

        # ocr_result.json 部分——读取####################################################################################
        with open(ocr_result_path, 'r', encoding='utf-8') as f:
            ocr_result = json.load(f)
        
        #提取文字内容部分
        all_ocr_text_results = []
        for ocr in ocr_result:
            all_ocr_text_results.append(get_ocr_text_from_paddleocr3(ocr))

        #获得image_ocr_order
        # image_ocr_order = []
        # for ocr in ocr_result:
        #     if "input_path" in ocr:
        #         image_ocr_order.append(ocr["input_path"])
        
        # print(f"image_ocr_order:{image_ocr_order}\n")

        ###############################################################################################################


        # 后处理 #######################################################################################################

        # 处理时间
        year, month, day = get_uploade_time(str(create_time))

        # 1. 获取原始列表
        raw_paths_all_info_list = get_paths_all_info(pages)
        print(f"初始的raw_paths_all_info_list:{raw_paths_all_info_list}\n")

        # 2. 报告类别名称 -> 数字 ID
        type_map = {item[0].lower(): item[1] for item in REPORT_TYPE_LIST}

        paths_all_info_list = []
        for path, category_names, country, vig in raw_paths_all_info_list:
            category_ids = []
            for name in category_names:
                name_lower = name.lower()

                if name_lower in type_map:
                    category_ids.append(type_map[name_lower])
                else:
                    print(f"[Warn] 未知报告类型: {name} (在图片 {os.path.basename(path)} 中)")
                    print("此种情况默认归为无法分类 UNCLASSIFIED_REPORT_ID")
                    category_ids.append(UNCLASSIFIED_REPORT_ID)

            paths_all_info_list.append([path, category_ids, country, vig])
        
        # 3. 去重类别 ID 列表中的重复项
        for item in paths_all_info_list:
            item[1] = list(set(item[1]))

        print(f"类别名映射为数字的paths_all_info_list:{paths_all_info_list}\n")      

        total_img_list = [item[0] for item in paths_all_info_list]

        solvable_img_list = [
            item[0]
            for item in paths_all_info_list
            if UNCLASSIFIED_REPORT_ID not in item[1] and item[3] == 0 
        ]

        unsolvable_img_list = list(set(total_img_list) - set(solvable_img_list))

        unsolvable_img_list_unclassified = [
            item[0]
            for item in paths_all_info_list
            if UNCLASSIFIED_REPORT_ID  in item[1]
        ]
        unsolvable_img_list_vig_not_zero = [
            [item[0],item[3]]
            for item in paths_all_info_list
            if item[3] != 0
        ]


        print(f"可处理的图片列表为:{solvable_img_list}")
        print(f"不可处理的图片列表为:{unsolvable_img_list}")
        print(f"由于无法分类不可处理的图片列表为:{unsolvable_img_list_unclassified}")
        print(f"由于vig不为0不可处理的图片列表为:{unsolvable_img_list_vig_not_zero}")

        total_file_number = file_count
        total_img_number = page_count
        solvable_img_number = len(solvable_img_list)
        unsolvable_img_number = len(unsolvable_img_list)
        unsolvable_img_unclassified_number = len(unsolvable_img_list_unclassified)
        unsolvable_img_vig_not_zero_number = len(unsolvable_img_list_vig_not_zero)

        print(f"总图片数: {total_img_number}")
        print(f"可处理的图片数: {solvable_img_number}")
        print(f"不可处理的图片数: {unsolvable_img_number}")
        print(f"由于无法分类不可处理的图片数: {unsolvable_img_unclassified_number}")
        print(f"由于vig不为0不可处理的图片数: {unsolvable_img_vig_not_zero_number}")


        #尝试对图片按照ocr顺序排序
        #paths_all_info_list = sort_paths_list_by_image_ocr_order(paths_all_info_list, image_ocr_order)
        #print(f"按照image_ocr_order排序后得到的paths_all_info_list:{paths_all_info_list}\n")     

        ###############################################################################################################

        result = [
            task_id,
            paths_all_info_list,
            (year, month, day),
            2,  #经期占位
            0,  #孕期占位
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
        ]
        return result

    except Exception as e:
        print(f"[错误] 读取基本信息失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def format_d_tree(x):
    """
    - 如果输入为字符串：原封不动返回（字符串）
    - 有小数点：
        * 小数位 > 2：
            - 若小数点前不是 0：四舍五入到 2 位小数（float）
            - 若小数点前是 0：保留 2 位有效数字（float）
        * 小数位 <= 2：原值转 float 返回
    - 无小数点：返回 int
    """
    # 字符串直接返回
    if isinstance(x, str):
        return x

    s = str(x).strip()

    # 非数字 → 原样返回
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError, TypeError):
        return x

    if "." in s:
        int_part, frac_part = s.split(".", 1)

        if len(frac_part) > 2:
            # 小数点前不是 0 → 保留 2 位小数
            if int_part not in ("0", "-0"):
                return round(float(d), 2)

            # 小数点前是 0 → 保留 2 位有效数字
            else:
                # 计算需要的指数位
                exp = d.adjusted()  # 有效数字的数量级
                quant = Decimal(f'1e{exp - 1}')
                return float(d.quantize(quant, rounding=ROUND_HALF_UP))

        # 小数位 ≤ 2
        return float(d)

    else:
        # 无小数点 → 返回 int
        return int(d)

def try_convert_unit(val, unit_str, unit_conver_config):
    """
    尝试对数值和单位进行换算。
    :param val: 原始数值 (float or int or -1)
    :param unit_str: 原始单位字符串
    :param unit_conver_config: report_structure中的配置，例如 [["ng/mL"], ["nmol"], [0.314]]
    :return: (new_val, new_unit)
    """
    if val == -1:
        return val, unit_str

    # 如果配置为空或不完整，不换算
    if not unit_conver_config or len(unit_conver_config) <= 1:
        return val, unit_str

    required_unit_list = unit_conver_config[0]
    possible_units = unit_conver_config[1]
    coefficients = unit_conver_config[2]

    # 默认要求的单位是列表第一个
    target_unit = required_unit_list[0] if required_unit_list else unit_str

    for p_unit, coeff in zip(possible_units, coefficients):
        if p_unit in unit_str:
            try:
                new_val = val * coeff
                return new_val, target_unit
            except Exception:
                return val, unit_str

    return val, unit_str

def find_best_match_key(
    alias_list,
    candidate_keys,
    report_type,
    idx_num,
    template_id,
    case_insensitive=True
):
    """
    返回 (matched_key, matched_alias)
    - matched_key: candidate_keys 里真正存在的 key（用于取值）
    - matched_alias: alias_list 里命中的那个词（用于打印“命中-xxx”）
    """
    if not alias_list or not candidate_keys:
        return None, None

    config = KEY_NORMALIZE_CONFIG
    remove_chars = config.get("remove_chars", [])

    def normalize(s: str) -> str:

        """根据参数统一做字符串归一化"""
        for remove_char in remove_chars:
            s = s.replace(remove_char, "")

        if case_insensitive:
            s = s.lower()

        #  按 report_type / idx_num / template_id追加规则（示例）
        if report_type == 0:
            pass
        if idx_num == 0:
            pass
        if template_id == 1:
            pass

        return s

    # 构建 normalized_key -> original_key 的映射
    normalized_key_map = {
        normalize(key): key
        for key in candidate_keys
    }

    for alias in alias_list:
        normalized_alias = normalize(alias)
        if normalized_alias in normalized_key_map:
            return normalized_key_map[normalized_alias], alias

    return None, None

def parse_llm_json(final_response):
    """
    从 LLM 输出文本中提取最后一个 JSON dict。
    失败返回 None。
    """
    if not final_response or not isinstance(final_response, str):
        return None

    text = final_response.strip()

    # 找到最后一个 '}'
    end = text.rfind('}')
    if end == -1:
        return None

    # 从后往前找与之匹配的 '{'
    brace_count = 0
    start = -1
    for i in range(end, -1, -1):
        if text[i] == '}':
            brace_count += 1
        elif text[i] == '{':
            brace_count -= 1
            if brace_count == 0:
                start = i
                break

    if start == -1:
        return None

    json_str = text[start:end + 1]

    try:
        return json.loads(json_str)
    except Exception:
        return None

def normalize_val_unit_fields(raw_json):
    if not isinstance(raw_json, dict):
        return {}

    out = {}

    for indicator_name, item in raw_json.items():

        # 标量：直接当 val
        if not isinstance(item, dict):
            out[indicator_name] = {
                "val": item,
                "unit": ""
            }
            continue

        values = list(item.values())

        out[indicator_name] = {
            "val": values[0] if len(values) >= 1 else -1,
            "unit": values[1] if len(values) >= 2 else ""
        }

    return out

def normalize_scientific_text(s: str) -> str:
    """
    将医学报告中常见的科学计数法变体统一为标准格式
    例如：
      57.8 ×10⁶/ml → 57.8x10^6/ml
    """
    if not isinstance(s, str):
        return s

    sup_map = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3",
        "⁴": "4", "⁵": "5", "⁶": "6",
        "⁷": "7", "⁸": "8", "⁹": "9",
    }

    for k, v in sup_map.items():
        s = s.replace(k, v)

    # 统一乘号
    s = s.replace("×", "x").replace("X", "x")

    return s

def clean_indicator_value(raw_val, report_type, indicator_index):
    if raw_val is None:
        return -1, False

    if isinstance(raw_val, str):
        s = raw_val.strip()
        if not s:
            return -1, False


        rule = (REPORT_VALUE_FILTER_MAP.get(report_type, {}) or {}).get(indicator_index)
        if rule:
            before = s
            removed_chars = rule.get("remove_chars", []) or []
            removed_regex = rule.get("removed_regex", []) or []
            
            for ch in removed_chars:
                s = s.replace(ch, "")
            for pattern in removed_regex:
                s = re.sub(pattern, "", s).strip()

            if s != before:
                print(
                    f"[指标值筛除] 报告类型={report_type} 指标名称={indicator_index} "
                    f"筛除前='{before}' 筛除后='{s}' "
                )


        # 归一化科学计数法（×10⁶ -> x10^6）
        s_norm = normalize_scientific_text(s)

        # 1️⃣ 标准科学计数法（支持 x10^6）
        sci_match = re.search(
            r'([-+]?\d+\.?\d*)\s*x\s*10\^?\s*([-+]?\d+)',
            s_norm
        )
        if sci_match:
            try:
                base = float(sci_match.group(1))
                exp = int(sci_match.group(2))
                return base * (10 ** exp), False
            except Exception:
                pass

        # 2️⃣ 普通数字（float 本身也支持 1.2e-3）
        try:
            return float(s_norm), False
        except Exception:
            # 3️⃣ 文本值
            return s_norm, True

    # 非字符串：保持你原逻辑
    try:
        return float(raw_val), False
    except Exception:
        return -1, False

def build_data_map_for_assemble(normalized_json):
    """
    normalized_json: {指标名: {val:..., unit:...}}
    输出： data_map（供 assemble_single_report 匹配与单位换算使用）
    """
    if not isinstance(normalized_json, dict):
        return {}

    data_map_for_assemble = {}

    for k, v in normalized_json.items():
        if not isinstance(v, dict):
            continue

        raw_val = v.get("val", -1)
        raw_unit = str(v.get("unit", "")).strip()

        # ✅ 核心新增逻辑：字符串数字 → 数值
        if isinstance(raw_val, str):
            s = raw_val.strip()
            if s != "":
                try:
                    # 尝试转 float（支持 int / float / 科学计数法）
                    num = float(s)

                    # 如果是整数形式，转 int（保持语义更干净）
                    if num.is_integer():
                        raw_val = int(num)
                    else:
                        raw_val = num
                except ValueError:
                    # 转失败，说明是“语义型文本”，保持字符串
                    raw_val = raw_val

        data_map_for_assemble[k] = {
            "val": raw_val,
            "unit": raw_unit
        }

    return data_map_for_assemble

def extract_list_from_text(texts, num):
    """
    从文本中提取最后一个括号（()或[]）中的最多 num 个数字，返回一个长度为 num 的列表。
    :param texts: 字符串文本
    :param num: 要提取的数字个数
    :return: 包含 num 个数字的列表，不足部分以 -1 补齐
    """
    # 匹配所有 () 或 [] 中的内容
    bracket_contents = re.findall(r'\(([^()]*)\)|\[([^\[\]]*)\]', texts)
    if not bracket_contents:
        return [-1] * num

    # 因为正则有两个分组，所以要取非空的那一个
    contents = [c1 if c1 else c2 for c1, c2 in bracket_contents]

    # 取最后一个括号内的内容
    last_content = contents[-1]

    # 匹配其中的数字（整数或小数，含正负号）
    nums = re.findall(r'[+-]?\d+(?:\.\d+)?', last_content)

    result = []
    for num_str in nums[:num]:
        try:
            num_val = float(num_str)
            if num_val.is_integer():
                num_val = int(num_val)
            result.append(num_val)
        except:
            continue

    # 补齐为指定长度
    while len(result) < num:
        result.append(-1)

    return result

def extract_str_list_from_text(texts, num):

    bracket_contents = re.findall(r'\(([^()]*)\)|\[([^\[\]]*)\]', texts)
    if not bracket_contents:
        return [''] * num

    contents = [c1 if c1 else c2 for c1, c2 in bracket_contents]
    last_content = contents[-1]

    items = [x.strip() for x in last_content.split(',')]

    result = items[:num]
    while len(result) < num:
        result.append('')

    return result

def normalize_llm_output_keys(llm_output: dict) -> dict:
    """
    规范化 llm_output 顶层 key（配置驱动）：
    - 删除 config["remove_chars"] 中的字符
    - 按 config["char_convert_map"] 转换字符

    - 仅处理顶层 key
    - value 不修改
    - key 冲突：后者覆盖前者
    """
    if not isinstance(llm_output, dict):
        return llm_output

    config = KEY_NORMALIZE_CONFIG

    remove_chars = config.get("remove_chars", [])
    char_convert_map = config.get("char_convert_map", {})

    normalized_output = {}

    for key, value in llm_output.items():
        new_key = key

        if isinstance(new_key, str):
            # 1️⃣ 删除指定字符
            for ch in remove_chars:
                new_key = new_key.replace(ch, "")

            # 2️⃣ 转换指定字符
            for old, new in char_convert_map.items():
                new_key = new_key.replace(old, new)
            

        normalized_output[new_key] = value

    return normalized_output

def apply_report_value_keep_map(report_type: int, report_data):
    keep_cfg = REPORT_VALUE_KEEP_MAP.get(report_type)
    if not keep_cfg or not isinstance(report_data, list):
        return report_data

    def keep_one(idx: int, v):
        cfg = keep_cfg.get(idx)
        if not cfg:
            return v

        keywords = cfg.get("keep_contains")
        if not keywords:
            return v

        if not isinstance(v, str):
            return v
        s = v.strip()
        if not s:
            return v

        for kw in keywords:
            if kw in s:
                return kw
        return v

    out = []
    for i, v in enumerate(report_data):
        if isinstance(v, list):
            out.append([keep_one(i, x) for x in v])
        else:
            out.append(keep_one(i, v))
    return out


def apply_report_value_map(report_type: int, report_data):
    """
    对组装完成的 report_data 做值映射（单位换算后调用）
    - 映射表结构：report_type -> index -> [ {"from":[...], "to":...}, ... ]
    - 不支持 __DEFAULT__：没配置就不处理
    - 只改 report_data，不修改 unit
    """
    mapping_for_report = REPORT_VALUE_MAP.get(report_type)
    if not mapping_for_report:
        return report_data

    def normalize_candidates(v):
        """
        为了让匹配更稳：同时给出若干候选形态
        - 原值
        - str 去空格
        - 如果是数字字符串，转 float / int（0.0/0 都给）
        """
        cands = [v]

        if isinstance(v, str):
            s = v.strip()
            cands.append(s)

            # 尝试解析数字字符串
            try:
                f = float(s)
                cands.append(f)
                if f.is_integer():
                    cands.append(int(f))
            except Exception:
                pass

        elif isinstance(v, (int, float)):
            # 兼容 0.0 vs 0
            if float(v).is_integer():
                cands.append(int(v))
            cands.append(float(v))

        return cands

    def map_one(idx: int, v):
        rules = mapping_for_report.get(idx)
        if not rules:
            return v

        candidates = normalize_candidates(v)

        # 按规则顺序匹配，命中第一条即返回
        for rule in rules:
            from_list = rule.get("from", [])
            to_val = rule.get("to", v)

            # 只要候选值里有任意一个在 from_list 中，就命中
            # （from_list 用 list 也行；数据大可改 set 提升性能）
            for c in candidates:
                if c in from_list:
                    return to_val

        return v

    # 通用报告：report_data 是 list，按位置 index 映射
    if isinstance(report_data, list):
        out = []
        for i, v in enumerate(report_data):
            if isinstance(v, list):
                # 嵌套 list：逐元素用同一个 index 的规则映射
                out.append([map_one(i, x) for x in v])
            else:
                out.append(map_one(i, v))
        return out

    # 非 list（极少）：没有 index 语义，直接返回
    return report_data

def convert_and_save_target_format(original_data):
    """
    输入：原始格式的数据字典 (Nested Dict)
    输出：保存扁平化后的目标格式 JSON (Flat List with pic_from)
    """
    print(f"正在将原始数据转换为目标扁平格式...")
    
    flat_info_list = []
    flat_llm_list = []
    
    # 1. 遍历原始数据的 info 字典
    # original_data['info'] 结构: { "img_name": { "data": [...], "llm_output": {...} }, ... }
    raw_info_map = original_data.get('info', {})
    
    for img_name, content in raw_info_map.items():
        # --- 处理 info (报告列表) ---
        reports = content.get('data', [])
        for report in reports:
            # 深拷贝一份，避免修改原数据
            new_report = copy.deepcopy(report)
            # 注入图片来源
            new_report['pic_from'] = img_name
            new_report['file_from'] = img_name
            flat_info_list.append(new_report)
            
        # --- 处理 llm_output ---
        llm_out = content.get('llm_output', {})
        # 确保 llm_out 是字典且不为空 (视需求而定，空字典是否要保留)
        if isinstance(llm_out, dict):
            new_llm = copy.deepcopy(llm_out)
            # 注入图片来源
            new_llm['pic_from'] = img_name
            new_llm['file_from'] = img_name
            flat_llm_list.append(new_llm)

    # 2. 组装最终结果
    target_output = {
        "info": flat_info_list,
        "total_img_number": original_data.get('total_img_number', 0),
        "solvable_img_number": original_data.get('solvable_img_number', 0),
        "unsolvable_img_number": original_data.get('unsolvable_img_number', 0),
        "unsolvable_img_unclassified_number": original_data.get('unsolvable_img_unclassified_number', 0),
        "unsolvable_img_vig_not_zero_number": original_data.get('unsolvable_img_vig_not_zero_number', 0),
        "solvable_img_list": original_data.get('solvable_img_list', []),
        "unsolvable_img_list": original_data.get('unsolvable_img_list', []),
        "unsolvable_img_list_unclassified": original_data.get('unsolvable_img_list_unclassified', []),
        "unsolvable_img_list_vig_not_zero": original_data.get('unsolvable_img_list_vig_not_zero', []),
        "llm_output": flat_llm_list,
        "img_size_list": original_data.get('img_size_list', []),
        "img_w_h_list": original_data.get('img_w_h_list', [])
    }

    return target_output

def need_extra_ocr(template_id, report_types, rules):
    """
    判断当前(template_id, report_types)是否需要额外OCR文本辅助。
    返回: (need: bool, reason: str|None)

    命中规则判定：
        template_ok = (rule.template_ids为空) 或 (template_id在rule.template_ids中)
        report_ok   = (rule.report_types为空) 或 (report_types与rule.report_types有交集)
        template_ok AND report_ok → 命中
    """
    report_types_set = set(report_types or [])

    for rule in rules:
        t_ids = set(rule.get("template_ids") or [])
        r_types = set(rule.get("report_types") or [])
        reason = rule.get("reason", "未说明原因")

        template_ok = (not t_ids) or (template_id in t_ids)
        report_ok = (not r_types) or (len(report_types_set & r_types) > 0)

        if template_ok and report_ok:
            return True, reason

    return False, None

def get_img_meta(path):
    try:
        img_size_str = f"{os.path.getsize(path) / 1024 / 1024:.2f} MB"
    except:
        img_size_str = "Unknown"

    try:
        with Image.open(path) as img:
            w, h = img.size
        img_w_h = [w, h]
    except:
        img_w_h = [-1, -1]

    return img_size_str, img_w_h

