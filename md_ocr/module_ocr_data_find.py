import sys
import os
import re
import json
import ast
import concurrent.futures
from typing import Optional


from keywords import NEED_EXTRA_OCR_RULES
from keywords import SECONDARY_EXTRACTION_CONFIG

from module_ocr_data_process import need_extra_ocr
from module_ocr_data_process import extract_list_from_text, extract_str_list_from_text


# 先加路径，再 import llm_polish 里的东西（更稳）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llm_polish")))
from llm_api import ali_api_text, ali_api_vision
from llm_prompt import (
    DEFAULT_EXTRATCT_SYSTEM_PROMPT,

    DEFAULT_VISION_EXTRATCT_SYSTEM_PROMPT,
    DEFAULT_TEXT_EXTRATCT_SYSTEM_PROMPT,

    DEFAULT_VISION_EXTRATCT_ENHANCER,
    DEFAULT_TEXT_EXTRATCT_ENHANCER,

)

class FindMethod:
    def __init__(self, ocr_text, img_path, report_types, template_id):
        self.ocr_text = ocr_text
        self.img_path = img_path
        self.report_types = report_types #类别列表
        self.template_id = template_id #模板id

    def extract_all_indicators_from_general_image(
        self,
        key_vision,
        vision_model = 0
    ):
        """
        只负责：调用 LLM/视觉模型，得到 final_response（字符串）
        不再负责：JSON解析、字段规范化、数值清洗
        """
        extra_text = ""

        extra_ocr_text = f"""
            补充要求:结合对报告进行OCR得到的报告纯文本信息，进行医学检验指标提取。
            以下是得到的报告纯文本信息:
            {self.ocr_text}
            """

        need_ocr, reason = need_extra_ocr(self.template_id, self.report_types,NEED_EXTRA_OCR_RULES)
        if need_ocr:
            extra_text = extra_ocr_text
            print(
                f"命中需要额外OCR信息规则。 原因:{reason}\n"
            )

                # 视觉提取 Prompt

        prompt_vision = f"""
        请从下面的医疗报告内容中，提取所有出现的医学检验指标，并严格按下面要求整理成 JSON 输出。
        只允许输出合法 JSON，不得包含任何分析、解释、描述性文字或结论。
        要求：
        1. JSON 顶层 Key 为提取到的检验项目名称（即报告中显示的项目名，包含定性结果的项目也必须提取）。
        2. 每个指标包含字段：val 与 unit
        - val：报告中显示的原始结果文本(如“未检出”，“不存在”，“阳性”，“5.2”等）
        - unit：报告中对应的单位；若无单位，必须为字符串 ""（两个双引号）
        3. 即使结果为定性描述（如“未检出”，“不存在”，“阴性”，“阳性”等），也必须作为 val 输出。
        4. 若未提取到任何指标，则仅输出空 JSON：{{}}
        {extra_text}
        """

        print(f"--- 视觉模型Prompt ---\n{prompt_vision}")

        response_vision = ali_api_vision(
            DEFAULT_VISION_EXTRATCT_SYSTEM_PROMPT,
            prompt_vision,
            self.img_path,
            vision_model,
            key_vision,
            prompt_enhancer=DEFAULT_VISION_EXTRATCT_ENHANCER,
            temperature=0.0,
            )
        print(f"--- 视觉模型回复  ---\n{response_vision}")

        return response_vision

    # --- B超相关逻辑 (并发执行) ---
    def _fetch_b_info_task(
        self,
        prompt: str,
        expected_length: int,
        input_text: str,
        sys_prompt: str,
        model_number: int,
        key: str,
        prompt_enhancer: Optional[str] = None,  # ✅ Python 3.9 兼容
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        content_text = ali_api_text(
            sys_prompt=sys_prompt,
            prompt=prompt,
            input_text=input_text,
            model_number=model_number,
            key=key,
            prompt_enhancer=prompt_enhancer,
            temperature=temperature,
            top_p=top_p,
        )

        return extract_list_from_text(content_text, expected_length)

    def find_b_info_in_text(self, key):
        input_text = self.ocr_text
        model_number = 2

        prompt_intima = (
            "以下是一份b超报告的文字内容，从以下内容中提取子宫内膜厚度信息(单位位毫米)为一个长度为2的列表，"
            "若包括左子宫内膜厚度和右子宫内膜厚度，则分别为列表的第一第二项;"
            "若只有一个内膜厚度，则列表第二项为-1;"
            "若没有内膜厚度信息，则列表两项均为-1。返回这个长度为2的列表。"
        )
        follicle_acc_info = (
            "以下是一份b超报告的文字内容，从以下内容中关于卵泡尺寸部分提取信息，"
            "若给出卵泡最大值的具体值，则你返回[0]；若只说明了其小于某个值，则返回[-1]；若只说明其大于某个值，则返回[1]。"
        )
        prompt_follicle_right = (
            "以下是一份b超报告的文字内容，从以下内容中提取右侧卵巢卵泡总个数(注意有多个子宫也只有一个卵巢），"
            "右侧最大卵泡尺寸(单位为毫米)。返回一个长度为2的列表代表[右侧卵巢卵泡总个数,右侧最大卵泡尺寸]，若某一项没有则对应值为-1。"
        )
        prompt_follicle_left = (
            "以下是一份b超报告的文字内容，从以下内容中提取左侧卵巢卵泡总个数(注意有多个子宫也只有一个卵巢），"
            "左侧最大卵泡尺寸(单位为毫米)。返回一个长度为2的列表代表[左侧卵巢卵泡总个数,左侧最大卵泡尺寸]，若某一项没有则对应值为-1。"
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_intima = executor.submit(
                self._fetch_b_info_task,
                prompt=prompt_intima,
                expected_length=2,
                input_text=input_text,
                sys_prompt=DEFAULT_TEXT_EXTRATCT_SYSTEM_PROMPT,
                model_number=model_number,
                key=key,
                temperature=0.0,
            )

            future_acc = executor.submit(
                self._fetch_b_info_task,
                prompt=follicle_acc_info,
                expected_length=1,
                input_text=input_text,
                sys_prompt=DEFAULT_TEXT_EXTRATCT_SYSTEM_PROMPT,
                model_number=model_number,
                key=key,
                temperature=0.0,
            )

            future_right = executor.submit(
                self._fetch_b_info_task,
                prompt=prompt_follicle_right,
                expected_length=2,
                input_text=input_text,
                sys_prompt=DEFAULT_TEXT_EXTRATCT_SYSTEM_PROMPT,
                model_number=model_number,
                key=key,
                temperature=0.0,
            )

            future_left = executor.submit(
                self._fetch_b_info_task,
                prompt=prompt_follicle_left,
                expected_length=2,
                input_text=input_text,
                sys_prompt=DEFAULT_TEXT_EXTRATCT_SYSTEM_PROMPT,
                model_number=model_number,
                key=key,
                temperature=0.0,
            )

            b_info_intima = future_intima.result()
            b_info_follicle_acc = future_acc.result()
            b_info_follicle_right = future_right.result()
            b_info_follicle_left = future_left.result()

        # [后续逻辑保持不变]
        b_info_intima = [-1 if x == 0 else x for x in b_info_intima]

        b_info_follicle_right = [-1 if x == 0 else x for x in b_info_follicle_right]
        if b_info_follicle_right[0] == -1:
            b_info_follicle_right = [b_info_follicle_right[0], -1]

        b_info_follicle_left = [-1 if x == 0 else x for x in b_info_follicle_left]
        if b_info_follicle_left[0] == -1:
            b_info_follicle_left = [b_info_follicle_left[0], -1]

        b_info_follicle = [-1, -1, -1]
        right_count = 0 if b_info_follicle_right[0] == -1 else b_info_follicle_right[0]
        left_count = 0 if b_info_follicle_left[0] == -1 else b_info_follicle_left[0]

        b_info_follicle[0] = right_count + left_count
        if b_info_follicle[0] < 0:
            b_info_follicle[0] = 0

        b_info_follicle[1] = [max(b_info_follicle_right[1], b_info_follicle_left[1])]
        b_info_follicle[2] = b_info_follicle_acc[0]

        b_info = []
        b_info.append(b_info_intima)
        for items in b_info_follicle:
            b_info.append(items)

        return b_info

    def find_tct_info_in_vision(self, key):
        model_number = 0  # 视觉模型

        prompt = (
            "这是一张TCT（宫颈细胞学）检查报告图片。\n"
            "请你根据报告中的文字内容，提取报告里用于表示最终检测结论的那一项结果（如检测结果、检查结论、结果判定等）。\n"
            "请尽量保持与报告原文完全一致，不要改写、补充或解释。\n"
            "你只需要输出该结果本身，不要输出任何其它内容。"
        )

        response = ali_api_vision(
            DEFAULT_EXTRATCT_SYSTEM_PROMPT,
            prompt,
            self.img_path,
            model_number,
            key,
            temperature=0.0,
        )

        print(f"tct特殊提取模型原始输出:{response}\n")

        return [response]

    def find_mycoplasma_info_in_vision(self, key):
        model_number = 0  # 视觉模型

        prompt = (
            "这是一张支原体检查报告图片。\n"
            "请你根据报告中的表格、勾选项、结论区内容，判断是否存在以下结论，"
            "并按顺序返回一个长度为6的Python列表：\n\n"
            "1. 解脲支原体\n"
            "2. 人型支原体\n"
            "规则：\n"
            "- 如果明确出现且结论为存在，对应位置为 阳性\n"
            "- 如果明确出现且结论为不存在，对应位置为 阴性\n"
            "- 如果没有出现，对应位置为 不存在\n\n"
            "只返回 Python 列表，例如：[阳性, 阴性]\n"
            "注意！如果报告中结论提到了“未生长支原体”或类似字样，请直接返回列表[阴性, 阴性]，此规则优先级最高\n"
            "不要输出任何解释性文字。"
        )

        response = ali_api_vision(
            DEFAULT_VISION_EXTRATCT_SYSTEM_PROMPT,
            prompt,
            self.img_path,
            model_number,
            key,
            prompt_enhancer=DEFAULT_VISION_EXTRATCT_ENHANCER,
            temperature=0.0,
        )

        print(f"支原体特殊提取模型原始输出:{response}\n")

        try:
            result = extract_str_list_from_text(response, 2)
            if isinstance(result, list) and len(result) == 2:
                return result
        except Exception:
            pass

        return [-1] * 6

        
    def find_neisseria_gonorrhoeae_culture_info_in_vision(self, key):
        model_number = 0  # 视觉模型

        prompt = (
            "这是一张淋球菌检查报告图片。\n"
            "请你根据报告中的内容，判断患者是否感染淋病。并按照以下规则输出\n"
            "规则：\n"
            "- 如果可以明确判断患者感染淋病，输出：阳性\n"
            "- 如果可以明确判断患者没有感染淋病，输出：阴性\n"
            "- 如果无法明确判断患者是否感染淋病，输出：阴性\n\n"
            "你只需要按照规则输出阴性或者阳性这两个字，不要输出任何其它内容。"
        )

        response = ali_api_vision(
            DEFAULT_EXTRATCT_SYSTEM_PROMPT,
            prompt,
            self.img_path,
            model_number,
            key,
            temperature=0.0,
        )

        print(f"淋球菌特殊提取模型原始输出:{response}\n")

        return [response]
    
    def smart_secondary_extraction(
        self,
        report_type,
        target_name_list,
        indicator_value_list,
        indicator_unit_list,
        api_key,
        model_number = 2,
    ):
        config = SECONDARY_EXTRACTION_CONFIG.get(report_type)
        if not config:
            return indicator_value_list, indicator_unit_list

        positions_config = config.get("positions", {})

        for idx, pos_cfg in positions_config.items():
            # 越界保护
            if idx >= len(indicator_value_list):
                continue

            # 只处理字符串值
            if not isinstance(indicator_value_list[idx], str):
                continue

            default_value = pos_cfg.get("default_value", -1)
            default_unit = pos_cfg.get("default_unit", "")

            # 1. 获取别名配置
            aliases = pos_cfg.get("aliases", [])
            alias_prompt_part = ""
            if aliases:
                alias_str = "、".join(aliases)
                alias_prompt_part = f"(常见名称包含: {alias_str})"

            # 2. 获取特别说明配置
            special_instruction = pos_cfg.get("special_instruction", "")
            special_instruction_prompt = ""
            if special_instruction:
                # 注意：这里在开头加了换行符，并保持了缩进格式，以便插入 Prompt 时排版美观
                special_instruction_prompt = f"\n            特别说明: {special_instruction}"

            print(
                f"报告类型{report_type}，"
                f"{target_name_list[idx]}指标值出现字符串，"
                f"进行智能二次提取，原始值:{indicator_value_list[idx]}"
            )

            # 3. 构建 Prompt
            # 修改点：将 {special_instruction_prompt} 紧接在第一句话后面
            # 逻辑：
            # - 若为空，代码直接换行到 "要求："，无空行。
            # - 若有值，变量自带 \n，先换行显示说明，再换行到 "要求："。
            prompt = f"""
            从以下医疗报告文本中提取 {target_name_list[idx]}{alias_prompt_part} 的【数值】和【单位】。{special_instruction_prompt}
            要求：
            1. 只返回 JSON,不要输出其它任何内容。
            2.JSON格式如下:
            {{"value": 12.3, "unit": "{default_unit}"}}
            3.如果无法找到该指标的数值,"value"字段设为{default_value}
            4.如果无法找到该指标的单位,"unit"字段设为"{default_unit}"
            """

            response = ali_api_text(
                DEFAULT_EXTRATCT_SYSTEM_PROMPT,
                prompt,
                indicator_value_list[idx],
                model_number,
                api_key,
                temperature=0.0,
            )

            print(f"智能二次提取模型返回原始内容:{response}")

            try:
                data = json.loads(response.strip())
                value = float(data.get("value", default_value))
                unit = str(data.get("unit", default_unit)).strip()
            except Exception as e:
                print(f"智能二次提取解析失败:{e}")
                value, unit = default_value, default_unit

            indicator_value_list[idx] = value
            indicator_unit_list[idx] = unit

            print(f"智能二次提取后{target_name_list[idx]}值:{value}, 单位:{unit}\n")

        return indicator_value_list, indicator_unit_list
