import sys
import os
from module_ocr_data_process import parse_llm_json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llm_polish")))
from llm_api import ali_api_vision,ali_api_text
from llm_prompt import TRAIN_VISION_EXTRATCT_SYSTEM_PROMPT,TRAIN_TEXT_EXTRATCT_SYSTEM_PROMPT

def get_train_required_info_basic(img_path, ocr_text, key_vison, key_text, vision_model=0, text_model=2, vision_or_text="vision"):
    """
    利用不含医疗属性的基础大模型，从图片/ocr文本中尝试提取治疗结论信息
    """
    if vision_or_text == "vision":

        prompt = (
            """
            请从下面这张医疗报告图片中，执行【严格的信息抽取任务】：

            【抽取目标】
            1. 仅抽取报告中【明确存在】的以下栏目内容之一或多个（如果有）：
            - 诊断
            - 临床诊断
            - 影像学诊断
            - 结论
            - 提示
            - 印象
            - Impression / Conclusion / Diagnosis

            【说明】
            - “提示 / 印象 / Impression” 等栏目中的内容，通常包含医生原文中的不确定性判断，
            这些措辞属于诊断原文的一部分，必须完整、原样抽取。

            2. 如果报告中【不存在任何上述栏目或等价表达】，请返回空结果，不要编写任何内容。

            【重要约束（必须遵守）】
            - ❌ 不允许根据指标、图片内容或常识进行任何医学推断
            - ❌ 不允许补全、总结、改写、解释原文
            - ❌ 不允许合并多个地方的信息
            - ❌ 不允许弱化、删除或重写原文中的不确定性措辞
            （如“考虑”“可能”“提示”“不排除”“倾向”“建议复查”等）
            - ✅ 只能逐字或近似逐字抽取图片中“医生已经写出的文字”
            - ✅ 抽取内容必须能在图片中被直接指认

            【输出要求】
            - 只输出 JSON
            - 不要输出多余解释说明
            - 所有字段必须出现

            【输出格式】
            {
            "section_found": true 或 false,  // 表示图片中是否存在明确的诊断/结论栏目
            "diagnosis_text": "抽取到的原文诊断内容；若未找到则为空字符串"
            }

            【一致性要求】
            - 如果你不确定图片中是否存在明确的诊断/结论栏目，请选择 section_found = false
            - 宁可漏抽，也不要编造
            """
        )

        # 调用视觉模型
        content_text = ali_api_vision(TRAIN_VISION_EXTRATCT_SYSTEM_PROMPT, prompt, img_path, vision_model, key_vison, temperature = 0.0)
    
    else :
        prompt = (
            """
            请从下面这份医疗报告的文本内容中，执行【严格的信息抽取任务】：

            【抽取目标】
            1. 仅抽取报告中【明确存在】的以下栏目内容之一或多个（如果有）：
            - 诊断
            - 临床诊断
            - 影像学诊断
            - 结论
            - 提示
            - 印象
            - Impression / Conclusion / Diagnosis

            【说明】
            - “提示 / 印象 / Impression” 等栏目中的内容，通常包含医生原文中的不确定性判断，
            这些措辞属于诊断原文的一部分，必须完整、原样抽取。

            2. 如果报告中【不存在任何上述栏目或等价表达】，请返回空结果，不要编写任何内容。

            【重要约束（必须遵守）】
            - ❌ 不允许根据指标、图片内容或常识进行任何医学推断
            - ❌ 不允许补全、总结、改写、解释原文
            - ❌ 不允许合并多个地方的信息
            - ❌ 不允许弱化、删除或重写原文中的不确定性措辞
            （如“考虑”“可能”“提示”“不排除”“倾向”“建议复查”等）
            - ✅ 只能逐字或近似逐字抽取文本中“医生已经写出的文字”
            - ✅ 抽取内容必须能在文本中被直接指认

            【输出要求】
            - 只输出 JSON
            - 不要输出多余解释说明
            - 所有字段必须出现

            【输出格式】
            {
            "section_found": true 或 false,  // 表示文本中是否存在明确的诊断/结论栏目
            "diagnosis_text": "抽取到的原文诊断内容；若未找到则为空字符串"
            }

            【一致性要求】
            - 如果你不确定文本中是否存在明确的诊断/结论栏目，请选择 section_found = false
            - 宁可漏抽，也不要编造
            """
        )

        # 调用文本模型
        content_text = ali_api_text(TRAIN_TEXT_EXTRATCT_SYSTEM_PROMPT, prompt, ocr_text, text_model, key_text, temperature=0.0)

    train_required_info = parse_llm_json(content_text)
    
    return train_required_info





