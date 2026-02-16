from keywords import TEMPLATE_FINGERPRINT_LIST

def check_template_type(result):
    """
    输入单张图片的 OCR 结果，返回该图片的特殊类型 ID。
    如果不匹配任何特征，返回 0。

    新结构：
    TEMPLATE_FINGERPRINT_LIST = [
        ["关键词1", "关键词2"],  # template_id = 1
        ["关键词A"],            # template_id = 2
    ]
    """
    if result == "":
        return 0

    combined_text = result

    # 遍历 TEMPLATE_FINGERPRINT_LIST
    for template_index, required_keywords in enumerate(TEMPLATE_FINGERPRINT_LIST):
        current_template_id = template_index + 1

        # required_keywords 现在就是关键词列表
        if required_keywords and all(kwd in combined_text for kwd in required_keywords):
            return current_template_id

    return 0
