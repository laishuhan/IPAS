import os as _os

# 让输出尽量不乱码（Windows）
_os.environ.setdefault("PYTHONUTF8", "1")
_os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# 关闭模型源连通性检查（避免卡在 connectivity check）
_os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

import paddle
from paddleocr import PaddleOCR
import cv2
import os
import json

from keywords import REPORT_TYPE_LIST
from keywords import UNCLASSIFIED_REPORT_ID
from module_ocr_data_process import get_uploade_time
from module_ocr_data_process import get_paths_all_info
from module_ocr_data_process import get_ocr_text_from_paddleocr3

from llm_api import ali_api_vision
from llm_keywords import  ali_api_vision_key_001

from llm_prompt import (
    DEFAULT_ROTATE_SYSTEM_PROMPT,
    DEFAULT_OCR_SYSTEM_PROMPT
)

# -----------------------------
# 全局 OCR 单例（避免反复加载模型）
# -----------------------------
_OCR_INSTANCE = None

def setup_device_auto():
    """
    自动选择设备：有可用 GPU 才用 GPU，否则用 CPU。
    注意：PaddleOCR 3.3.3 不支持 use_gpu 参数，所以使用 paddle.set_device 控制。
    """
    if paddle.is_compiled_with_cuda():
        try:
            if paddle.device.cuda.device_count() > 0:
                paddle.set_device("gpu")
                print("检测到可用 GPU，已设置为 gpu")
                return
        except Exception:
            pass

    paddle.set_device("cpu")
    print("未检测到可用 GPU，已设置为 cpu")


def create_ocr():
    """
    创建并返回 OCR 实例（单例复用）。
    函数名不能变（外部有引用）。
    """
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE

    setup_device_auto()

    _OCR_INSTANCE = PaddleOCR(
        text_detection_model_name="PP-OCRv5_server_det",
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    return _OCR_INSTANCE


def get_img_angle(image_path, method = "DEFAULT", vision_model = 0):
    """
    获得图片相比于原始状态被顺时针的旋转角度（保持原函数名不变）
    """
    if method == "DEFAULT":
        return 0
    
    elif method == "LLM":

        prompt_rotate = ("""
        请判断这张医疗报告图片需要逆时针旋转多少度才能正常阅读。
        只输出一个数字。
        """)

        response_vision = ali_api_vision(
            DEFAULT_ROTATE_SYSTEM_PROMPT,
            prompt_rotate,
            image_path,
            vision_model,
            ali_api_vision_key_001,
            temperature=0.0,
            )

        return int(response_vision.strip())
        
       
    elif method == "OCR":

        ocr = create_ocr()
        ocr_result = ocr.predict(image_path)
        angle = 0
        if isinstance(ocr_result, dict):
            angle = ocr_result.get("doc_preprocessor_res", {}).get("angle", 0)
        elif isinstance(ocr_result, list) and ocr_result:
            first = ocr_result[0]
            if isinstance(first, dict):
                angle = first.get("doc_preprocessor_res", {}).get("angle", 0)
        return angle
    
    else:
        return 0


def get_img_text(image_path, method = "DEFAULT", vision_model = 0):
    """
    获取图片 OCR 文本（保持原函数名不变）
    """
    if method == "DEFAULT":
        return ""
    
    elif method == "LLM":

        prompt_rotate = ("""
        请提取这张医疗报告中的全部文字，并合并成一段长文本。
        只输出一个字符串。
        """)

        response_vision = ali_api_vision(
            DEFAULT_OCR_SYSTEM_PROMPT,
            prompt_rotate,
            image_path,
            vision_model,
            ali_api_vision_key_001,
            temperature=0.0,
            )

        return response_vision.strip()
       
    elif method == "OCR":
        ocr = create_ocr()
        ocr_result = ocr.predict(image_path)
        texts = []
        if isinstance(ocr_result, dict):
            texts = ocr_result.get("rec_texts", []) or []
        elif isinstance(ocr_result, list) and ocr_result:
            first = ocr_result[0]
            if isinstance(first, dict):
                texts = first.get("rec_texts", []) or []
        return "".join(texts)
    
    else:
        return ""


def reversed_rotate_image(angle, image_path, suffix):
    """
    根据检测到的角度，逆时针旋转图片，并保存为带后缀的新文件
    :param angle: 检测到图片被旋转的角度
    :param image_path: 图片绝对路径
    :param suffix: 新文件名后缀
    :return: new_image_path
    """

    if angle is None or abs(angle) < 1e-6:
        return image_path

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片，请检查路径是否正确: {image_path}")

    h, w = image.shape[:2]

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    dir_name = os.path.dirname(image_path)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    new_image_name = f"{name}{suffix}{ext}"
    new_image_path = os.path.join(dir_name, new_image_name)

    cv2.imwrite(new_image_path, rotated)
    return new_image_path


def get_basic_info_independent(task_record_path, ocr_result_path, is_idp_rotate, is_idp_ocr, method_rotate , method_text, vision_model_rotate = 0, vision_model_text = 0):
    """
    完全不依赖接口，从 task_record.json 读取所需基本信息，再后处理。
    保持原函数名不变。
    """
    try:
        # 读取 task_record.json
        with open(task_record_path, "r", encoding="utf-8") as f:
            task_record = json.load(f)

        task_key = next(iter(task_record))
        task_info = task_record[task_key]

        task_id = task_info["task_id"]
        file_count = task_info["file_count"]
        page_count = task_info["page_count"]
        create_time = task_info["create_time"]
        pages = task_info["pages"]

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

        # 3. 类别 ID 去重
        for item in paths_all_info_list:
            item[1] = list(set(item[1]))

        print(f"类别名映射为数字的paths_all_info_list:{paths_all_info_list}\n")

        if is_idp_rotate or is_idp_ocr:
            if method_rotate == "OCR" or method_text == "OCR":
                # 初始化 OCR（只做一次）
                try:
                    ocr = create_ocr()
                except Exception as e:
                    print(f"[ERROR] OCR 初始化失败: {e}")
                    import traceback
                    traceback.print_exc()
                    return [None, [], (None, None, None), 2, 0, [], 0, 0, 0, 0, 0, [], [], [], []]

        # 逐图：角度检测 -> 回正 -> 回正后 OCR
        all_ocr_text_results = []
        for item in paths_all_info_list:
            path = item[0]
            
            if is_idp_rotate:
                angle = get_img_angle(path, method_rotate, vision_model_rotate)
                print(f"通过方法{method_rotate}检测到图片{path}相比于原始状态被顺时针旋转角度{angle}度")
                new_image_path = reversed_rotate_image(angle, path, "_rotate_BY_mdocr")
                print(f"已反向旋转{angle}度回正，新图片路径为{new_image_path}")
                item[0] = new_image_path

            if is_idp_ocr:
                ocr_text = get_img_text(path, method_text,vision_model_text)
                print(f"图片{new_image_path} 通过方法{method_text}得到的OCR内容为:{ocr_text}")
                all_ocr_text_results.append(ocr_text)
        
        if not is_idp_ocr:
            with open(ocr_result_path, 'r', encoding='utf-8') as f:
                ocr_result = json.load(f)
            #提取文字内容部分
            all_ocr_text_results = []
            for ocr in ocr_result:
                all_ocr_text_results.append(get_ocr_text_from_paddleocr3(ocr))

        # 统计（注意：这里使用的路径已经是可能回正后的路径）
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
            if UNCLASSIFIED_REPORT_ID in item[1]
        ]

        unsolvable_img_list_vig_not_zero = [
            [item[0], item[3]]
            for item in paths_all_info_list
            if item[3] != 0
        ]

        print(f"可处理的图片列表为:{solvable_img_list}")
        print(f"不可处理的图片列表为:{unsolvable_img_list}")
        print(f"由于无法分类不可处理的图片列表为:{unsolvable_img_list_unclassified}")
        print(f"由于vig不为0不可处理的图片列表为:{unsolvable_img_list_vig_not_zero}")

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

        result = [
            task_id,
            paths_all_info_list,
            (year, month, day),
            2,  # 经期占位
            0,  # 孕期占位
            all_ocr_text_results,
            total_img_number,
            solvable_img_number,
            unsolvable_img_number,
            unsolvable_img_unclassified_number,
            unsolvable_img_vig_not_zero_number,
            solvable_img_list,
            unsolvable_img_list,
            unsolvable_img_list_unclassified,
            unsolvable_img_list_vig_not_zero,
        ]
        return result

    except Exception as e:
        print(f"[错误] 读取基本信息失败: {e}")
        import traceback
        traceback.print_exc()
        # ✅ 返回 15 项占位，避免主程序解包崩
        return [None, [], (None, None, None), 2, 0, [], 0, 0, 0, 0, 0, [], [], [], []]
