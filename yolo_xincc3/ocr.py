from cnocr import CnOcr
from PIL import Image
from .utils import get_size
import numpy as np
import json
import re

ocr = CnOcr(
    rec_model_name='densenet_lite_136-gru',     # 识别模型名称
    det_model_name='ch_PP-OCRv4_det',           # 检测模型名称
    rec_root='./model/cnocr',
    det_root='./model/cnocr',
    rec_model_backend='onnx',                   # 识别后端
    det_model_backend='onnx',                   # 检测后端
) 

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # 将 ndarray 转换为列表
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}  # 处理字典
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]  # 处理列表
    return obj

def crop_and_ocr(ocr, img_path, save_json=False):

    with Image.open(img_path) as img:
        w, h = img.size
        resize = 0.25
        top = int(h * (1 - resize))
        cropped_img = img.crop((0, top, w, h))

        img_array = np.array(cropped_img)
        # crop_img = Image.fromarray(img_array)
        # crop_img.save(f"crop.png")
        # exit()

        result = ocr.ocr(img_array)
    
    if save_json:
        save_path = img_path.replace('.png', '.json').replace('.jpg', '.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            result_serializable = convert_ndarray_to_list(result)
            json.dump(result_serializable, f, ensure_ascii=False, indent=4)

    return result

def OCR_digital_extraction(img_path, debug=False):

    result_ocr = crop_and_ocr(ocr, img_path, save_json = True)
    result_text, ovary_exist, dist_exist, endom_exist, plus_exist = get_size(result_ocr, debug)
    num_size_info = re.findall(r'\d+\.\d+', result_text)  # 一个或多个数字 + 一个小数点 + 一个或多个数字

    return num_size_info, ovary_exist, dist_exist, endom_exist, plus_exist






