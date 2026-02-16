import os
import re
import cv2
import numpy as np
from difflib import SequenceMatcher

'''
===================分割报告中的图片========================
'''
def order_points(pts):
    """将四个点排序为左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    return rect

def extract_ultrasound_regions(image_path, min_area=50000, max_aspect_ratio=2.0):
    # 读取图像并预处理
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # 形态学操作闭合小孔洞
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(closed, (5, 5), 0)
    
    # 边缘检测+膨胀连接断裂边缘
    edges = cv2.Canny(blurred, 30, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for cnt in contours:
        # 面积过滤
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        
        # 计算宽高比（考虑旋转）
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if aspect_ratio > max_aspect_ratio:
            continue
        
        valid_contours.append((rect, cnt))
    
    # 使用行分组的方式排序：先将y坐标相近的区域分为同一行，然后在每行内按x坐标排序
    def sort_regions_by_position(contours):
        if not contours:
            return contours
        
        # 先粗略按y坐标排序
        contours_with_pos = [(rect, cnt, rect[0][0], rect[0][1]) for rect, cnt in contours]
        contours_with_pos.sort(key=lambda x: x[3])  # 按y坐标排序
        
        # 分组：将y坐标相近的区域归为同一行
        groups = []
        current_group = [contours_with_pos[0]]
        
        for i in range(1, len(contours_with_pos)):
            y_current = contours_with_pos[i][3]
            y_prev = current_group[-1][3]
            # 如果y坐标差异小于100像素，认为是同一行
            if abs(y_current - y_prev) < 100:
                current_group.append(contours_with_pos[i])
            else:
                groups.append(current_group)
                current_group = [contours_with_pos[i]]
        
        groups.append(current_group)
        
        # 在每个组（行）内按x坐标从左到右排序
        sorted_contours = []
        for group in groups:
            group.sort(key=lambda x: x[2])  # 按x坐标排序
            sorted_contours.extend([(rect, cnt) for rect, cnt, _, _ in group])
        
        return sorted_contours
    
    # 使用改进的排序函数
    valid_contours = sort_regions_by_position(valid_contours)
    
    results = []
    for rect, cnt in valid_contours:
        # 获取旋转矩形四个顶点
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 对顶点进行排序
        ordered_pts = order_points(box).astype(np.float32)
        
        # 计算目标尺寸
        (tl, tr, br, bl) = ordered_pts
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        max_width = max(int(width_top), int(width_bottom))
        
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_height = max(int(height_left), int(height_right))
        
        # 定义目标点坐标
        dst = np.array([
            [0, 0],
            [max_width-1, 0],
            [max_width-1, max_height-1],
            [0, max_height-1]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(ordered_pts, dst)
        
        # 执行变换并裁剪
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        results.append(warped)
    
    return results

def process_single_image(input_file_path, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # 检查文件是否存在
    if not os.path.isfile(input_file_path):
        print(f"Error: File {input_file_path} does not exist.")
        return
    # 检查文件扩展名是否有效
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    if not input_file_path.lower().endswith(valid_extensions):
        print(f"Error: File {input_file_path} has an unsupported extension.")
        return
    
    print(f"正在处理图像: {input_file_path}")
    
    # 提取超声区域
    regions = extract_ultrasound_regions(input_file_path)

    # 获取文件的基本名称
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}")
    save_path = os.path.join(output_path, "crop")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存每个区域，按照从上到下、从左到右的顺序编号
    for i, region in enumerate(regions):
        region_number = i + 1
        output = os.path.join(save_path, f"{base_name}_region_{region_number}.png")
        cv2.imwrite(output, region)
    
    return output_path, base_name

def preprocess(path):
    def extract_number(file_path):
        filename = os.path.basename(file_path)  # 获取文件名（不含路径）
        num_str = ''
        for char in filename:
            if char.isdigit():  # 当前字符是数字
                num_str += char
            elif num_str:  # 遇到非数字且已提取到数字，结束提取
                break
            # 若未遇到数字则继续扫描
        return int(num_str) if num_str else 0  # 转换数字或返回0（无数字时）
    
    if not os.path.exists(path):
        print(f"Error: File {path} does not exist.")
        return
    
    if os.path.isdir(path):
        file_list = []
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path):
                file_list.append(os.path.abspath(full_path))
        # 按文件名开头的数字排序，数字相同则按原始文件名排序
        file_list.sort(key=lambda x: (extract_number(x), os.path.basename(x)))
        return file_list
    else:
        return [os.path.abspath(path)]


def sort_by_region_number(strings):
    def get_region_number(s):
        match = re.search(r'_region_(\d+)', s)
        return int(match.group(1)) if match else 0
    
    return sorted(strings, key=get_region_number)



'''
===================尺寸识别========================
'''

# 尺寸相关关键词列表
SIZE_KEYWORDS = ["Dist", "D1st", "Dlst", "tDist", "lDist", "mm", "cm", 'MM', 'Dtst', 'Disl', 'FDist']
ULTRASOUND_KEYWORDS = ["Ut-Endom.Th.", "Ut-Endom", "Endom"]
DIMENSION_KEYWORDS = ["1D", "2D", "3D", "4D", "5D"]
OVARY_KEYWORDS = ["ovary", "0vary", "Ovary"]
STANDARD_UNITS = ['mm', 'cm']

# OCR错误映射表
DIGIT_CORRECTIONS = {
    'O': '0', 'o': '0', 'Q': '0',
    'L': '1', 'l': '1', 'I': '1', '|': '1', 'i': '1',
    'Z': '2', 'z': '2',
    'A': '4', 't': '4',
    'S': '5', 's': '5',
    'G': '6', 'b': '6',
    'T': '7', 'J': '7',
    'B': '8', 'R': '8',
    'g': '9', 'q': '9',
    ',': '.', '·': '.', '`': '.', "'": '.', ':': '.', '、': '.', '-': '.',
}

UNIT_CORRECTIONS = {
    'mn': 'mm', 'nm': 'mm', 'rn': 'mm', 'nn': 'mm', 'rnm': 'mm', 'nrn': 'mm',
    'cn': 'cm', 'crn': 'cm', 'em': 'cm',
    'mm': 'mm', 'cm': 'cm'
}

# ==================== 基础工具函数 ====================

def similarity_score(a, b):
    """计算两个字符串的相似度"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def get_bbox_center(position):
    """计算边界框中心点坐标"""
    x_coords = [pt[0] for pt in position]
    y_coords = [pt[1] for pt in position]
    return (min(x_coords) + max(x_coords)) / 2, (min(y_coords) + max(y_coords)) / 2

def get_bbox_bounds(position):
    """获取边界框的边界坐标"""
    x_coords = [pt[0] for pt in position]
    y_coords = [pt[1] for pt in position]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def calculate_distance(pos1, pos2):
    """计算两个位置的距离"""
    center1 = get_bbox_center(pos1)
    center2 = get_bbox_center(pos2)
    return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

def is_same_horizontal_line(pos1, pos2, threshold=20):
    """判断两个位置是否在同一水平线上"""
    _, y1_min, _, y1_max = get_bbox_bounds(pos1)
    _, y2_min, _, y2_max = get_bbox_bounds(pos2)
    return not (y1_max + threshold < y2_min or y2_max + threshold < y1_min)

def contains_potential_size(text):
    """检查文本是否包含潜在的尺寸信息"""
    import re
    return bool(re.search(r'\d+\.?\d*[a-zA-Z]+', text))

# ==================== 关键词检测函数 ====================

def check_keyword_match(text, keywords, threshold=0.7):
    """统一的关键词匹配函数（支持容错）"""
    text_clean = text.strip().lower().replace(':', '')
    for keyword in keywords:
        if text_clean == keyword.lower():
            return True
        if similarity_score(text_clean, keyword.lower()) > threshold:
            return True
    return False

def scan_keywords(result):
    """一次遍历检测所有关键词类型，返回检测结果"""
    dist_keywords = ["Dist", "D1st", "Dlst", "tDist", 'iDist', 'Disl', '1Dist', '2Dist', 'Oist']
    
    keyword_flags = {
        'has_ovary': False,
        'has_dist': False,
        'has_ultrasound': False,
        'has_dimension': False,
        'ovary_item': None
    }
    
    for item in result:
        text = item["text"]
        
        # 检查ovary（最高优先级，找到即返回）
        if not keyword_flags['has_ovary'] and check_keyword_match(text, OVARY_KEYWORDS):
            keyword_flags['has_ovary'] = True
            keyword_flags['ovary_item'] = item
            return keyword_flags  # ovary优先级最高，直接返回
        
        # 检查dist
        if not keyword_flags['has_dist']:
            if any(kw in text for kw in dist_keywords) or check_keyword_match(text, dist_keywords, 0.65):
                keyword_flags['has_dist'] = True
        
        # 检查ultrasound
        if not keyword_flags['has_ultrasound']:
            if any(kw in text for kw in ULTRASOUND_KEYWORDS):
                keyword_flags['has_ultrasound'] = True
        
        # 检查dimension
        if not keyword_flags['has_dimension']:
            if text.strip() in DIMENSION_KEYWORDS:
                keyword_flags['has_dimension'] = True
            else:
                for dim_kw in DIMENSION_KEYWORDS:
                    if text.startswith(dim_kw) and contains_potential_size(text[len(dim_kw):]):
                        keyword_flags['has_dimension'] = True
                        break
    
    return keyword_flags

# ==================== 数字和单位修正函数 ====================

def convert_cm_to_mm(value_str):
    """将cm单位转换为mm单位"""
    import re
    if not value_str:
        return value_str
    
    match = re.match(r'(\d+\.?\d*)(cm|CM)', value_str, re.IGNORECASE)
    if match:
        try:
            mm_value = float(match.group(1)) * 10
            if mm_value == int(mm_value):
                return f"{int(mm_value)}.0mm"
            formatted = f"{mm_value:.2f}".rstrip('0')
            if formatted.endswith('.'):
                formatted += '0'
            return formatted + "mm"
        except ValueError:
            return value_str
    return value_str

def correct_digits_simple(text):
    """简单数字修正"""
    import re
    if not text.strip():
        return text
    
    error_chars = ''.join(DIGIT_CORRECTIONS.keys())
    pattern = rf'([\d{re.escape(error_chars)}]+\.?[\d{re.escape(error_chars)}]*[a-zA-Z]*)'
    
    def correct_match(match):
        corrected = match.group(1)
        for wrong, correct in DIGIT_CORRECTIONS.items():
            corrected = corrected.replace(wrong, correct)
        return corrected
    
    return re.sub(pattern, correct_match, text)

def identify_keywords_in_text(text, keywords):
    """识别文本中的关键词位置"""
    import re
    found = []
    for keyword in keywords:
        pattern = rf'\b{re.escape(keyword)}:?\b'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            found.append({'start': match.start(), 'end': match.end()})
        if 'dist' in keyword.lower():
            pattern = rf'\b\d*{re.escape(keyword)}:?\b'
            for match in re.finditer(pattern, text, re.IGNORECASE):
                found.append({'start': match.start(), 'end': match.end()})
    found.sort(key=lambda x: x['start'])
    return found

def correct_digits_targeted(text):
    """智能数字修正（保护关键词）"""
    if check_keyword_match(text, SIZE_KEYWORDS):
        return text
    
    keywords_info = identify_keywords_in_text(text, SIZE_KEYWORDS)
    if not keywords_info:
        return correct_digits_simple(text)
    
    result = ""
    last_end = 0
    for kw_info in keywords_info:
        result += correct_digits_simple(text[last_end:kw_info['start']])
        result += text[kw_info['start']:kw_info['end']]
        last_end = kw_info['end']
    result += correct_digits_simple(text[last_end:])
    return result

def extract_size_values_tolerant(text, method='hybrid', enable_digit_correction=True):
    """容错的尺寸数值提取"""
    import re
    
    if enable_digit_correction:
        text = correct_digits_targeted(text)
    
    matches = re.findall(r'(\d+\.?\d*)([a-zA-Z]+)', text)
    filtered = []
    
    for number, unit in matches:
        try:
            num_value = float(number)
            if 0.1 <= num_value <= 100:
                filtered.append((number, unit))
        except ValueError:
            continue
    
    if len(filtered) > 1:
        scored = []
        for number, unit in filtered:
            score = 10 if unit.lower() in ['mm', 'cm'] else 0
            try:
                if 0.1 <= float(number) <= 100:
                    score += 5
            except ValueError:
                score = 0
            scored.append((score, number, unit))
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [(scored[0][1], scored[0][2])]
    
    results = []
    for number, unit in filtered:
        corrected_unit = UNIT_CORRECTIONS.get(unit.lower(), unit)
        if corrected_unit not in STANDARD_UNITS:
            best_match, best_score = unit, 0
            for std_unit in STANDARD_UNITS:
                score = similarity_score(unit.lower(), std_unit)
                if score > best_score and score >= 0.6:
                    best_score, best_match = score, std_unit
            corrected_unit = best_match if best_score >= 0.6 else unit
        
        if corrected_unit in STANDARD_UNITS:
            size_value = convert_cm_to_mm(f"{number}{corrected_unit}")
            results.append(size_value)
    
    return results

# ==================== 统一的排序和去重函数 ====================

def sort_size_values(size_items, sort_by_y_only=False, debug=False):
    """统一的尺寸排序函数
    
    Args:
        size_items: 尺寸项列表
        sort_by_y_only: 是否只按Y坐标排序（ovary模式）
        debug: 调试模式
    """
    if not size_items:
        return size_items
    
    if sort_by_y_only:
        # ovary模式：严格按Y坐标排序
        size_items.sort(key=lambda x: get_bbox_center(x['position'])[1])
        if debug:
            print("按Y坐标排序（ovary模式）")
    else:
        # Dist/超声模式：智能排序
        y_coords = [get_bbox_center(item['position'])[1] for item in size_items]
        y_range = max(y_coords) - min(y_coords) if y_coords else 0
        
        if y_range < 20:
            size_items.sort(key=lambda x: get_bbox_center(x['position'])[0])
            if debug:
                print("同一行，按X坐标排序")
        else:
            size_items.sort(key=lambda x: (get_bbox_center(x['position'])[1], 
                                          get_bbox_center(x['position'])[0]))
            if debug:
                print("多行，按Y坐标优先排序")
    
    return size_items

def deduplicate_values(values_list, debug=False):
    """统一的去重函数"""
    unique_values = []
    distance_threshold = 30
    
    for current in values_list:
        is_duplicate = False
        for existing in unique_values:
            if (current['value'] == existing['value'] and 
                calculate_distance(current['position'], existing['position']) < distance_threshold):
                is_duplicate = True
                if debug:
                    print(f"跳过重复: {current['value']}")
                break
        
        if not is_duplicate:
            unique_values.append(current)
            if debug:
                print(f"确认: {current['value']} from '{current['original_text']}'")
    
    return unique_values

# ==================== 新增：检查右侧是否有dist关键词 ====================

def has_dist_keyword_on_right(current_item, all_items, debug=False):
    """检查当前项右侧是否有dist关键词
    
    Args:
        current_item: 当前尺寸项（包含position）
        all_items: OCR结果的所有项
        debug: 调试模式
    
    Returns:
        bool: 右侧有dist关键词返回True，否则返回False
    """
    dist_keywords = ["Dist", "D1st", "Dlst", "tDist", 'iDist', 'Disl', '1Dist', '2Dist', 'Oist', '+D1St']
    
    current_x_max, _, _, _ = get_bbox_bounds(current_item['position'])
    current_center_y = get_bbox_center(current_item['position'])[1]
    
    # 查找右侧的文本项
    for item in all_items:
        item_x_min, _, _, _ = get_bbox_bounds(item['position'])
        item_center_y = get_bbox_center(item['position'])[1]
        
        # 判断是否在右侧且在同一水平线上
        if item_x_min > current_x_max and abs(item_center_y - current_center_y) < 20:
            text = item['text']
            # 检查是否包含dist关键词
            if any(kw.lower() in text.lower() for kw in dist_keywords):
                if debug:
                    print(f"发现右侧dist关键词: {text}")
                return True
    
    if debug:
        print("右侧未发现dist关键词")
    return False

# ==================== 三大逻辑分支 ====================

def extract_ovary_sizes(result, debug=False):
    """逻辑3: 提取ovary相关尺寸"""
    if debug:
        print("\n=== 逻辑3: ovary分支 ===")
    
    size_items = []
    for item in result:
        if check_keyword_match(item["text"], OVARY_KEYWORDS):
            continue  # 跳过关键词本身
        
        if contains_potential_size(item["text"]):
            values = extract_size_values_tolerant(item["text"])
            for value in values:
                size_items.append({
                    'value': value,
                    'position': item['position'],
                    'original_text': item["text"]
                })
                if debug:
                    print(f"提取: {value} from '{item['text']}'")
    
    return sort_size_values(size_items, sort_by_y_only=True, debug=debug)

def extract_dist_sizes(result, debug=False):
    """逻辑1: 提取Dist区域尺寸
    
    【修改点1】：添加了对每个尺寸项记录其在原始result中的索引，以便后续检查右侧是否有dist关键词
    """
    if debug:
        print("\n=== 逻辑1: Dist分支 ===")
    
    dist_keywords = ["Dist", "D1st", "Dlst", "tDist", 'iDist', 'Disl', '1Dist', '2Dist', 'Oist']
    extended_keywords = SIZE_KEYWORDS + ['mn', 'nm', 'cn', 'rn', 'nn', 'crn']
    
    # 定位Dist区域
    dist_info = []
    for item in result:
        text = item["text"]
        if any(kw in text for kw in extended_keywords) or check_keyword_match(text, dist_keywords):
            dist_info.append(item)
    
    if not dist_info:
        region_items = result
    else:
        # 计算区域边界
        all_bounds = [get_bbox_bounds(item["position"]) for item in dist_info]
        x_min, y_min = min(b[0] for b in all_bounds), min(b[1] for b in all_bounds)
        x_max, y_max = max(b[2] for b in all_bounds), max(b[3] for b in all_bounds)
        
        # 收集区域内文本
        region_items = []
        margin = 20
        for item in result:
            center_x, center_y = get_bbox_center(item["position"])
            if (x_min - margin <= center_x <= x_max + margin and 
                y_min - margin <= center_y <= y_max + margin):
                region_items.append(item)
    
    # 提取尺寸
    size_items = []
    for item in region_items:
        text = item["text"]
        has_keywords = any(kw in text for kw in extended_keywords)
        has_size = contains_potential_size(text)
        
        if has_keywords or has_size:
            values = extract_size_values_tolerant(text, enable_digit_correction=True)
            for value in values:
                size_items.append({
                    'value': value,
                    'position': item['position'],
                    'original_text': text,
                    'original_item': item  # 【修改点1】保存原始item引用
                })
                if debug:
                    print(f"提取: {value} from '{text}'")
    
    return sort_size_values(size_items, debug=debug)

def extract_ultrasound_dimension_sizes(result, debug=False):
    """逻辑2: 提取超声/维度尺寸"""
    if debug:
        print("\n=== 逻辑2: 超声/维度分支 ===")
    
    ultrasound_sizes = []
    dimension_sizes = []
    
    # 提取超声厚度尺寸
    for item in result:
        if any(kw in item["text"] for kw in ULTRASOUND_KEYWORDS):
            values = extract_size_values_tolerant(item["text"])
            for value in values:
                ultrasound_sizes.append({
                    'value': value,
                    'position': item['position'],
                    'original_text': item["text"],
                    'type': 'ultrasound'
                })
                if debug:
                    print(f"超声: {value} from '{item['text']}'")
    
    # 提取维度相关尺寸
    dimension_items = [item for item in result if item["text"].strip() in DIMENSION_KEYWORDS]
    
    for item in result:
        text = item["text"]
        # 检查组合格式（如1D14.3mm）
        for dim_kw in DIMENSION_KEYWORDS:
            if text.startswith(dim_kw):
                remaining = text[len(dim_kw):]
                if contains_potential_size(remaining):
                    values = extract_size_values_tolerant(remaining)
                    for value in values:
                        dimension_sizes.append({
                            'value': value,
                            'position': item['position'],
                            'original_text': text,
                            'type': 'dimension'
                        })
                        if debug:
                            print(f"维度组合: {value} from '{text}'")
                    break
    
    # 如果没有组合格式，查找对齐格式
    if not dimension_sizes:
        for dim_item in dimension_items:
            candidates = []
            for item in result:
                if item["text"].strip() in DIMENSION_KEYWORDS:
                    continue
                if is_same_horizontal_line(dim_item['position'], item['position']):
                    if contains_potential_size(item["text"]):
                        values = extract_size_values_tolerant(item["text"])
                        for value in values:
                            distance = calculate_distance(dim_item['position'], item['position'])
                            candidates.append({
                                'value': value,
                                'position': item['position'],
                                'original_text': item["text"],
                                'type': 'dimension',
                                'distance': distance
                            })
            
            if candidates:
                closest = min(candidates, key=lambda x: x['distance'])
                dimension_sizes.append(closest)
                if debug:
                    print(f"维度对齐: {closest['value']} from '{closest['original_text']}'")
    
    # 超声尺寸在前，维度尺寸在后
    ultrasound_sizes = sort_size_values(ultrasound_sizes, debug=debug)
    dimension_sizes = sort_size_values(dimension_sizes, debug=debug)
    
    return ultrasound_sizes + dimension_sizes

# ==================== 主函数 ====================

def get_size_tolerant(result, method='hybrid', enable_digit_correction=True, debug=False):
    """容错尺寸提取主函数
    
    【修改点2】：添加了plus_exist返回值，用于标识最后一个数字右侧是否没有dist信息
    """
    if debug:
        print("=== 容错尺寸提取开始 ===")
    
    # 一次扫描检测所有关键词
    flags = scan_keywords(result)
    
    # 优先级1: ovary
    if flags['has_ovary']:
        size_items = extract_ovary_sizes(result, debug)
        size_items = deduplicate_values(size_items, debug)
        result_str = ' '.join([item['value'] for item in size_items])
        if debug:
            print(f"\n最终结果(ovary): {result_str}")
        # ovary模式不检查plus_exist，返回0
        return result_str, 1, 0, 0, 0
    
    # 优先级2: Dist
    if flags['has_dist']:
        size_items = extract_dist_sizes(result, debug)
        size_items = deduplicate_values(size_items, debug)
        
        # 【修改点2】检查最后一个数字右侧是否有dist关键词
        plus_exist = 0
        if size_items:
            last_item = size_items[-1]
            # 检查最后一个数字右侧是否没有dist关键词
            if not has_dist_keyword_on_right(last_item, result, debug):
                plus_exist = 1
                if debug:
                    print(f"最后一个数字 {last_item['value']} 右侧没有dist关键词，plus_exist=1")
            else:
                if debug:
                    print(f"最后一个数字 {last_item['value']} 右侧有dist关键词，plus_exist=0")
        
        result_str = ' '.join([item['value'] for item in size_items])
        if debug:
            print(f"\n最终结果(Dist): {result_str}, plus_exist={plus_exist}")
        return result_str, 0, 1, 0, plus_exist
    
    # 优先级3: 超声/维度
    if flags['has_ultrasound'] or flags['has_dimension']:
        size_items = extract_ultrasound_dimension_sizes(result, debug)
        size_items = deduplicate_values(size_items, debug)
        result_str = ' '.join([item['value'] for item in size_items])
        if debug:
            print(f"\n最终结果(超声/维度): {result_str}")
        # 超声/维度模式不检查plus_exist，返回0
        return result_str, 0, 0, 1, 0
    
    # 无任何关键词
    if debug:
        print("\n未检测到任何关键词")
    return "", 0, 0, 0, 0

def get_size(result, debug=False):

    return get_size_tolerant(result, method='hybrid', enable_digit_correction=True, debug=debug)
