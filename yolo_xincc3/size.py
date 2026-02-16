import os
import cv2
import numpy as np
from pathlib import Path
import re
import json
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from decimal import Decimal, ROUND_HALF_UP

def standard_round(value: float, decimals: int = 1) -> float:
    """标准四舍五入函数"""
    multiplier = Decimal(10 ** decimals)
    return float(Decimal(str(value)).quantize(Decimal('1') / multiplier, rounding=ROUND_HALF_UP))

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """计算两个点之间的欧式距离"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_normal_endometrial_size(size: float) -> bool:
    """判断是否为正常的内膜尺寸（0-16mm）"""
    try:
        size_val = float(size)
        return 0.0 <= size_val <= 16.0
    except (ValueError, TypeError):
        return False

def calculate_polygon_area(coords: List[float], img_width: int, img_height: int) -> float:
    """计算多边形面积（平方像素）"""
    if len(coords) < 9:
        return 0.0
    
    points = []
    for i in range(4):
        x = coords[1 + i*2] * img_width
        y = coords[2 + i*2] * img_height
        points.append([x, y])
    
    # 使用鞋带公式计算多边形面积
    area = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def get_follicle_polygon(coords: List[float], img_width: int, img_height: int) -> np.ndarray:
    """从卵泡的4个顶点坐标获取多边形点集"""
    if len(coords) < 9:
        return np.array([])
    
    points = []
    for i in range(4):
        x = int(coords[1 + i*2] * img_width)
        y = int(coords[2 + i*2] * img_height)
        points.append([x, y])
    
    return np.array(points, np.int32)

def get_follicle_center(coords: List[float], img_width: int, img_height: int) -> Tuple[float, float]:
    """从卵泡的4个顶点坐标计算中心点"""
    if len(coords) < 9:
        return (0.0, 0.0)
    
    x_coords = [coords[i] * img_width for i in [1, 3, 5, 7]]
    y_coords = [coords[i] * img_height for i in [2, 4, 6, 8]]
    
    return (sum(x_coords) / 4, sum(y_coords) / 4)

def get_digit_center(coords: List[float], img_width: int, img_height: int) -> Tuple[float, float]:
    """从数字标注的中心坐标获取中心点"""
    if len(coords) < 5:
        return (0.0, 0.0)
    
    return (coords[1] * img_width, coords[2] * img_height)

def read_annotations(file_path: str) -> List[List[float]]:
    """读取标注文件"""
    annotations = []
    if not os.path.exists(file_path):
        return annotations
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        annotations.append([float(x) for x in line.split()])
                    except ValueError:
                        continue
    except Exception as e:
        print(f"读取标注文件时出错 {file_path}: {e}")
    
    return annotations

def find_matching_size_data(img_name: str, index_dict: Dict[str, List[str]]) -> List[str]:
    """查找匹配的尺寸数据"""
    # 直接匹配
    if img_name in index_dict:
        return index_dict[img_name]
    
    # 不带扩展名匹配
    img_name_no_ext = os.path.splitext(img_name)[0]
    if img_name_no_ext in index_dict:
        return index_dict[img_name_no_ext]
    
    # 模糊匹配
    for key in index_dict:
        if img_name_no_ext in key or key in img_name_no_ext:
            return index_dict[key]
    
    # 数字匹配
    img_numbers = re.findall(r'\d+', img_name_no_ext)
    if img_numbers:
        for key in index_dict:
            key_numbers = re.findall(r'\d+', key)
            if key_numbers and any(num in key_numbers for num in img_numbers):
                return index_dict[key]
    
    return []

def draw_follicle_with_label(img: np.ndarray, polygon: np.ndarray, center: Tuple[float, float], 
                             label: str, color: Tuple[int, int, int]) -> None:
    """绘制卵泡边界框和标签"""
    if len(polygon) > 0:
        cv2.polylines(img, [polygon], True, (0, 255, 0), 2)
    
    text_x = max(10, int(center[0]))
    text_y = max(30, int(center[1] - 10))
    
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), 
                 (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
    cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def draw_digit_markers(img: np.ndarray, digit_centers: List[Tuple[float, float]], 
                       digit_labels: List[int]) -> None:
    """绘制数字标注点（调试用）"""
    for center, label in zip(digit_centers, digit_labels):
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)
        cv2.putText(img, str(label), (int(center[0]) + 10, int(center[1])), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def remove_duplicate_digits(digit_centers: List[Tuple[float, float]], 
                            digit_labels: List[int], 
                            threshold: float = 20.0,
                            debug: bool = False) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    移除距离过近的重复数字标注
    
    Args:
        digit_centers: 数字标注中心点列表
        digit_labels: 数字标签列表
        threshold: 距离阈值（像素），小于此距离的相同数字将被视为重复
        debug: 是否打印调试信息
    
    Returns:
        去重后的(digit_centers, digit_labels)
    """
    if len(digit_centers) <= 1:
        return digit_centers, digit_labels
    
    keep_indices = []
    used = set()
    removed_count = 0
    
    for i in range(len(digit_centers)):
        if i in used:
            continue
        keep_indices.append(i)
        
        # 检查是否有相同标签的其他数字在附近
        for j in range(i+1, len(digit_centers)):
            if j not in used and digit_labels[i] == digit_labels[j]:
                dist = calculate_distance(digit_centers[i], digit_centers[j])
                if dist < threshold:
                    used.add(j)  # 标记为重复，不保留
                    removed_count += 1
                    if debug:
                        print(f"  [去重] 移除重复数字{digit_labels[j]} "
                              f"(位置{digit_centers[j]}, 与数字{digit_labels[i]}距离={dist:.1f}px < {threshold}px)")
    
    filtered_centers = [digit_centers[i] for i in keep_indices]
    filtered_labels = [digit_labels[i] for i in keep_indices]
    
    if debug and removed_count > 0:
        print(f"  [去重] 共移除{removed_count}个重复数字标注，保留{len(filtered_labels)}个")
    
    return filtered_centers, filtered_labels


def match_follicles_with_digits_distance_only(follicles: List[List[float]], 
                                              digit_annotations: List[List[float]], 
                                              img_width: int, img_height: int, 
                                              size_data: List[str], 
                                              is_endometrium_branch: bool = False, 
                                              debug: bool = False) -> Tuple:
    """
    【智能匹配算法】优先一对一直接匹配,其次多对一综合匹配
    
    匹配策略:
    1. 一对一情况（卵泡数 = 数字数）：使用匈牙利算法全局最优匹配，确保每个卵泡都有数字
    2. 数量不等情况（卵泡数 ≠ 数字数）：综合考虑距离和面积-尺寸匹配度
    
    多对一匹配权重:
    - 距离因素：数字标注与卵泡中心的距离
    - 面积-尺寸匹配度：卵泡面积与尺寸数值的相关性
    
    优先级: 一对一直接匹配 > 多对一综合匹配
    """
    
    # ========== 数据准备 ==========
    # 计算卵泡数据（包含面积）
    follicle_data = []
    for follicle in follicles:
        center = get_follicle_center(follicle, img_width, img_height)
        polygon = get_follicle_polygon(follicle, img_width, img_height)
        area = calculate_polygon_area(follicle, img_width, img_height)
        follicle_data.append((center, polygon, area))
    
    # 计算数字数据
    digit_centers = []
    digit_labels = []
    for digit in digit_annotations:
        if len(digit) >= 5:
            center = get_digit_center(digit, img_width, img_height)
            digit_centers.append(center)
            digit_labels.append(int(digit[0]) + 1)
    
    # 去除重复的数字标注（距离过近的相同数字）
    if digit_centers:
        if debug:
            print(f"\n{'='*60}")
            print(f"[数字标注去重] 原始数字数量: {len(digit_centers)}")
        digit_centers, digit_labels = remove_duplicate_digits(digit_centers, digit_labels, threshold=20.0, debug=debug)
    
    if debug:
        print(f"\n{'='*60}")
        print(f"[智能匹配算法] 开始匹配")
        print(f"[数据统计] 卵泡数量: {len(follicle_data)}, 数字数量: {len(digit_centers)}")
        if debug and follicle_data:
            print(f"[卵泡面积统计]")
            for j, (center, polygon, area) in enumerate(follicle_data):
                print(f"  卵泡{j+1}: 面积={area:.1f}px²")
    
    # 如果没有数字标注，返回未匹配结果
    if not digit_centers:
        matched_follicles = []
        for j in range(len(follicle_data)):
            matched_follicles.append({
                'index': j, 'digits': [], 'sizes': [], 'size': "未匹配", 'matched': False
            })
        return matched_follicles, [(c, p) for c, p, _ in follicle_data], digit_centers, digit_labels
    
    # 如果没有卵泡
    if not follicle_data:
        return [], [(c, p) for c, p, _ in follicle_data], digit_centers, digit_labels
    
    # ========== 计算距离矩阵 ==========
    distance_matrix = np.zeros((len(follicle_data), len(digit_centers)))
    
    if debug:
        print(f"\n{'='*60}")
        print(f"[距离矩阵计算]")
    
    for j, (center, polygon, area) in enumerate(follicle_data):
        for k, digit_center in enumerate(digit_centers):
            distance_matrix[j, k] = calculate_distance(center, digit_center)
            
            if debug:
                print(f"  卵泡{j+1} ← 数字{digit_labels[k]}: 距离={distance_matrix[j, k]:.1f}px")
    
    # ========== 判断匹配模式 ==========
    is_one_to_one = len(follicle_data) == len(digit_centers)
    
    # 距离阈值：超过此距离的匹配将被拒绝
    DISTANCE_THRESHOLD = 200.0  # 像素
    
    if is_one_to_one and len(follicle_data) > 0:
        # ========== 模式1: 一对一最优匹配（带距离阈值） ==========
        if debug:
            print(f"\n{'='*60}")
            print(f"[模式1: 一对一匹配] 使用匈牙利算法，距离阈值={DISTANCE_THRESHOLD}px")
        
        # 使用匈牙利算法求解最优分配
        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        if debug:
            print(f"[匈牙利算法] 求解完成，总代价={distance_matrix[row_ind, col_ind].sum():.1f}px")
        
        # 构建匹配结果
        matched_follicles = []
        
        for j in range(len(follicle_data)):
            # 找到分配给卵泡j的数字索引
            digit_idx = col_ind[j]
            digit_num = digit_labels[digit_idx]
            match_distance = distance_matrix[j, digit_idx]
            
            # 检查距离是否超过阈值
            if match_distance > DISTANCE_THRESHOLD:
                # 距离过远，拒绝匹配
                matched_follicles.append({
                    'index': j,
                    'digits': [],
                    'sizes': [],
                    'size': "未匹配",
                    'matched': False
                })
                if debug:
                    print(f"[卵泡{j+1}] ✗ 距离过远，拒绝匹配数字{digit_num} "
                          f"(距离={match_distance:.1f}px > 阈值{DISTANCE_THRESHOLD}px)")
                continue
            
            # 距离在阈值内，获取尺寸数据
            idx = digit_num if is_endometrium_branch else digit_num - 1
            size = None
            if 0 <= idx < len(size_data):
                try:
                    size = float(size_data[idx])
                except (ValueError, TypeError):
                    pass
            
            if size is not None:
                matched_follicles.append({
                    'index': j,
                    'digits': [digit_num],
                    'sizes': [size],
                    'size': str(standard_round(size, 1)),
                    'matched': True
                })
                if debug:
                    print(f"[卵泡{j+1}] ✓ 一对一匹配数字{digit_num}, "
                          f"距离={match_distance:.1f}px, 尺寸={standard_round(size, 1)}mm")
            else:
                matched_follicles.append({
                    'index': j,
                    'digits': [digit_num],
                    'sizes': [],
                    'size': "未知",
                    'matched': False
                })
                if debug:
                    print(f"[卵泡{j+1}] ✗ 匹配数字{digit_num}但无尺寸数据 "
                          f"(距离={match_distance:.1f}px)")
        
        if debug:
            print(f"{'='*60}\n")
        
        return matched_follicles, [(c, p) for c, p, _ in follicle_data], digit_centers, digit_labels
    
    else:
        # ========== 模式2: 多对一综合匹配（距离 + 面积-尺寸匹配度） ==========
        if debug:
            print(f"\n{'='*60}")
            print(f"[模式2: 多对一综合匹配] 卵泡数≠数字数，使用距离+面积-尺寸综合评分")
        
        # 获取所有有效的尺寸数据
        valid_sizes = []
        digit_to_size = {}  # {数字索引: 尺寸值}
        
        for k in range(len(digit_centers)):
            idx = digit_labels[k] if is_endometrium_branch else digit_labels[k] - 1
            if 0 <= idx < len(size_data):
                try:
                    size = float(size_data[idx])
                    valid_sizes.append(size)
                    digit_to_size[k] = size
                except (ValueError, TypeError):
                    pass
        
        if debug and valid_sizes:
            print(f"[尺寸数据] 有效尺寸: {valid_sizes}")
            print(f"[面积数据] 卵泡面积: {[f'{area:.1f}' for _, _, area in follicle_data]}")
        
        # ========== 计算综合评分矩阵 ==========
        # 综合评分 = 距离得分 + 面积-尺寸匹配得分
        DISTANCE_WEIGHT = 0.35
        SIZE_AREA_WEIGHT = 0.65
        
        score_matrix = np.zeros((len(follicle_data), len(digit_centers)))
        
        # 1. 归一化距离得分（距离越小，得分越高）
        max_distance = np.max(distance_matrix) if np.max(distance_matrix) > 0 else 1.0
        normalized_distance = distance_matrix / max_distance
        distance_score = 1 - normalized_distance  # 转换为得分（越小距离越高分）
        
        # 2. 计算面积-尺寸匹配得分
        size_area_score = np.zeros((len(follicle_data), len(digit_centers)))
        
        if valid_sizes and len(valid_sizes) > 1:
            # 提取卵泡面积
            follicle_areas = [area for _, _, area in follicle_data]
            
            # 归一化面积和尺寸（避免除零）
            max_area = max(follicle_areas) if max(follicle_areas) > 0 else 1.0
            max_size = max(valid_sizes) if max(valid_sizes) > 0 else 1.0
            
            norm_areas = [a / max_area for a in follicle_areas]
            
            if debug:
                print(f"\n[归一化数据]")
                print(f"  归一化面积: {[f'{a:.3f}' for a in norm_areas]}")
            
            # 为每个卵泡-数字对计算面积-尺寸匹配度
            for j in range(len(follicle_data)):
                for k in range(len(digit_centers)):
                    if k in digit_to_size:
                        size = digit_to_size[k]
                        norm_size = size / max_size
                        
                        # 计算面积与尺寸的相关性得分
                        # 使用负指数距离：差异越小，得分越高
                        diff = abs(norm_areas[j] - norm_size)
                        size_area_score[j, k] = np.exp(-3 * diff)  # 使用负指数衰减
                        
                        if debug:
                            print(f"  卵泡{j+1}(面积={norm_areas[j]:.3f}) ← 数字{digit_labels[k]}(尺寸={norm_size:.3f}): "
                                  f"差异={diff:.3f}, 得分={size_area_score[j, k]:.3f}")
        
        # 3. 综合评分
        score_matrix = DISTANCE_WEIGHT * distance_score + SIZE_AREA_WEIGHT * size_area_score
        
        if debug:
            print(f"\n[综合评分矩阵] (距离{DISTANCE_WEIGHT*100:.0f}% + 面积-尺寸{SIZE_AREA_WEIGHT*100:.0f}%)")
            for j in range(len(follicle_data)):
                scores_str = ", ".join([f"{score_matrix[j, k]:.3f}" for k in range(len(digit_centers))])
                print(f"  卵泡{j+1}: [{scores_str}]")
        
        # ========== 基于综合评分进行匹配 ==========
        # 策略：优先为每个卵泡分配至少一个数字，然后再分配剩余数字
        
        # 第一轮：为每个卵泡分配得分最高的数字（确保每个卵泡都有匹配）
        follicle_to_digits = {}  # {卵泡索引: [数字索引列表]}
        assigned_digits = set()  # 已分配的数字
        
        if debug:
            print(f"\n[第一轮分配] 为每个卵泡分配最优数字")
        
        # 按卵泡遍历，为每个卵泡找到未分配的最高分数字
        for j in range(len(follicle_data)):
            best_digit = -1
            best_score = -1
            
            for k in range(len(digit_centers)):
                if k not in assigned_digits and score_matrix[j, k] > best_score:
                    best_score = score_matrix[j, k]
                    best_digit = k
            
            if best_digit >= 0:
                follicle_to_digits[j] = [best_digit]
                assigned_digits.add(best_digit)
                
                if debug:
                    print(f"  卵泡{j+1} ← 数字{digit_labels[best_digit]}, "
                          f"综合得分={best_score:.3f} "
                          f"(距离={distance_matrix[j, best_digit]:.1f}px, "
                          f"距离得分={distance_score[j, best_digit]:.3f}, "
                          f"面积-尺寸得分={size_area_score[j, best_digit]:.3f})")
        
        # 第二轮：分配剩余的数字（允许多对一）
        if len(assigned_digits) < len(digit_centers):
            if debug:
                print(f"\n[第二轮分配] 分配剩余数字（允许多对一）")
            
            for k in range(len(digit_centers)):
                if k not in assigned_digits:
                    # 找到得分最高的卵泡
                    best_follicle = np.argmax(score_matrix[:, k])
                    follicle_to_digits[best_follicle].append(k)
                    
                    if debug:
                        print(f"  数字{digit_labels[k]} → 卵泡{best_follicle+1}, "
                              f"综合得分={score_matrix[best_follicle, k]:.3f} "
                              f"(距离={distance_matrix[best_follicle, k]:.1f}px, "
                              f"距离得分={distance_score[best_follicle, k]:.3f}, "
                              f"面积-尺寸得分={size_area_score[best_follicle, k]:.3f})")
        
        # 构建最终结果
        if debug:
            print(f"\n[生成最终结果]")
        
        matched_follicles = []
        
        for j in range(len(follicle_data)):
            if j in follicle_to_digits:
                digit_indices = follicle_to_digits[j]
                # 按数字标签排序
                digit_indices.sort(key=lambda k: digit_labels[k])
                digit_numbers = [digit_labels[k] for k in digit_indices]
                sizes = []
                
                # 获取尺寸数据
                for k in digit_indices:
                    idx = digit_labels[k] if is_endometrium_branch else digit_labels[k] - 1
                    if 0 <= idx < len(size_data):
                        try:
                            sizes.append(float(size_data[idx]))
                        except (ValueError, TypeError):
                            pass
                
                # 计算最终尺寸
                if sizes:
                    if len(sizes) > 1:
                        avg_size = standard_round(sum(sizes) / len(sizes), 1)
                    else:
                        avg_size = standard_round(sizes[0], 1)
                    
                    matched_follicles.append({
                        'index': j,
                        'digits': digit_numbers,
                        'sizes': sizes,
                        'size': str(avg_size),
                        'matched': True
                    })
                    
                    if debug:
                        if len(sizes) > 1:
                            print(f"[卵泡{j+1}] ✓ 多对一匹配数字{digit_numbers}, "
                                  f"尺寸{sizes}, 平均={avg_size}mm (基于综合评分)")
                        else:
                            print(f"[卵泡{j+1}] ✓ 匹配数字{digit_numbers[0]}, "
                                  f"尺寸={avg_size}mm (基于综合评分)")
                else:
                    matched_follicles.append({
                        'index': j,
                        'digits': digit_numbers,
                        'sizes': [],
                        'size': "未知",
                        'matched': False
                    })
                    if debug:
                        print(f"[卵泡{j+1}] ✗ 匹配数字{digit_numbers}但无尺寸数据")
            else:
                matched_follicles.append({
                    'index': j,
                    'digits': [],
                    'sizes': [],
                    'size': "未匹配",
                    'matched': False
                })
                if debug:
                    print(f"[卵泡{j+1}] ✗ 未匹配任何数字")
        
        if debug:
            print(f"{'='*60}\n")
        
        return matched_follicles, [(c, p) for c, p, _ in follicle_data], digit_centers, digit_labels


# 为了保持兼容性，保留原函数名作为别名
def match_follicles_with_digits(follicles: List[List[float]], 
                                digit_annotations: List[List[float]], 
                                img_width: int, img_height: int, 
                                size_data: List[str], 
                                is_endometrium_branch: bool = False, 
                                debug: bool = False) -> Tuple:
    """
    兼容性包装函数，调用纯距离匹配版本
    """
    return match_follicles_with_digits_distance_only(
        follicles, digit_annotations, img_width, img_height, 
        size_data, is_endometrium_branch, debug
    )


def extract_region_number(file_path):
    """提取文件名中_region_后的数字用于排序"""
    match = re.search(r'_region_(\d+)', file_path.name)
    return int(match.group(1)) if match else 0

def generate_report_data(results: dict, output_dir: str):
    """根据处理结果生成report_info.json文件"""
    report_data = {
        "report_type": 5,
        "sex": 0,
        "age": 35.0,
        "report_time": [2025, 1, 1],
        "user_time": [-2, -2, -2],
        "period_info": 0,
        "preg_info": 0,
        "report_data": [[-1, -1], 0, [], 0]
    }
    
    endometrial_thicknesses = []
    all_follicle_sizes = []
    total_follicles_detected = 0
    
    for img_result in results["images"]:
        if img_result.get("endometrial_thickness"):
            try:
                endometrial_thicknesses.append(float(img_result["endometrial_thickness"]))
            except (ValueError, TypeError):
                pass
        
        total_follicles_detected += img_result.get("follicles_detected", 0)
        
        for follicle in img_result.get("follicles", []):
            if follicle.get("matched") and follicle.get("final_size"):
                try:
                    all_follicle_sizes.append(float(follicle["final_size"]))
                except (ValueError, TypeError):
                    pass
    
    # 构建内膜尺寸信息
    if not endometrial_thicknesses:
        report_data["report_data"][0] = [-1, -1]
    elif len(endometrial_thicknesses) == 1:
        report_data["report_data"][0] = [endometrial_thicknesses[0], -1]
    else:
        report_data["report_data"][0] = endometrial_thicknesses[:2]
    
    report_data["report_data"][1] = total_follicles_detected
    report_data["report_data"][2] = all_follicle_sizes if all_follicle_sizes else [-1]
    report_data["report_data"][3] = 0 if total_follicles_detected == len(all_follicle_sizes) else 1

    report_data["images"] = results["images"]
    
    # 包装到 info 数组中
    final_output = {
        "info": [report_data],
        "number": 1,
    }
    
    # 保存文件
    report_data_path = os.path.join(output_dir, "report_info.json")
    try:
        with open(report_data_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        print(f"\n报告数据已保存到: {report_data_path}")
        print("=== 报告数据摘要 ===")
        print(f"内膜厚度: {report_data['report_data'][0]}")
        print(f"检测到的卵泡数量: {report_data['report_data'][1]}")
        print(f"卵泡尺寸: {report_data['report_data'][2]}")
        print(f"数据一致性: {'一致' if report_data['report_data'][3] == 0 else '不一致'}")
        print(f"图片详细信息: 已包含{len(report_data['images'])}张图片的详细数据")
    except Exception as e:
        print(f"保存report_info.json文件时出错: {e}")
    
    return final_output


def process_images(img_dir: str, follicle_dir: str, digit_dir: str, output_dir: str, 
                   index_dict: Dict[str, List[str]], dist: Dict[str, int], 
                   endom: Dict[str, int], ovary: Dict[str, int], 
                   plus: Dict[str, int], json_output_path: str = None):
    """
    主要处理函数（支持三种并行逻辑）
    """
    # 初始化结果
    results = {
        "summary": {"total_images": 0, "processed_images": 0, "images_with_follicles": 0, "images_with_anomalies": 0},
        "images": []
    }
    
    if not os.path.exists(img_dir):
        print(f"错误: 图片目录不存在: {img_dir}")
        return results
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取并排序图片文件
    img_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF']:
        img_files.extend(Path(img_dir).glob(f'*{ext}'))
    
    if not img_files:
        print(f"在目录 {img_dir} 中未找到图片文件")
        return results
    
    img_files.sort(key=extract_region_number)
    results["summary"]["total_images"] = len(img_files)
    
    image_position = {1:'第一行（左）', 2:'第一行（右）', 3:'第二行（左）', 4:'第二行（右）',
                     5:'第三行（左）', 6:'第三行（右）', 7:'第四行（左）', 8:'第四行（右）',
                     9:'第五行（左）', 10:'第五行（右）'}
    
    for i, img_path in enumerate(img_files):
        img_name = img_path.stem
        full_img_name = img_path.name
        output_path = os.path.join(output_dir, full_img_name)
        
        print(f"{'='*50}")
        print(f"处理图片: {full_img_name}, 即{image_position.get(i+1, '未知位置')}侧图片")
        
        # 初始化结果
        current_result = {
            "image_name": full_img_name,
            "image_position": image_position.get(i+1, '未知位置'),
            "status": "normal",
            "follicles_detected": 0,
            "follicles": [],
            "endometrial_thickness": None,
            "size_data": [],
            "message": ""
        }
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图片: {img_path}")
            current_result["status"] = "error"
            current_result["message"] = "无法读取图片"
            results["images"].append(current_result)
            continue
        
        img_height, img_width = img.shape[:2]
        
        # 确定图片类型
        dist_value = dist.get(img_name, 0)
        endom_value = endom.get(img_name, 0)
        ovary_value = ovary.get(img_name, 0)
        plus_value = plus.get(img_name, 0)  #
        
        # 获取尺寸数据
        size_data = find_matching_size_data(full_img_name, index_dict)
        current_result["size_data"] = size_data
        
        # 读取标注（只在需要时读取）
        follicle_annotations = []
        digit_annotations = []
        follicles = []
        
        if ovary_value != 1 or size_data:  # Ovary类且无尺寸时不需要读取
            follicle_file = os.path.join(follicle_dir, f"{img_name}.txt")
            follicle_annotations = read_annotations(follicle_file)
            follicles = [ann for ann in follicle_annotations if len(ann) > 0 and int(ann[0]) == 0]
        
        num_follicles = len(follicles)
        
        # ========== Ovary类图片 ==========
        if ovary_value == 1:
            print(f"[Ovary类图片]")
            current_result["image_type"] = "ovary"
            
            if num_follicles == 0:
                cv2.imwrite(output_path, img)
                print("无卵泡和内膜信息\n")
                current_result["status"] = "no_info"
                current_result["message"] = "无卵泡和内膜信息"
            else:
                if size_data:
                    output_img = img.copy()
                    polygon = get_follicle_polygon(follicles[0], img_width, img_height)
                    center = get_follicle_center(follicles[0], img_width, img_height)
                    # 格式化为一位小数（使用标准四舍五入）
                    formatted_size = str(standard_round(float(size_data[0]), 1))
                    draw_follicle_with_label(output_img, polygon, center, f"1: {formatted_size}", (0, 0, 255))
                    cv2.imwrite(output_path, output_img)
                    
                    print(f"卵泡1尺寸为{formatted_size}\n")
                    current_result["follicles_detected"] = 1
                    current_result["follicles"].append({
                        "follicle_index": 1, 
                        "digit_number": 1, 
                        "final_size": formatted_size,
                        "matched": True
                    })
                    current_result["message"] = f"卵泡1尺寸为{formatted_size}"
                    results["summary"]["images_with_follicles"] += 1
                else:
                    cv2.imwrite(output_path, img)
                    print(f"检测到卵泡但无尺寸信息\n")
                    current_result["status"] = "anomaly"
                    current_result["follicles_detected"] = num_follicles
                    current_result["anomaly_type"] = "ovary_no_size"
                    current_result["message"] = "检测到卵泡但无尺寸信息"
                    results["summary"]["images_with_anomalies"] += 1
            
            results["images"].append(current_result)
            results["summary"]["processed_images"] += 1
            continue
        
        # ========== 24版图片（endom=1） ==========
        if endom_value == 1:
            print(f"[24版图片]")
            current_result["image_type"] = "endom"
            
            if num_follicles == 0:
                cv2.imwrite(output_path, img)
                if size_data:
                    print(f"内膜厚度为{size_data[0]}\n")
                    current_result["endometrial_thickness"] = size_data[0]
                    current_result["message"] = f"内膜厚度为{size_data[0]}"
                else:
                    print("无内膜和卵泡信息\n")
                    current_result["status"] = "no_info"
                    current_result["message"] = "无内膜和卵泡信息"
            else:
                # 读取数字标注
                digit_file = os.path.join(digit_dir, f"{img_name}.txt")
                digit_annotations = read_annotations(digit_file)
                
                # 特殊情况处理
                if (not digit_annotations or len(digit_annotations) == 0) and len(size_data) == 1:
                    print(f"[特殊情况] 无数字标注但有1个尺寸数据，直接匹配给第一个卵泡")
                    output_img = img.copy()
                    
                    polygon = get_follicle_polygon(follicles[0], img_width, img_height)
                    center = get_follicle_center(follicles[0], img_width, img_height)
                    label = f"1: {size_data[0]}"
                    draw_follicle_with_label(output_img, polygon, center, label, (0, 0, 255))
                    cv2.imwrite(output_path, output_img)
                    
                    print(f"卵泡1尺寸为{size_data[0]}\n")
                    current_result["follicles_detected"] = 1
                    current_result["follicles"].append({
                        "follicle_index": 1,
                        "digit_numbers": [1],
                        "sizes": [float(size_data[0])],
                        "final_size": size_data[0],
                        "matched": True
                    })
                    current_result["message"] = f"卵泡1尺寸为{size_data[0]}"
                    results["summary"]["images_with_follicles"] += 1
                else:
                    # 使用纯距离匹配算法
                    matched_follicles, follicle_data, digit_centers, digit_labels = match_follicles_with_digits(
                        follicles, digit_annotations, img_width, img_height, size_data, is_endometrium_branch=True, debug=True
                    )
                    
                    output_img = img.copy()
                    
                    for j, (matched_info, (center, polygon)) in enumerate(zip(matched_follicles, follicle_data)):
                        if matched_info['matched']:
                            label = f"{matched_info['digits']}: {matched_info['size']}"
                            if len(matched_info['digits']) > 1:
                                label += "(avg)"
                            color = (0, 0, 255)
                        else:
                            label = "not matched"
                            color = (0, 165, 255)
                        
                        draw_follicle_with_label(output_img, polygon, center, label, color)
                    
                    draw_digit_markers(output_img, digit_centers, digit_labels)
                    cv2.imwrite(output_path, output_img)
                    
                    matched_with_size = [m for m in matched_follicles if m['matched']]
                    if matched_with_size:
                        matched_with_size.sort(key=lambda x: x['digits'][0])
                        messages = [f"卵泡{m['digits'][0]}尺寸为{m['size']}" for m in matched_with_size]
                        print("，".join(messages) + "\n")
                        
                        for match in matched_with_size:
                            current_result["follicles"].append({
                                "follicle_index": match['index'] + 1,
                                "digit_numbers": match['digits'],
                                "sizes": match['sizes'],
                                "final_size": match['size'],
                                "matched": True
                            })
                        
                        current_result["follicles_detected"] = len(matched_with_size)
                        current_result["message"] = "，".join(messages)
                        results["summary"]["images_with_follicles"] += 1
                    else:
                        print(f"检测到{num_follicles}个卵泡，但未能匹配到尺寸\n")
                        current_result["status"] = "anomaly"
                        current_result["anomaly_type"] = "follicles_no_match"
                        current_result["message"] = f"检测到{num_follicles}个卵泡，但未能匹配到尺寸"
                        results["summary"]["images_with_anomalies"] += 1
            
            results["images"].append(current_result)
            results["summary"]["processed_images"] += 1
            continue
        
        # ========== Dist类图片（dist=1） ==========
        if dist_value == 1:
            print(f"[Dist类图片]")
            current_result["image_type"] = "dist"
            
            if num_follicles == 0:
                cv2.imwrite(output_path, img)
                if not size_data:
                    print("无内膜和卵泡信息\n")
                    current_result["status"] = "no_info"
                    current_result["message"] = "无内膜和卵泡信息"
                elif len(size_data) == 1:
                    print(f"内膜厚度为{size_data[0]}\n")
                    current_result["endometrial_thickness"] = size_data[0]
                    current_result["message"] = f"内膜厚度为{size_data[0]}"
                else:
                    normal_sizes = []
                    abnormal_sizes = []
                    
                    for size in size_data:
                        try:
                            size_val = float(size)
                            if is_normal_endometrial_size(size_val):
                                normal_sizes.append(size)
                            else:
                                abnormal_sizes.append(size)
                        except (ValueError, TypeError):
                            abnormal_sizes.append(size)
                    
                    if len(abnormal_sizes) >= 2 and len(normal_sizes) == 1:
                        print(f"内膜厚度为{normal_sizes[0]}\n")
                        current_result["endometrial_thickness"] = normal_sizes[0]
                        current_result["message"] = f"内膜厚度为{normal_sizes[0]}"
                    else:
                        print("无内膜和卵泡信息（忽略多个尺寸）\n")
                        current_result["status"] = "no_info"
                        current_result["message"] = "无内膜和卵泡信息（忽略多个尺寸）"
            else:
                if not size_data:
                    cv2.imwrite(output_path, img)
                    print(f"检测到{num_follicles}个卵泡，但无对应尺寸信息\n")
                    current_result["status"] = "anomaly"
                    current_result["follicles_detected"] = num_follicles
                    current_result["anomaly_type"] = "follicles_no_sizes"
                    current_result["message"] = f"检测到{num_follicles}个卵泡，但无对应尺寸信息"
                    results["summary"]["images_with_anomalies"] += 1
                else:
                    if (plus_value == 1 and len(size_data) == 2 and num_follicles == 1):
                        try:
                            size1 = float(size_data[0])
                            size2 = float(size_data[1])
                            # 判断两个尺寸是否相差不大（相差小于20%）
                            avg_size_val = (size1 + size2) / 2
                            diff_ratio = abs(size1 - size2) / avg_size_val if avg_size_val > 0 else 1.0
                            
                            if diff_ratio < 0.8:
                                print(f"[特殊情况] plus_exist=1，有2个尺寸({size1}, {size2})相差不大，1个卵泡，平均后匹配")
                                
                                output_img = img.copy()
                                polygon = get_follicle_polygon(follicles[0], img_width, img_height)
                                center = get_follicle_center(follicles[0], img_width, img_height)
                                
                                # 计算平均值并四舍五入
                                avg_size = standard_round(avg_size_val, 1)
                                label = f"1,2: {avg_size}(avg)"
                                draw_follicle_with_label(output_img, polygon, center, label, (0, 0, 255))
                                cv2.imwrite(output_path, output_img)
                                
                                print(f"卵泡1尺寸为{avg_size}（数字1,2平均）\n")
                                current_result["follicles_detected"] = 1
                                current_result["follicles"].append({
                                    "follicle_index": 1,
                                    "digit_numbers": [1, 2],
                                    "sizes": [size1, size2],
                                    "final_size": str(avg_size),
                                    "matched": True,
                                    "note": "plus_exist=1特殊处理：平均两个尺寸"
                                })
                                current_result["message"] = f"卵泡1尺寸为{avg_size}（数字1,2平均）"
                                results["summary"]["images_with_follicles"] += 1
                                
                                results["images"].append(current_result)
                                results["summary"]["processed_images"] += 1
                                continue
                            else:
                                print(f"[特殊情况检查] plus_exist=1但两个尺寸({size1}, {size2})相差较大(差异比={diff_ratio:.2%})，使用常规匹配")
                        except (ValueError, TypeError) as e:
                            print(f"[特殊情况检查] 尺寸转换失败: {e}，使用常规匹配")
                    
                    # 常规匹配流程
                    digit_file = os.path.join(digit_dir, f"{img_name}.txt")
                    digit_annotations = read_annotations(digit_file)
                    
                    if (not digit_annotations or len(digit_annotations) == 0) and len(size_data) == 1:
                        print(f"[特殊情况] 无数字标注但有1个尺寸数据，直接匹配给第一个卵泡")
                        output_img = img.copy()
                        
                        polygon = get_follicle_polygon(follicles[0], img_width, img_height)
                        center = get_follicle_center(follicles[0], img_width, img_height)
                        label = f"1: {size_data[0]}"
                        draw_follicle_with_label(output_img, polygon, center, label, (0, 0, 255))
                        cv2.imwrite(output_path, output_img)
                        
                        print(f"卵泡1尺寸为{size_data[0]}\n")
                        current_result["follicles_detected"] = 1
                        current_result["follicles"].append({
                            "follicle_index": 1,
                            "digit_numbers": [1],
                            "sizes": [float(size_data[0])],
                            "final_size": size_data[0],
                            "matched": True
                        })
                        current_result["message"] = f"卵泡1尺寸为{size_data[0]}"
                        results["summary"]["images_with_follicles"] += 1
                    else:
                        # 使用纯距离匹配算法
                        matched_follicles, follicle_data, digit_centers, digit_labels = match_follicles_with_digits(
                            follicles, digit_annotations, img_width, img_height, size_data, is_endometrium_branch=False, debug=True
                        )
                        
                        output_img = img.copy()
                        
                        for matched_info, (center, polygon) in zip(matched_follicles, follicle_data):
                            if matched_info['matched']:
                                label = f"{matched_info['digits']}: {matched_info['size']}"
                                if len(matched_info['digits']) > 1:
                                    label += "(avg)"
                                color = (0, 0, 255)
                            else:
                                label = "not matched"
                                color = (0, 165, 255)
                            
                            draw_follicle_with_label(output_img, polygon, center, label, color)
                        
                        draw_digit_markers(output_img, digit_centers, digit_labels)
                        cv2.imwrite(output_path, output_img)
                        
                        matched_with_size = [m for m in matched_follicles if m['matched']]
                        if matched_with_size:
                            matched_with_size.sort(key=lambda x: x['digits'][0])
                            messages = [f"卵泡{m['digits'][0]}尺寸为{m['size']}" for m in matched_with_size]
                            print("，".join(messages) + "\n")
                            
                            for match in matched_with_size:
                                current_result["follicles"].append({
                                    "follicle_index": match['index'] + 1,
                                    "digit_numbers": match['digits'],
                                    "sizes": match['sizes'],
                                    "final_size": match['size'],
                                    "matched": True
                                })
                            
                            current_result["follicles_detected"] = len(matched_with_size)
                            current_result["message"] = "，".join(messages)
                            results["summary"]["images_with_follicles"] += 1
                        else:
                            print(f"检测到{num_follicles}个卵泡，但未能匹配到尺寸\n")
                            current_result["status"] = "anomaly"
                            current_result["anomaly_type"] = "follicles_no_match"
                            current_result["message"] = f"检测到{num_follicles}个卵泡，但未能匹配到尺寸"
                            results["summary"]["images_with_anomalies"] += 1
            
            results["images"].append(current_result)
            results["summary"]["processed_images"] += 1
            continue
        
        # ========== 默认情况 ==========
        cv2.imwrite(output_path, img)
        print(f"[未标记类型] 无卵泡和内膜信息\n")
        current_result["status"] = "no_info"
        current_result["message"] = "无卵泡和内膜信息"
        results["images"].append(current_result)
        results["summary"]["processed_images"] += 1
    
    # 保存结果
    if json_output_path is None:
        json_output_path = os.path.join(output_dir, "results.json")
    
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {json_output_path}")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
    
    # 生成报告数据
    generate_report_data(results, os.path.dirname(json_output_path))
    
    return results

def size_matching(input, index_dict, dist, endom, ovary, plus, json_output_path=None):
    """尺寸匹配主函数
    """
    IMAGE_DIR = os.path.join(input, "follicle", 'predict')
    ROT_BBOX_DIR = os.path.join(input, "follicle", 'predict/labels')
    NUM_BBOX_DIR = os.path.join(input, "number", 'predict/labels')
    OUTPUT_DIR = os.path.join(input, "final")

    results = process_images(IMAGE_DIR, ROT_BBOX_DIR, NUM_BBOX_DIR, OUTPUT_DIR, 
                            index_dict, dist, endom, ovary, plus, json_output_path)
    
    return results
