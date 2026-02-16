# === Python代码文件: context_rules.py ===

def adjust_severity_contextually(report_type, idx, status_code, current_severity, age, sex):
    """
    独立出来的医学规则引擎：根据上下文调整严重度
    
    :param report_type: 报告类型ID
    :param idx: 指标索引
    :param status_code: 状态ID
    :param current_severity: 基础严重度
    :param age: 年龄 (-1表示未知)
    :param sex: 性别 (0=女, 1=男, -1=未知)
    :return: 调整后的严重度
    """
    
    # 如果基础严重度已经是0（正常），通常不需要调整（除非有极特殊规则）
    if current_severity == 0:
        return 0

    # =======================================================
    # 规则区块 1: 性别相关修正 (Sex Context)
    # =======================================================
    
    # --- Report 1: 性激素六项 ---
    if report_type == 1:
        # Index 3: 睾酮(T)
        if idx == 3:
            # 女性(0) + 偏高/过高(3,4) -> 严重度加重 (多囊风险)
            if sex == 0 and status_code in [3, 4]: 
                return current_severity + 1
            # 男性(1) + 偏高(3) -> 严重度降低 (通常无害)
            elif sex == 1 and status_code == 3: 
                return max(0, current_severity - 1)

    # =======================================================
    # 规则区块 2: 年龄相关修正 (Age Context)
    # =======================================================
    
    if age != -1: # 只有年龄已知时才判断
        
        # --- Report 2: AMH ---
        if report_type == 2:
            # Index 0: AMH - 状态 1(偏低), 2(过低)
            if idx == 0 and status_code in [1, 2]:
                if age >= 40:
                    # 40岁以上AMH低属于生理性衰退，严重度降低
                    return max(1, current_severity - 1)
                elif age < 30:
                    # 30岁以下AMH低，提示卵巢早衰，严重度增加
                    return current_severity + 1

        # --- Report 26: 血糖 ---
        if report_type == 26:
            # Index 0: 空腹血糖 - 状态 3(偏高)
            if idx == 0 and status_code == 3 and age > 60:
                # 老年人血糖控制标准放宽
                return max(0, current_severity - 1)

    # =======================================================
    # 规则区块 3: 其他复杂逻辑 (可继续在此处扩展)
    # =======================================================
    # if report_type == X and ...

    # 如果没有命中任何规则，返回原值
    return current_severity

# ==========================================
# [更新] 全局分数调整函数
# ==========================================
def adjust_total_score_global(sex, age, period_info, preg_info, abnormal_ratio_percent, total_score):
    """
    根据患者全局属性（性别、年龄、经期、孕产史、异常指标占比）对最终总分进行宏观调整。
    
    :param sex: 性别
    :param age: 年龄
    :param period_info: 经期信息
    :param preg_info: 孕产史
    :param abnormal_ratio_percent: 异常指标占比 (0-100的浮点数) [新增]
    :param total_score: 原始总分
    :return: 调整后的新总分
    """
    
    if total_score <= 0:
        return 0.0

    final_score = float(total_score)

    # 1. 年龄修正 (Age)
    if age != -1:
        if age >= 40:
            final_score *= 1.2
        elif age >= 35:
            final_score *= 1.1

    # 2. 经期信息修正 (Period Info)
    if period_info is not None and period_info != -1:
        # if "不规律" in str(period_info): final_score += 5
        pass 

    # 3. 孕产史修正 (Pregnancy Info)
    if preg_info is not None and preg_info != -1:
        pass

    # ---------------------------------------------------
    # 4. [新增] 异常占比修正 (Systemic Risk)
    # ---------------------------------------------------
    if abnormal_ratio_percent == 100:
        final_score *= 3
    elif abnormal_ratio_percent > 80:
        final_score *= 1.85 
    elif abnormal_ratio_percent > 50:
        final_score *= 1.25 
    elif abnormal_ratio_percent > 30:
        final_score *= 1.15

    elif abnormal_ratio_percent > 0 and final_score < 10:
        # 如果分数很低但有异常项，确保不会被完全忽略（可选微调）
        pass

    return final_score