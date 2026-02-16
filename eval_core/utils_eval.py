def flatten_list(lst):
    result = []
    for x in lst:
        if isinstance(x, list):
            result.extend(x)   # 子列表展开
        else:
            result.append(x)
    return result

def get_need_info_from_all_info(all_info):
    result = []
    result.append(all_info[0])

    for item in all_info[1:]:
        sub_info = [item[0], item[3]]
        result.append(sub_info)

    return result

def calculate_urgency(score, config_list):
    """
    根据总分和配置区间，计算急迫等级
    采用 [a, b) 前闭后开区间逻辑
    """
    
    # 预防性处理：如果分数是负数，且配置从0开始，默认归为Level 1
    # 或者如果分数极高超过了配置范围，归为最后一级
    
    for item in config_list:
        min_val, max_val = item["range"]
        
        # 核心逻辑: 前闭后开 [min, max)
        if min_val <= score < max_val:
            return item
            
    # 如果循环结束还没找到匹配 (理论上不应发生，除非分数是负数且区间没覆盖负数)
    # 这里做一个兜底，返回最高等级或者最低等级
    if score >= config_list[-1]["range"][0]:
        return config_list[-1]
    
    return config_list[0]

def extract_global_user_info(info_list):
    """
    从 info_list 中抽取全局用户信息：
    - period_info / preg_info 默认取第一个 item
    - sex / age 从头遍历，分别取第一个 != -1 的值
    """
    if not info_list:
        return -1, -1, None, None

    first_item = info_list[0]
    user_period_info = first_item.get("period_info")
    user_preg_info = first_item.get("preg_info")

    user_sex = -1
    user_age = -1

    for item in info_list:
        if user_sex == -1:
            sex = item.get("sex", -1)
            if sex != -1:
                user_sex = sex

        if user_age == -1:
            age = item.get("age", -1)
            if age != -1:
                user_age = age

        if user_sex != -1 and user_age != -1:
            break

    return user_sex, user_age, user_period_info, user_preg_info
