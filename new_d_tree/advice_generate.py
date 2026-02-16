# 2026/2/8-2:00
import json
import random
import argparse
import os

# 导入所有指标字典
from advice_rulebase import *

def merge_reports_by_type(report_list: list) -> list:
    """
    合并相同report_type的字典列表
    
    Args:
        report_list: 包含多个报告字典的列表，每个字典格式如下：
            {
                "report_type": int,
                "report_time": [year, month, day],
                "sex": int,
                "age": int,
                "report_data": list,
                "user_name": str,
                "report_unit": list,
                "report_original_data": list,
                "report_original_unit": list,
                "user_time": list,
                "period_info": int,
                "preg_info": int,
                ...
            }
    
    Returns:
        合并后的列表
    """
    if not report_list:
        return []
    
    # 按report_type分组
    grouped = {}
    for report in report_list:
        report_type = report.get("report_type")
        if report_type is None:
            continue
        if report_type not in grouped:
            grouped[report_type] = []
        grouped[report_type].append(report)
    
    # 如果没有相同的report_type，直接返回原列表
    if len(grouped) == len(report_list):
        return report_list
    
    # 对每组进行合并
    result = []
    for report_type, reports in grouped.items():
        if len(reports) == 1:
            # 只有一个报告，直接添加
            result.append(reports[0])
        else:
            # 找到report_time最靠后的字典作为基准
            reports_sorted = sorted(reports, key=lambda x: x.get("report_time", [0, 0, 0]), reverse=True)
            
            # 找到最晚的report_time
            latest_time = reports_sorted[0].get("report_time", [0, 0, 0])
            
            # 找出所有具有最晚report_time的字典
            base_reports = [r for r in reports_sorted if r.get("report_time", [0, 0, 0]) == latest_time]
            
            if len(base_reports) == 1:
                # 只有一个基准字典，保留它
                result.append(base_reports[0])
            else:
                # 多个基准字典，需要合并
                merged = base_reports[0].copy()
                
                # 合并sex：如果基准字典中该参数为0或1，则不做改动，如果基准字典为-1，其余被合并字典为0或1则需要将该参数合并到基准字典中去
                if merged.get("sex") == -1:
                    for report in base_reports[1:]:
                        if report.get("sex") in [0, 1]:
                            merged["sex"] = report.get("sex")
                            break
                
                # 合并age：如果基准字典中该参数为一个非负数字，则不做改动，如果基准字典为-1，其余被合并字典有非负数字，则需要将最大的非负数字合并到基准字典中去
                if merged.get("age") == -1:
                    max_age = -1
                    for report in base_reports:
                        age = report.get("age")
                        if isinstance(age, (int, float)) and age >= 0 and age > max_age:
                            max_age = age
                    if max_age != -1:
                        merged["age"] = max_age
                
                # 合并report_data：该参数为一个列表，如果基准字典中的该列表元素不为-1，则该列表元素以基准字典中的值为准，如果基准字典中的该列表元素为-1，则选择其他字典对应的非-1元素填入
                report_data = merged.get("report_data", [])
                for i in range(len(report_data)):
                    if report_data[i] == -1:
                        for report in base_reports[1:]:
                            other_data = report.get("report_data", [])
                            if i < len(other_data) and other_data[i] != -1:
                                report_data[i] = other_data[i]
                                break
                merged["report_data"] = report_data
                
                # 以下字段以基准字典为准（第一个基准字典）
                # report_time, user_time, period_info, preg_info, user_name, report_unit, report_original_data, report_original_unit
                
                result.append(merged)
    
    return result

def process_reports_data(reports_data: dict) -> dict:
    """
    处理reports_data中的info键值对，使用merge_reports_by_type函数进行合并，生成新的new_reports_data
    
    Args:
        reports_data: 读取json文件得到的字典，包含info键，对应的值是报告字典列表
    
    Returns:
        处理后的新字典，其中info键对应的值为合并后的报告字典列表
    """
    if not isinstance(reports_data, dict):
        return reports_data
    
    # 获取info列表
    info_list = reports_data.get("info", [])
    if not isinstance(info_list, list):
        return reports_data
    
    # 合并info列表
    merged_info = merge_reports_by_type(info_list)
    
    # 生成新的reports_data
    new_reports_data = reports_data.copy()
    new_reports_data["info"] = merged_info
    
    return new_reports_data

class Indicator:
    """
    检测指标类，用于封装和处理检测指标字典
    """
    def __init__(self, indicator_dict: dict, value: any = -1):
        """
        初始化指标类
        :param indicator_dict: 包含指标信息的字典
        """
        self.name: str = indicator_dict.get("name")
        self.unit: str = indicator_dict.get("unit")
        self.size_range: list = indicator_dict.get("size_range", [])
        self.size_category: list = indicator_dict.get("size_category", [])
        self.analysis: list = indicator_dict.get("analysis", [])
        # 报告读取的指标值value
        self.value: any = value
        # xy分别表示计算得到的结果(阶段x, 范围y)
        self.result_xy: list = [-1, -1]
        # 相关信息
        # self.correspongding_report = report
        self.condition: list = []
    def get_result_x(self, result_x: int):
        """
        写入result_x值到result_xy列表中
        """
        self.result_xy[0] = result_x

    def get_result_y(self) -> list:
        """
        已知result_x 计算数值所在的区间result_y
        判定指标值所属区间, 获取conclusion 由数值经过区间范围得到该数值是否正常或者偏大偏小
        """
        result_y = -1
        if self.result_xy[0] == -1: # 不存在范围
            self.result_xy[1] = -1

        else:
            # 获取范围列表
            range = self.size_range[self.result_xy[0]]

            # 若value为字符串，则无需使用范围界定，直接一一比对即可
            if isinstance(self.value, str):
                for i, category in enumerate(range):
                    if self.value == category or category == "Placeholder":
                        result_y = i
                        break

            # 若指标值为数值，对应数值区间
            elif isinstance(self.value, (float, int)):
                if self.value == -1: # -1为特殊处理，无区间
                    result_y = -1
                else:
                    for i, (start, end) in enumerate(range):
                        if start == end:
                            continue  # 空区间跳过
                        elif end == 1000000 and self.value >= start: # 1000000为特殊处理，无穷大
                            result_y = i
                            break
                        elif start <= self.value and self.value <= end:
                            result_y = i
                            break

            # 若指标值value 为列表格式，B超报告，那么result_y也要是list
            elif isinstance(self.value, list):
                result_y = []
                for v in self.value:
                    for i, (start, end) in enumerate(range):
                        if start == end:
                            continue  # 空区间跳过
                        elif end == 1000000 and v >= start: # 1000000为特殊处理，无穷大
                            result_y.append(i)
                            break
                        elif start <= v and v <= end:
                            result_y.append(i)
                            break
                # 如果列表为空，则赋值为[-1,]
                if len(result_y) == 0: # 列表为空
                    result_y = [-1,]
                    
            # 初始值为-1,所以无需再赋值-1
            self.result_xy[1] = result_y
        return self.result_xy

    def output_abnormal_analysis(self) -> str:
        """
        打印所有异常信息（测试用）
        """
        # 如果数值和类别相同，则只输出一遍数值即可（解决阴性阳性等）
        if self.value == self.size_category[self.result_xy[0]][self.result_xy[1]]:
            abnormal_analysis_text = (self.name + ": " + self.size_category[self.result_xy[0]][self.result_xy[1]] + "。" 
                                        + self.analysis[self.result_xy[0]][self.result_xy[1]] + "\n")
        else:
            # print(self.result_xy[0])
            # print(self.result_xy[1])


            abnormal_analysis_text = (self.name + " = " + str(self.value) + self.unit + ": " + self.size_category[self.result_xy[0]][self.result_xy[1]] + "。" 
                                        + self.analysis[self.result_xy[0]][self.result_xy[1]] + "\n")
        return abnormal_analysis_text

    def output_indicator_analysis(self) -> str:
        """
        打印指标结论
        """
        if self.value == self.size_category[self.result_xy[0]][self.result_xy[1]]:
            indicator_analysis_text = (self.name + ": " + self.size_category[self.result_xy[0]][self.result_xy[1]] + "。" + "\n")
        else:
            indicator_analysis_text = (self.name + " = " + str(self.value) + self.unit + ": " + self.size_category[self.result_xy[0]][self.result_xy[1]] + "。" + "\n") 
        return indicator_analysis_text

    def output_character_analysis(self) -> str:
        """
        打印文本结论
        """
        character_analysis_text = (self.name + self.size_category[self.result_xy[0]][self.result_xy[1]] + "。" + self.analysis[self.result_xy[0]][self.result_xy[1]] + "\n") 
        return character_analysis_text

    def get_condition(self):
        """
        获取指标的异常条件
        返回：
            异常条件列表
        """
        if isinstance(self.result_xy[1], list):
            # 处理列表情况（如B超报告中的多个卵泡）
            condition = []
            for category in self.result_xy[1]:
                if category != -1:
                    condition.append(self.name + self.size_category[self.result_xy[0]][category])
            return condition
        else:
            # 处理单个值情况
            if self.result_xy[1] == -1: # 超出范围
                return []
            else:
                # 正常情况下返回对应的异常条件
                return [self.name + self.size_category[self.result_xy[0]][self.result_xy[1]],]

    def print_info(self):
        """
        打印指标信息
        """
        
        print(f"指标名称: {self.name}")
        print(f"单位: {self.unit}")
        print(f"数值: {self.value}")
        print(f"结论坐标: {self.result_xy}")
        print(f"异常条件: {self.condition}")
        print()

class Report:
    """
    检测报告类，用于封装和处理检测报告数据
    """
    def __init__(self, report_dict):
        """
        初始化报告类
        :param report_dict: 包含报告信息的字典
        """
        self.report_type: int = report_dict.get("report_type", 0)
        self.sex: int = report_dict.get("sex", -1)
        self.age: float = report_dict.get("age")
        self.report_time: list = report_dict.get("report_time", [])
        self.user_time: list = report_dict.get("user_time", [])
        self.period_info: int = report_dict.get("period_info")
        self.preg_info: int = report_dict.get("preg_info")
        self.report_data: list = report_dict.get("report_data", [])
        self.images: list[dict] = report_dict.get("images", [])
        # 后续需要增加
        # self.ispregnant = report_dict.get("is_pregnant")    # 是否怀孕
        # self.pregnancy = report_dict.get("pregnancy")    # 孕期第几周
        self.ispregnant: int = 1    # 是否怀孕
        self.pregnancy: int = 1    # 孕期第几周
        # 指标实例化列表
        self.indicators_list: list = []
        # target属性
        self.targets: list = []
        # conditions属性
        self.conditions: list = []

    def pre_process(self):
        """
        预处理报告数据
        """
        # No.1 性激素六项检测
        if self.report_type == report_type_number["sex_hormone"]:
            # 如果女性经期阶段为默认值0，则设置为卵泡期2
            if self.period_info == 0: # 默认值0
                self.period_info = 2
            pass
            
        # No.2 AMH
        if self.report_type == report_type_number["amh"]:
            pass

        # No.3 精子报告
        if self.report_type == report_type_number["sperm_status"]:
            pass

        # No.4 中文B超
        if self.report_type == report_type_number["bUltrasound_Chinese"]:
            # 卵泡数量为-1时，重置为0
            if self.report_data[1] == -1:
                self.report_data[1] = 0


            # 过滤尺寸小于3mm的卵泡
            if self.report_data[1] > 0: # 列表不为空
                self.report_data[2] = [i for i in self.report_data[2] if i >= 3]
                if len(self.report_data[2]) > 0:
                    self.report_data[1] = len(self.report_data[2])
                else:
                    self.report_data[1] = 0
                    self.report_data[2] = [-1, ]

        # No.5 泰国B超
        if self.report_type == report_type_number["bUltrasound_Thailand"]:
            # 过滤尺寸小于3mm的卵泡
            if self.report_data[1] > 0: # 列表不为空
                self.report_data[2] = [i for i in self.report_data[2] if i >= 3]
                if len(self.report_data[2]) > 0:
                    self.report_data[1] = len(self.report_data[2])
                else:
                    self.report_data[1] = 0
                    self.report_data[2] = [-1, ]

        # No.6 免疫五项
        if self.report_type == report_type_number["immuno_five"]:
            pass

        # No.7 凝血功能四项
        if self.report_type == report_type_number["coag_function"]:
            pass

        # No.8 肾功
        if self.report_type == report_type_number["renal_function"]:
            # -1 的性别默认为0
            if self.sex == -1:
                self.sex = 0

        # No.9 血型
        if self.report_type == report_type_number["blood_type"]:
            pass
                    
        # No.10 血常规
        if self.report_type == report_type_number["blood_routine"]:
            pass
                    
        # No.11 衣原体
        if self.report_type == report_type_number["ct_dna"]:
            pass
                    
        # No.12 传染病四项
        if self.report_type == report_type_number["infectious_disease"]:
            pass
                    
        # No.13 优生五项TORCH
        if self.report_type == report_type_number["torch"]:
            pass
                    
        # No.14 支原体
        if self.report_type == report_type_number["mycoplasma"]:
            pass
                    
        # No.15 hcg妊娠诊断报告
        if self.report_type == report_type_number["hcg_pregnancy"]:
            pass
                    
        # No.16 地中海贫血症
        if self.report_type == report_type_number["thalassemia"]:
            pass
                    
        # No.17 贫血四项
        if self.report_type == report_type_number["anemia_four"]:
            pass
                    
        # No.18 肝功五项
        if self.report_type == report_type_number["liver_function"]:
            pass
                    
        # No.19 甲功
        if self.report_type == report_type_number["thyroid_function"]:
            pass
                    
        # No.20 25-羟维生素D
        if self.report_type == report_type_number["preconception_health"]:
            pass
                    
        # No.21 尿常规
        if self.report_type == report_type_number["urine_routine"]:
            pass
                    
        # No.22 核医学
        if self.report_type == report_type_number["nuclear_medicine"]:
            pass
                    
        # No.23 结核感染t细胞检测
        if self.report_type == report_type_number["tb_tcell"]:
            pass
                    
        # No.24 rf分型（类风湿因子）
        if self.report_type == report_type_number["rf_typing"]:
            pass
                    
        # No.25 血脂
        if self.report_type == report_type_number["blood_lipid"]:
            pass
                    
        # No.26 血糖
        if self.report_type == report_type_number["blood_glucose"]:
            pass
                    
        # No.27 同型半胱氨酸报告
        if self.report_type == report_type_number["homocysteine"]:
            pass
                    
        # No.28 tct
        if self.report_type == report_type_number["tct"]:
            pass
                    
        # No.29 Y - 染色体微缺失
        if self.report_type == report_type_number["y_microdeletion"]:
            pass
                    
        # No.30 狼疮
        if self.report_type == report_type_number["lupus"]:
            pass

        # No.31 白带常规
        if self.report_type == report_type_number["leukorrhea_routine"]:
            pass

        # No.32 肿瘤标记物
        if self.report_type == report_type_number["tumor_marker"]:
            pass

        # No.33 淋球菌
        if self.report_type == report_type_number["neisseria_gonorrhoeae_culture"]:
            pass

        # No.34 精子线粒体膜电位
        if self.report_type == report_type_number["membrane_potential"]:
            pass

        # No.35 DNA碎片化指数
        if self.report_type == report_type_number["dna_fragmentation_index"]:
            pass

    # 决策前验证
    @staticmethod
    def val_report_data(report):
        """
        验证报告数据是否有具体信息
        
        参数：
        report: 单份报告的数据字典
        
        返回：
        如果符合要求，返回None；否则返回****指标未找到
        """
        error_messages = []

        # No.1 性激素六项检测
        if report["report_type"] == report_type_number["sex_hormone"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到性激素报告的相关数据\n")
            
        # No.2 AMH
        if report["report_type"] == report_type_number["amh"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到AMH报告的相关数据\n")

        # No.3 精子报告
        if report["report_type"] == report_type_number["sperm_status"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到精子检测报告的相关数据\n")

        # No.4 中文B超
        if report["report_type"] == report_type_number["bUltrasound_Chinese"]:
            if (report["report_data"][0] == [] or report["report_data"][0] == [-1, -1] ) and report["report_data"][1] == 0:
                error_messages.append(f"未识别到内膜和卵泡的相关数据\n")

        # No.5 泰国B超
        if report["report_type"] == report_type_number["bUltrasound_Thailand"]:
            if (report["report_data"][0] == [] or report["report_data"][0] == [-1, -1] ) and report["report_data"][1] == 0:
                error_messages.append(f"未识别到内膜和卵泡的相关数据\n")

        # No.6 免疫五项
        if report["report_type"] == report_type_number["immuno_five"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到免疫五项报告的相关数据\n")

        # No.7 凝血功能四项
        if report["report_type"] == report_type_number["coag_function"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到凝血功能报告的相关数据\n")
        
        # No.8 肾功
        if report["report_type"] == report_type_number["renal_function"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到肾功报告的相关数据\n")

        # No.9 血型
        if report["report_type"] == report_type_number["blood_type"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到血型报告的相关数据\n")  
                    
        # No.10 血常规
        if report["report_type"] == report_type_number["blood_routine"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到血常规报告的相关数据\n")
                    
        # No.11 衣原体
        if report["report_type"] == report_type_number["ct_dna"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到衣原体报告的相关数据\n")
                    
        # No.12 传染病四项
        if report["report_type"] == report_type_number["infectious_disease"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到传染病四项报告的相关数据\n")
                    
        # No.13 优生五项TORCH
        if report["report_type"] == report_type_number["torch"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到优生五项报告的相关数据\n")
                    
        # No.14 支原体
        if report["report_type"] == report_type_number["mycoplasma"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到支原体报告的相关数据\n")
                    
        # No.15 hcg妊娠诊断报告
        if report["report_type"] == report_type_number["hcg_pregnancy"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到hcg妊娠诊断报告的相关数据\n")
                    
        # No.16 地中海贫血症
        if report["report_type"] == report_type_number["thalassemia"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到地中海贫血症报告的相关数据\n")

        # No.17 贫血四项
        if report["report_type"] == report_type_number["anemia_four"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到贫血四项报告的相关数据\n")
                    
        # No.18 肝功五项
        if report["report_type"] == report_type_number["liver_function"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到肝功五项报告的相关数据\n")
                    
        # No.19 甲功
        if report["report_type"] == report_type_number["thyroid_function"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到甲功报告的相关数据\n")
                    
        # No.20 25-羟维生素D
        if report["report_type"] == report_type_number["preconception_health"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到25-羟维生素D报告的相关数据\n")
                    
        # No.21 尿常规
        if report["report_type"] == report_type_number["urine_routine"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到尿常规报告的相关数据\n")
                    
        # No.22 核医学
        if report["report_type"] == report_type_number["nuclear_medicine"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到核医学报告的相关数据\n")
                    
        # No.23 结核感染t细胞检测
        if report["report_type"] == report_type_number["tb_tcell"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到结核感染t细胞检测报告的相关数据\n")
                    
        # No.24 rf分型（类风湿因子）
        if report["report_type"] == report_type_number["rf_typing"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到rf分型（类风湿因子）报告的相关数据\n")
                    
        # No.25 血脂
        if report["report_type"] == report_type_number["blood_lipid"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到血脂报告的相关数据\n")
                    
        # No.26 血糖
        if report["report_type"] == report_type_number["blood_glucose"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到血糖报告的相关数据\n")
                    
        # No.27 同型半胱氨酸报告
        if report["report_type"] == report_type_number["homocysteine"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到同型半胱氨酸报告的相关数据\n")
                    
        # No.28 tct
        if report["report_type"] == report_type_number["tct"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到tct报告的相关数据\n")
                    
        # No.29 Y - 染色体微缺失
        if report["report_type"] == report_type_number["y_microdeletion"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到Y - 染色体微缺失报告的相关数据\n")
                    
        # No.30 狼疮
        if report["report_type"] == report_type_number["lupus"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到狼疮报告的相关数据\n")

        # No.31 白带常规
        if report["report_type"] == report_type_number["leukorrhea_routine"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到白带常规报告的相关数据\n")
                    
        # No.32 肿瘤标记物
        if report["report_type"] == report_type_number["tumor_marker"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到肿瘤标记物报告的相关数据\n")    

        # No.33 淋球菌
        if report["report_type"] == report_type_number["neisseria_gonorrhoeae_culture"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到淋球菌报告的相关数据\n")

        # No.34 精子线粒体膜电位
        if report["report_type"] == report_type_number["membrane_potential"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到精子线粒体膜电位报告的相关数据\n")

        # No.35 DNA碎片化指数
        if report["report_type"] == report_type_number["dna_fragmentation_index"]:
            if report["report_data"] == [] or all(x == -1 for x in report["report_data"]):
                error_messages.append(f"未识别到DNA碎片化指数报告的相关数据\n")

        is_pass = (len(error_messages) == 0)
        return is_pass, error_messages

    def generate_indicator_list(self):
        """
        生成该报告中需要的指标实例，并存入列表中
        """
        # No.1 性激素六项检测
        if self.report_type == report_type_number["sex_hormone"]:
            self.indicators_list = [
                Indicator(sex_hormone_LH, self.report_data[0]),
                Indicator(sex_hormone_FSH, self.report_data[1]),
                Indicator(sex_hormone_P4, self.report_data[2]),
                Indicator(sex_hormone_Testosterone, self.report_data[3]),
                Indicator(sex_hormone_E2, self.report_data[4]),
                Indicator(sex_hormone_PRL, self.report_data[5])
            ]
        # No.2 AMH
        if self.report_type == report_type_number["amh"]:
            self.indicators_list = [
                Indicator(amh_AMH, self.report_data[0])
            ]
        # No.3 精子报告
        if self.report_type == report_type_number["sperm_status"]:
            self.indicators_list = [
                Indicator(sperm_status_semenVolume, self.report_data[0]),
                Indicator(sperm_status_liquefactionTime, self.report_data[1]),
                Indicator(sperm_status_phValue, self.report_data[2]),
                Indicator(sperm_status_whiteBloodCellConcentration, self.report_data[3]),
                Indicator(sperm_status_spermConcentration, self.report_data[4]),
                Indicator(sperm_status_totalSpermCount, self.report_data[5]),
                Indicator(sperm_status_totalSpermMotility, self.report_data[6]),
                Indicator(sperm_status_progressiveSpermPercentage, self.report_data[7]),
                Indicator(sperm_status_normalSpermMorphologyRate, self.report_data[8]),
                Indicator(sperm_status_hemoglobinA, self.report_data[9]),
                # Indicator(sperm_status_dnaFragmentationIndex, self.report_data[10])
                Indicator(sperm_status_A, self.report_data[10]),
                Indicator(sperm_status_B, self.report_data[11]),
            ]
        # No.4 中文B超
        if self.report_type == report_type_number["bUltrasound_Chinese"]:
            self.indicators_list = [
                Indicator(bUltrasound_Chinese_endometrial, self.report_data[0]),        # 子宫内膜值，列表
                Indicator(bUltrasound_Chinese_follicle, self.report_data[2])            # 卵泡值，列表
            ]
        # No.5 泰国B超
        if self.report_type == report_type_number["bUltrasound_Thailand"]:
            self.indicators_list = [
                Indicator(bUltrasound_Thailand_endometrial, self.report_data[0]),        # 子宫内膜值，列表
                Indicator(bUltrasound_Thailand_follicle, self.report_data[2])            # 卵泡值，列表
            ]
        # No.6 免疫五项
        if self.report_type == report_type_number["immuno_five"]:
            self.indicators_list = [
                Indicator(immuno_five_IgG, self.report_data[0]),
                Indicator(immuno_five_IgA, self.report_data[1]),
                Indicator(immuno_five_IgM, self.report_data[2]),
                Indicator(immuno_five_C3, self.report_data[3]),
                Indicator(immuno_five_C4, self.report_data[4])
            ]
        # No.7 凝血功能四项
        if self.report_type == report_type_number["coag_function"]:
            self.indicators_list = [
                Indicator(coag_function_PT, self.report_data[0]),
                Indicator(coag_function_PT_R, self.report_data[1]),
                Indicator(coag_function_PT1, self.report_data[2]),
                Indicator(coag_function_PT_INR, self.report_data[3]),
                Indicator(coag_function_APTT, self.report_data[4]),
                Indicator(coag_function_TT, self.report_data[5]),
                Indicator(coag_function_AT_III, self.report_data[6]),
                Indicator(coag_function_D_D, self.report_data[7]),
                Indicator(coag_function_FIB, self.report_data[8]),
                Indicator(coag_function_INR, self.report_data[9])
            ]
        # No.8 肾功
        if self.report_type == report_type_number["renal_function"]:
            self.indicators_list = [
                Indicator(renal_function_Urea, self.report_data[0]),
                Indicator(renal_function_UA, self.report_data[1]),
                Indicator(renal_function_Cr, self.report_data[2]),
                Indicator(renal_function_CYSC, self.report_data[3]),
                Indicator(renal_function_CO2, self.report_data[4]),
                Indicator(renal_function_GLU, self.report_data[5])
            ]
        # No.9 血型
        if self.report_type == report_type_number["blood_type"]:
            self.indicators_list = [
                Indicator(blood_type_ABO, self.report_data[0]),
                Indicator(blood_type_RH, self.report_data[1])
            ]
        # No.10 血常规
        if self.report_type == report_type_number["blood_routine"]:
            self.indicators_list = [
                Indicator(blood_routine_WBC, self.report_data[0]),
                Indicator(blood_routine_RDW_CV, self.report_data[1]),
                Indicator(blood_routine_RBC, self.report_data[2]),
                Indicator(blood_routine_PLT, self.report_data[3]),
                Indicator(blood_routine_PDW, self.report_data[4]),
                Indicator(blood_routine_PCT, self.report_data[5]),
                Indicator(blood_routine_NEU1, self.report_data[6]),
                Indicator(blood_routine_NEU2, self.report_data[7]),
                Indicator(blood_routine_MPV, self.report_data[8]),
                Indicator(blood_routine_MON2, self.report_data[9]),
                Indicator(blood_routine_MON1, self.report_data[10]),
                Indicator(blood_routine_MCV, self.report_data[11]),
                Indicator(blood_routine_MCHC, self.report_data[12]),
                Indicator(blood_routine_MCH, self.report_data[13]),
                Indicator(blood_routine_LYM2, self.report_data[14]),
                Indicator(blood_routine_LYM1, self.report_data[15]),
                Indicator(blood_routine_HGB, self.report_data[16]),
                Indicator(blood_routine_HCT, self.report_data[17]),
                Indicator(blood_routine_EO1, self.report_data[18]),
                Indicator(blood_routine_EO2, self.report_data[19]),
                Indicator(blood_routine_BAS2, self.report_data[20]),
                Indicator(blood_routine_BAS1, self.report_data[21]),
                Indicator(blood_routine_IG1, self.report_data[22]),
                Indicator(blood_routine_IG2, self.report_data[23]),
                Indicator(blood_routine_MCH, self.report_data[24]),
                Indicator(blood_routine_RDW, self.report_data[25]),
                Indicator(blood_routine_NRBC2, self.report_data[26]),
                Indicator(blood_routine_NRBC1, self.report_data[27]),
            ]
        # No.11 衣原体
        if self.report_type == report_type_number["ct_dna"]:
            self.indicators_list = [
                Indicator(ct_dna_CTDNA, self.report_data[0])
            ]
        # No.12 传染病四项
        if self.report_type == report_type_number["infectious_disease"]:
            self.indicators_list = [
                Indicator(infectious_disease_HBsAg, self.report_data[0]),
                Indicator(infectious_disease_HBsAb, self.report_data[1]),
                Indicator(infectious_disease_HBeAg, self.report_data[2]),
                Indicator(infectious_disease_HBeAb, self.report_data[3]),
                Indicator(infectious_disease_HBcAb, self.report_data[4]),
                Indicator(infectious_disease_HCV, self.report_data[5]),
                Indicator(infectious_disease_HIV, self.report_data[6]),
                Indicator(infectious_disease_TPAb, self.report_data[7])
            ]
        # No.13 优生五项TORCH
        if self.report_type == report_type_number["torch"]:
            self.indicators_list = [
                Indicator(torch_CMV_IgM, self.report_data[0]),
                Indicator(torch_CMV_IgG, self.report_data[1]),
                Indicator(torch_TOX_IgM, self.report_data[2]),
                Indicator(torch_TOX_IgG, self.report_data[3]),
                Indicator(torch_RV_IgM, self.report_data[4]),
                Indicator(torch_RV_IgG, self.report_data[5]),
                Indicator(torch_HSV_1_IgM, self.report_data[6]),
                Indicator(torch_HSV_1_IgG, self.report_data[7]),
                Indicator(torch_HSV_2_IgM, self.report_data[8]),
                Indicator(torch_HSV_2_IgG, self.report_data[9]),
                Indicator(torch_B19_IgM, self.report_data[10]),
                Indicator(torch_B19_IgG, self.report_data[11])
            ]
        # No.14 支原体
        if self.report_type == report_type_number["mycoplasma"]:
            self.indicators_list = [
                Indicator(mycoplasma_Uu, self.report_data[0]),
                Indicator(mycoplasma_Mh, self.report_data[1])
            ]
        # No.15 hcg妊娠诊断报告
        if self.report_type == report_type_number["hcg_pregnancy"]:
            self.indicators_list = [
                Indicator(hcg_pregnancy_HCG, self.report_data[0])
            ]
        # No.16 地中海贫血症
        if self.report_type == report_type_number["thalassemia"]:
            self.indicators_list = [
                Indicator(thalassemia_type_a_3_loss, self.report_data[0]),
                Indicator(thalassemia_type_a_3_Nonloss, self.report_data[1]),
                Indicator(thalassemia_type_b_17, self.report_data[2])
            ]
        # No.17 贫血四项
        if self.report_type == report_type_number["anemia_four"]:
            self.indicators_list = [
                Indicator(anemia_four_Fer, self.report_data[0]),
                Indicator(anemia_four_Folate, self.report_data[1]),
                Indicator(anemia_four_VitB12, self.report_data[2]),
                Indicator(anemia_four_TRF, self.report_data[3])
            ]
        # No.18 肝功五项
        if self.report_type == report_type_number["liver_function"]:
            self.indicators_list = [
                Indicator(liver_function_ALB, self.report_data[0]),
                Indicator(liver_function_ALT, self.report_data[1]),
                Indicator(liver_function_AST, self.report_data[2]),
                Indicator(liver_function_AST_ALT, self.report_data[3]),
                Indicator(liver_function_T_BiL, self.report_data[4]),
                Indicator(liver_function_D_BiL, self.report_data[5]),
                Indicator(liver_function_I_BiL, self.report_data[6]),
                Indicator(liver_function_GGT, self.report_data[7]),
                Indicator(liver_function_TP, self.report_data[8]),
                Indicator(liver_function_GLO, self.report_data[9]),
                Indicator(liver_function_A_G, self.report_data[10]),
                Indicator(liver_function_ALP, self.report_data[11]),
                Indicator(liver_function_ChE, self.report_data[12]),
                Indicator(liver_function_AFU, self.report_data[13]),
                Indicator(liver_function_ADA, self.report_data[14]),
                Indicator(liver_function_TBA, self.report_data[15]),              
            ]
        # No.19 甲功
        if self.report_type == report_type_number["thyroid_function"]:
            self.indicators_list = [
                Indicator(thyroid_function_TSH, self.report_data[0]),
                Indicator(thyroid_function_TT3, self.report_data[1]),
                Indicator(thyroid_function_TT4, self.report_data[2]),
                Indicator(thyroid_function_FT3, self.report_data[3]),
                Indicator(thyroid_function_FT4, self.report_data[4]),
                Indicator(thyroid_function_TPOAb, self.report_data[5]),
                Indicator(thyroid_function_TGAb, self.report_data[6]),
                Indicator(thyroid_function_TSHRAb, self.report_data[7])
            ]
        # No.20 25-羟维生素D
        if self.report_type == report_type_number["preconception_health"]:
            self.indicators_list = [
                Indicator(preconception_health_25_OH_VD, self.report_data[0])
            ]
        # No.21 尿常规
        if self.report_type == report_type_number["urine_routine"]:
            self.indicators_list = [
                Indicator(urine_routine_PRO, self.report_data[0]),
                Indicator(urine_routine_GLU, self.report_data[1]),
                Indicator(urine_routine_KET, self.report_data[2]),
                Indicator(urine_routine_BIL, self.report_data[3]),
                Indicator(urine_routine_URO, self.report_data[4]),
                Indicator(urine_routine_NIT, self.report_data[5])
            ]
        # No.22 核医学
        if self.report_type == report_type_number["nuclear_medicine"]:
            self.indicators_list = [
                Indicator(nuclear_medicine_CA199, self.report_data[0])
            ]
        # No.23 结核感染t细胞检测
        if self.report_type == report_type_number["tb_tcell"]:
            self.indicators_list = [
                Indicator(tb_tcell_IFN_N, self.report_data[0]),
                Indicator(tb_tcell_overall, self.report_data[1]),
                Indicator(tb_tcell_IFN_Y_T, self.report_data[2]),
                Indicator(tb_tcell_IFN_V_T_N, self.report_data[3])
            ]
        # No.24 rf分型（类风湿因子）
        if self.report_type == report_type_number["rf_typing"]:
            self.indicators_list = [
                Indicator(rf_typing_IgA, self.report_data[0]),
                Indicator(rf_typing_IgG, self.report_data[1]),
                Indicator(rf_typing_IgM, self.report_data[2])
            ]
        # No.25 血脂
        if self.report_type == report_type_number["blood_lipid"]:
            self.indicators_list = [
                Indicator(blood_lipid_TC, self.report_data[0]),
                Indicator(blood_lipid_TG, self.report_data[1]),
                Indicator(blood_lipid_LDL_C, self.report_data[2]),
                Indicator(blood_lipid_HDL_C, self.report_data[3])
            ]
        # No.26 血糖
        if self.report_type == report_type_number["blood_glucose"]:
            self.indicators_list = [
                Indicator(blood_glucose_FPG, self.report_data[0]),
                Indicator(blood_glucose_2hPG, self.report_data[1]),
                Indicator(blood_glucose_HbA1c, self.report_data[2])
            ]
        # No.27 同型半胱氨酸报告
        if self.report_type == report_type_number["homocysteine"]:
            self.indicators_list = [
                Indicator(homocysteine_HCY, self.report_data[0])
            ]
        # No.28 tct
        if self.report_type == report_type_number["tct"]:
            self.indicators_list = [
                # Indicator(tct_NILM, self.report_data[0]),
                # Indicator(tct_inflammatory_cell_changes, self.report_data[1]),
                # Indicator(tct_ASC_US, self.report_data[2]),
                # Indicator(tct_LSIL, self.report_data[3]),
                # Indicator(tct_HSIL, self.report_data[4]),
                # Indicator(tct_squamous_cell_carcinoma, self.report_data[5])
                Indicator(tct_summary, self.report_data[0])
            ]
        # No.29 Y - 染色体微缺失
        if self.report_type == report_type_number["y_microdeletion"]:
            self.indicators_list = [
                Indicator(y_microdeletion_sY84, self.report_data[0]),
                Indicator(y_microdeletion_sY86, self.report_data[1]),
                Indicator(y_microdeletion_sY127, self.report_data[2]),
                Indicator(y_microdeletion_sY134, self.report_data[3]),
                Indicator(y_microdeletion_sY254, self.report_data[4]),
                Indicator(y_microdeletion_sY255, self.report_data[5])
            ]
        # No.30 狼疮
        if self.report_type == report_type_number["lupus"]:
            self.indicators_list = [
                Indicator(lupus_LA1, self.report_data[0]),
                Indicator(lupus_LA2, self.report_data[1]),
                Indicator(lupus_LA1_LA2, self.report_data[2])
            ]
        # No.31 白带常规
        if self.report_type == report_type_number["leukorrhea_routine"]:
            self.indicators_list = [
                Indicator(leukorrhea_routine_Cleanliness, self.report_data[0]),
                Indicator(leukorrhea_routine_WBC, self.report_data[1]),
                Indicator(leukorrhea_routine_RBC, self.report_data[2]),
                Indicator(leukorrhea_routine_TV, self.report_data[3]),
                Indicator(leukorrhea_routine_FV, self.report_data[4]),
                Indicator(leukorrhea_routine_BV, self.report_data[5]),
                Indicator(leukorrhea_routine_pH, self.report_data[6]),
                Indicator(leukorrhea_routine_Clue_Cells, self.report_data[7]),
                Indicator(leukorrhea_routine_H2O2, self.report_data[8]),
                Indicator(leukorrhea_routine_GUS, self.report_data[9]),
                Indicator(leukorrhea_routine_SNA, self.report_data[10]),
                Indicator(leukorrhea_routine_NAG, self.report_data[11]),
                Indicator(leukorrhea_routine_LE, self.report_data[12])
            ]

        # No.32 肿瘤标记物
        if self.report_type == report_type_number["tumor_marker"]:
            self.indicators_list = [
                Indicator(tumor_marker_CA125, self.report_data[0]),
                Indicator(tumor_marker_AFP, self.report_data[1]),
                Indicator(tumor_marker_CEA, self.report_data[2])
            ]
        # No.33 淋球菌
        if self.report_type == report_type_number["neisseria_gonorrhoeae_culture"]:
            self.indicators_list = [
                Indicator(neisseria_gonorrhoeae_culture_gonorrhea, self.report_data[0]),

            ]

        # No.34 精子线粒体膜电位
        if self.report_type == report_type_number["membrane_potential"]:
            self.indicators_list = [
                Indicator(membrane_potential_MMP, self.report_data[0]),

            ]

        # No.35 DNA碎片化指数
        if self.report_type == report_type_number["dna_fragmentation_index"]:
            self.indicators_list = [
                Indicator(dna_fragmentation_index_DFI, self.report_data[0]),

            ]

        return self.indicators_list
    def get_list_of_result_xy(self):
        """
        根据相关条件判断result_x的值（阶段）

        返回：
        list_result_x: 存储该报告所有指标的result_x值(列表)
        """
        list_result_x = []

        ############ 先得到整个报告的result_x值列表
        # No.1 性激素六项检测
        if self.report_type == report_type_number["sex_hormone"]:
            if self.sex == 1:
                list_result_x = [0, 0, 0, 0, 0, 0]
            else:
                if (self.period_info in [2, 3, 4, 5]) and (self.preg_info in [0, 1, 2, 3]):
                    if self.period_info == 2:    # 卵泡期
                        list_result_x = [1, 1, 1, 1, 1, 1]
                    elif self.period_info == 3:    # 排卵期
                        list_result_x = [2, 2, 2, 2, 2, 2]
                    elif self.period_info == 4:    #  黄体期
                        list_result_x = [3, 3, 3, 3, 3, 3]
                    elif self.period_info == 5:    #  绝经期
                        list_result_x = [4, 4, 4, 4, 4, 4]
                    # 孕酮与孕期阶段的关系
                    if self.preg_info != 0:
                        list_result_x[2] = 4 + self.preg_info
                else:
                    list_result_x = [-1, -1, -1, -1, -1, -1]

        # No.2 AMH
        if self.report_type == report_type_number["amh"]:
            if self.sex == 1:
                list_result_x = [0,]
            else:
            # 女性（根据年龄确定阶段信息）
                if self.age <= 24:
                    list_result_x = [1]  # 20岁以下及20-24岁
                elif self.age <= 29:
                    list_result_x = [2]  # 25-29岁
                elif self.age <= 34:
                    list_result_x = [3]  # 30-34岁
                elif self.age <= 39:
                    list_result_x = [4]  # 35-39岁
                elif self.age <= 44:
                    list_result_x = [5]  # 40-44岁
                else:
                    list_result_x = [6]  # 45岁以上
            
        # No.3 精子报告
        if self.report_type == report_type_number["sperm_status"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]

        # No.4 中文B超
        if self.report_type == report_type_number["bUltrasound_Chinese"]:
            # 并无阶段约束
            list_result_x = [0, 0]

        # No.5 泰国B超
        if self.report_type == report_type_number["bUltrasound_Thailand"]:
            # 并无阶段约束
            list_result_x = [0, 0]

        # No.6 免疫五项
        if self.report_type == report_type_number["immuno_five"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0]

        # No.7 凝血功能四项
        if self.report_type == report_type_number["coag_function"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # No.8 肾功
        if self.report_type == report_type_number["renal_function"]:
            if self.sex == 1:
                list_result_x = [0, 0, 0, 0, 0, 0,]
            elif self.sex == 0:
                list_result_x = [0, 1, 1, 0, 0, 0,]

        # No.9 血型
        if self.report_type == report_type_number["blood_type"]:
            # 并无阶段约束
            list_result_x = [0, 0]

        # No.10 血常规
        if self.report_type == report_type_number["blood_routine"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # No.11 衣原体
        if self.report_type == report_type_number["ct_dna"]:
            # 并无阶段约束
            list_result_x = [0,]

        # No.12 传染病四项
        if self.report_type == report_type_number["infectious_disease"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0]

        # No.13 优生五项TORCH
        if self.report_type == report_type_number["torch"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # No.14 支原体
        if self.report_type == report_type_number["mycoplasma"]:
            # 并无阶段约束
            list_result_x = [0, 0]

        # No.15 hcg妊娠诊断报告
        if self.report_type == report_type_number["hcg_pregnancy"]:
            # 根据性别，是否怀孕, 孕期
            if self.sex == 1:
                list_result_x = [12,]
            else:
                if self.ispregnant == 0:
                    list_result_x = [11,]
                else:
                    if self.pregnancy >= 0 and self.pregnancy <= 18:
                        if self.pregnancy == 1:
                            list_result_x = [0,]
                        elif self.pregnancy == 2:
                            list_result_x = [1,]
                        elif self.pregnancy == 3:
                            list_result_x = [2,]
                        elif self.pregnancy == 4:
                            list_result_x = [3,]
                        elif self.pregnancy == 5:
                            list_result_x = [4,]
                        elif self.pregnancy == 6:
                            list_result_x = [5,]
                        elif self.pregnancy == 7:
                            list_result_x = [6,]
                        elif self.pregnancy == 8:
                            list_result_x = [7,]
                        elif self.pregnancy >= 9 and self.pregnancy <= 12:
                            list_result_x = [8,]
                        elif self.pregnancy >= 13 and self.pregnancy <= 16:
                            list_result_x = [9,]
                        elif self.pregnancy >= 17 and self.pregnancy <= 18:
                            list_result_x = [10,]
                    else:
                        list_result_x = [-1,]
            
        # No.16 地中海贫血症
        if self.report_type == report_type_number["thalassemia"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0]

        # No.17 贫血四项
        if self.report_type == report_type_number["anemia_four"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0]

        # No.18 肝功五项
        if self.report_type == report_type_number["liver_function"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # No.19 甲功
        if self.report_type == report_type_number["thyroid_function"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0]
            
        # No.20 25-羟维生素D
        if self.report_type == report_type_number["preconception_health"]:
            # 并无阶段约束
            list_result_x = [0, ]
            
        # No.21 尿常规
        if self.report_type == report_type_number["urine_routine"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0]
            
        # No.22 核医学
        if self.report_type == report_type_number["nuclear_medicine"]:
            # 并无阶段约束
            list_result_x = [0, ]

        # No.23 结核感染t细胞检测
        if self.report_type == report_type_number["tb_tcell"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0]
            
        # No.24 rf分型（类风湿因子）
        if self.report_type == report_type_number["rf_typing"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0]
            
        # No.25 血脂
        if self.report_type == report_type_number["blood_lipid"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0]
            
        # No.26 血糖
        if self.report_type == report_type_number["blood_glucose"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0]
            
        # No.27 同型半胱氨酸报告
        if self.report_type == report_type_number["homocysteine"]:
            # 并无阶段约束
            list_result_x = [0, ]
            
        # No.28 tct
        if self.report_type == report_type_number["tct"]:
            # 并无阶段约束
            list_result_x = [0,]
            
        # No.29 Y - 染色体微缺失
        if self.report_type == report_type_number["y_microdeletion"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0]
            
        # No.30 狼疮
        if self.report_type == report_type_number["lupus"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0]

        # No.31 白带常规
        if self.report_type == report_type_number["leukorrhea_routine"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]

        # No.32 肿瘤标记物
        if self.report_type == report_type_number["tumor_marker"]:
            # 并无阶段约束
            list_result_x = [0, 0, 0]

        # No.33 淋球菌
        if self.report_type == report_type_number["neisseria_gonorrhoeae_culture"]:
            # 并无阶段约束
            list_result_x = [0,]

        # No.34 精子线粒体膜电位
        if self.report_type == report_type_number["membrane_potential"]:
            # 并无阶段约束
            list_result_x = [0,]

        # No.35 DNA碎片化指数
        if self.report_type == report_type_number["dna_fragmentation_index"]:
            # 并无阶段约束
            list_result_x = [0,]

        ############ 将result_x，result_y值写入各自指标实例中
        for n, indicator in enumerate(self.indicators_list):
            indicator.get_result_x(list_result_x[n])
            indicator.get_result_y()

        return list_result_x
    def get_3_analysis(self):
        """
        返回三种不同结论：abnormal_analysis_text、indicator_analysis_text、character_analysis_text
        """
        # 初始化三个文本结论
        abnormal_analysis_text = ""
        indicator_analysis_text = ""
        character_analysis_text = ""

        # No.1 性激素六项检测
        if self.report_type == report_type_number["sex_hormone"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
            
        # No.2 AMH
        if self.report_type == report_type_number["amh"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()


        # No.3 精子报告
        if self.report_type == report_type_number["sperm_status"]:
            # 其他指标简略带过
            normal_text_list_temp = []           # 存放正常指标合集
            abnormal_text_list_temp = []         # 存放不正常指标合集

            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    #  重点关注6，7，8的指标结果
                    if i in [6, 7, 8]:
                        abnormal_analysis_text += indicator.output_abnormal_analysis()
                        indicator_analysis_text += indicator.output_indicator_analysis()
                        character_analysis_text += indicator.output_character_analysis()
                    # 针对10，11只输出data内容
                    elif i == 10 or i == 11:
                        abnormal_analysis_text += (indicator.name + ": " + str(indicator.value) + indicator.unit + "\n")
                        indicator_analysis_text += (indicator.name + ": " + str(indicator.value) + indicator.unit + "\n")
                        character_analysis_text += ""
                    else:
                        if indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1]] in ["正常", "阴性"]:
                            normal_text_list_temp.append(indicator.name)
                        else:
                            abnormal_text_list_temp.append(indicator.name + indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1]])
            # 处理正常指标列表
            normal_text = '，'.join(normal_text_list_temp) + '均正常。' if normal_text_list_temp else ''
            # 处理异常指标列表
            abnormal_text = '，'.join(abnormal_text_list_temp) + '。' if abnormal_text_list_temp else ''
            abnormal_analysis_text += abnormal_text + normal_text
            indicator_analysis_text += abnormal_text + normal_text
            character_analysis_text += ''

        # No.4 中文B超
        if self.report_type == report_type_number["bUltrasound_Chinese"]:
            # 双子宫结论
            analysis_double_endometrial = "正常女性为单子宫，该报告显示双子宫参数，可能存在子宫畸形，建议进一步检查。"
            
            # 判断符号
            sign1 = {
                -1: "小于",
                0: "为",
                1: "大于"
            }

            ### 初始化
            part_1_indicator_analysis_text = ""          # 内膜部分指标数值
            part_1_character_analysis_text = ""          # 内膜部分结论文本
            part_1_abnormal_analysis_text = ""          # 内膜部分

            part_2_abnormal_analysis_text = ""          # 卵泡数量

            part_3_abnormal_analysis_text = ""          # 卵泡尺寸
            part_3_indicator_analysis_text = ""          # 卵泡部分指标数值
            part_3_character_analysis_text = ""          # 卵泡部分结论文本

            for i, indicator in enumerate(self.indicators_list):
                # 内膜部分
                if i == 0:
                    # 双内膜，直接给出双子宫结论
                    if indicator.value[1] != -1:
                        part_1_abnormal_analysis_text = analysis_double_endometrial + "\n"
                        part_1_indicator_analysis_text = analysis_double_endometrial + "\n"
                    else:
                        if indicator.result_xy[0] != -1 and indicator.result_xy[1][0] != -1:
                        # temp_list = indicator.value
                        # indicator.value = indicator.value[0]
                        # abnormal_analysis_text += indicator.output_abnormal()
                        # indicator.value = temp_list
                            part_1_abnormal_analysis_text += (indicator.name + " = " + str(indicator.value[0]) + indicator.unit + ": " + 
                                                                indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + 
                                                                indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]] + "\n")
                            part_1_indicator_analysis_text += (indicator.name + " = " + str(indicator.value[0]) + indicator.unit + ": " + 
                                                                indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + "\n")
                            part_1_character_analysis_text += (indicator.name + indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + 
                                                                indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]] + "\n")                          
                # 卵泡部分
                elif i == 1:
                    # 先输出卵泡数量
                    if (self.report_data[1] != -1) and (self.report_data[1] != 0):
                        part_2_abnormal_analysis_text = "检测出卵泡数量为:" + str(self.report_data[1]) + "个。"
                        if indicator.result_xy[0] != -1 and indicator.result_xy[1][0] != -1:
                            #输出卵泡各自的尺寸
                            part_3_abnormal_analysis_text += (indicator.name + sign1[self.report_data[3]] + str(indicator.value[0]) + indicator.unit + ": " + 
                                                                indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + 
                                                                indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]] + "\n")
                            part_3_indicator_analysis_text += (indicator.name + sign1[self.report_data[3]] + str(indicator.value[0]) + indicator.unit + ": " + 
                                                                indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + "\n")
                            part_3_character_analysis_text += (indicator.name + indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + 
                                                                indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]] + "\n")

            # 拼接文本
            abnormal_analysis_text = part_1_abnormal_analysis_text + part_2_abnormal_analysis_text + part_3_abnormal_analysis_text
            indicator_analysis_text = part_1_indicator_analysis_text + part_2_abnormal_analysis_text + part_3_indicator_analysis_text
            character_analysis_text = part_1_character_analysis_text + part_3_character_analysis_text

        # No.5 泰国B超
        if self.report_type == report_type_number["bUltrasound_Thailand"]:
            # 双子宫结论
            analysis_double_endometrial = "正常女性为单子宫，该报告显示双子宫参数，可能存在子宫畸形，建议进一步检查。"
            
            # 判断符号
            sign1 = {
                -1: "小于",
                0: "为",
                1: "大于"
            }
            ### 初始化
            part_1_indicator_analysis_text = ""          # 内膜部分指标数值
            part_1_character_analysis_text = ""          # 内膜部分结论文本
            part_1_abnormal_analysis_text = ""          # 内膜部分
            part_2_abnormal_analysis_text = ""          # 卵泡数量
            part_3_abnormal_analysis_text = ""          # 卵泡尺寸
            part_4_add_follicle_size_text = ''
            for i, indicator in enumerate(self.indicators_list):
                # 内膜部分
                if i == 0:
                    # 双内膜，直接给出双子宫结论
                    if indicator.value[1] != -1:
                        part_1_abnormal_analysis_text = analysis_double_endometrial + "\n"
                        part_1_indicator_analysis_text = analysis_double_endometrial + "\n"
                    else:
                        if indicator.result_xy[0] != -1 and indicator.result_xy[1][0] != -1:
                        # temp_list = indicator.value
                        # indicator.value = indicator.value[0]
                        # abnormal_analysis_text += indicator.output_abnormal()
                        # indicator.value = temp_list
                            part_1_abnormal_analysis_text += (indicator.name + " = " + str(indicator.value[0]) + indicator.unit + ": " + 
                                                        indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" 
                                                        + indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]] + "\n")
                            part_1_indicator_analysis_text += (indicator.name + " = " + str(indicator.value[0]) + indicator.unit + ": " + 
                                                        indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" + "\n")
                            part_1_character_analysis_text += (indicator.name + indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]] + "。" 
                                                                + indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]] + "\n")                          
                # 卵泡部分
                elif i == 1:
                    # 先输出卵泡数量
                    if (self.report_data[1] != -1) and (self.report_data[1] != 0):
                        part_2_abnormal_analysis_text = "检测出卵泡数量为:" + str(self.report_data[1]) + "个。"

                        # 循环遍历所有卵泡
                        for n, size in enumerate(indicator.value):
                            if indicator.result_xy[1][n] == -1:
                                continue
                            #输出卵泡各自的尺寸
                            part_3_abnormal_analysis_text += ('卵泡' + str(n+1) + sign1[self.report_data[3]] + str(size) + indicator.unit + ": " 
                                                        + indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][n]] + "。" )
                        part_3_abnormal_analysis_text += '\n'
                        # 存储该部分卵泡的尺寸结论编号：偏大，偏小等，用于后续统一给出结论分析
                        category_list = []
                        #输出卵泡总体去重的结论
                        category_list = indicator.result_xy[1]
                        # 过滤掉无效值-1
                        category_list = [c for c in category_list if c != -1]
                        # 去除重复类别
                        set_category_list = sorted(set(category_list))
                        
                        # 遍历输出所有类别的结论
                        for c in set_category_list:
                            part_4_add_follicle_size_text += ("卵泡尺寸" + indicator.size_category[indicator.result_xy[0]][c] + "," + 
                                                            indicator.analysis[indicator.result_xy[0]][c] + '\n')

            # images卵泡信息：测试时使用
            image_text = "\n【卵泡识别详细信息（测试用）】\n"
            result_text = ""

            # 从report中获取images信息
            if isinstance(self.images, list):
                for image_info in self.images:
                    if isinstance(image_info, dict):
                        # 获取位置信息和状态信息
                        position = image_info.get('image_position', '无图片位置')
                        message = image_info.get('message', '无状态信息')
                        result_text += f"{position}:{message}\n"
            
            # 如果有详细信息，则添加标题
            if result_text:
                image_text += result_text

            # 拼接文本
            abnormal_analysis_text = part_1_abnormal_analysis_text + part_2_abnormal_analysis_text + part_3_abnormal_analysis_text + part_4_add_follicle_size_text + image_text
            indicator_analysis_text = part_1_indicator_analysis_text + part_2_abnormal_analysis_text + part_3_abnormal_analysis_text
            character_analysis_text = part_1_character_analysis_text + part_4_add_follicle_size_text

        # No.6 免疫五项
        if self.report_type == report_type_number["immuno_five"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()

        # No.7 凝血功能四项
        if self.report_type == report_type_number["coag_function"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
        # No.8 肾功
        if self.report_type == report_type_number["renal_function"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()

        # No.9 血型
        if self.report_type == report_type_number["blood_type"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.10 血常规
        if self.report_type == report_type_number["blood_routine"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.11 衣原体
        if self.report_type == report_type_number["ct_dna"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.12 传染病四项
        if self.report_type == report_type_number["infectious_disease"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.13 优生五项TORCH
        if self.report_type == report_type_number["torch"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.14 支原体
        if self.report_type == report_type_number["mycoplasma"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.15 hcg妊娠诊断报告
        if self.report_type == report_type_number["hcg_pregnancy"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.16 地中海贫血症
        if self.report_type == report_type_number["thalassemia"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.17 贫血四项
        if self.report_type == report_type_number["anemia_four"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.18 肝功五项
        if self.report_type == report_type_number["liver_function"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.19 甲功
        if self.report_type == report_type_number["thyroid_function"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.20 25-羟维生素D
        if self.report_type == report_type_number["preconception_health"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.21 尿常规
        if self.report_type == report_type_number["urine_routine"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.22 核医学
        if self.report_type == report_type_number["nuclear_medicine"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.23 结核感染t细胞检测
        if self.report_type == report_type_number["tb_tcell"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.24 rf分型（类风湿因子）
        if self.report_type == report_type_number["rf_typing"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.25 血脂
        if self.report_type == report_type_number["blood_lipid"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.26 血糖
        if self.report_type == report_type_number["blood_glucose"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.27 同型半胱氨酸报告
        if self.report_type == report_type_number["homocysteine"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.28 tct
        if self.report_type == report_type_number["tct"]:
            
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += (str(indicator.value) + "\n")
                    indicator_analysis_text += (str(indicator.value) + "\n")
                    character_analysis_text += ""
                    
        # No.29 Y - 染色体微缺失
        if self.report_type == report_type_number["y_microdeletion"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.30 狼疮
        if self.report_type == report_type_number["lupus"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
        
        # No.31 白带常规
        if self.report_type == report_type_number["leukorrhea_routine"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()
                    
        # No.32 肿瘤标记物
        if self.report_type == report_type_number["tumor_marker"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()

        # No.33 淋球菌
        if self.report_type == report_type_number["neisseria_gonorrhoeae_culture"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()

        # No.34 精子线粒体膜电位
        if self.report_type == report_type_number["membrane_potential"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()

        # No.35 DNA碎片化指数
        if self.report_type == report_type_number["dna_fragmentation_index"]:
            # 无特殊规则，正常输出
            for i, indicator in enumerate(self.indicators_list):
                # 指标数值有效，且存在
                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                    abnormal_analysis_text += indicator.output_abnormal_analysis()
                    indicator_analysis_text += indicator.output_indicator_analysis()
                    character_analysis_text += indicator.output_character_analysis()

        return abnormal_analysis_text, indicator_analysis_text, character_analysis_text
    def get_targets(self):
        """
        获得建议的特殊处理标签
        输出：
            targets(list（str)):  标签列表
        """
        # 性别
        if self.sex == 0:
            self.targets.append("女性")
        elif self.sex == 1:
            self.targets.append("男性")

        # 年龄
        if self.age > 35:
            self.targets.append("大于35岁")

        # 精子报告
        if self.report_type == report_type_number["sperm_status"]:
            for i, indicator in enumerate(self.indicators_list):
                if indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1]] in ['过小', '偏小', '偏大', '过大']:
                    self.targets.append("精子其他指标异常")
                    break

        return self.targets
    def get_conditions(self):
        """
        获得指标的异常条件
        返回：
            conditions(list(str)): 异常条件列表
        """
        conditions = []
        # 泰国B超只针对最大尺寸卵泡做condition
        if self.report_type == report_type_number["bUltrasound_Thailand"]:
            # 内膜
            conditions.extend(self.indicators_list[0].get_condition())
            # 最大卵泡
            max_value = max(self.indicators_list[1].value)
            first_max_index = (self.indicators_list[1].value).index(max_value)
            # 最大卵泡尺寸的异常条件
            conditions.extend([self.indicators_list[1].name + self.indicators_list[1].size_category[self.indicators_list[1].result_xy[0]][first_max_index]])
            
        else:
            for indicator in self.indicators_list:
                conditions.extend(indicator.get_condition())
        self.conditions = conditions
        return conditions

    @staticmethod
    def process_single_report(report_dict):
        """
        处理单份报告，返回异常信息和建议信息
        
        参数：
        report_dict: 单份报告的数据字典
        
        返回：
        conclusion_text: 异常信息文本（测试用，包含所有结论信息）
        indicator_analysis_text: 指标结论文本
        character_analysis_text: 文本结论文本
        jianyi_text: 建议信息文本
        
        """
        # 创建报告对象
        report = Report(report_dict)

        # 预处理报告数据
        report.pre_process()

        # 验证report_data是否有具体信息（report_data） 
        is_content_pass, content_error_messages_list  = Report.val_report_data(report_dict) 

        #验证是否包含有效内容
        if is_content_pass:
            # 创建指标对象
            report.generate_indicator_list()

            # 得到各个指标对应result_xy
            report.get_list_of_result_xy()

            # 获取特例条件
            report.get_conditions()
            report.get_targets()
            # 调试输出report
            # report.print_info()
            # 获得各个指标的结论
            abnormal_analysis_text, indicator_analysis_text, character_analysis_text = report.get_3_analysis()

        else:
            # 报告数据无效，返回空字符串
            abnormal_analysis_text = "\n".join([f"{msg}" for msg in content_error_messages_list])
            indicator_analysis_text = abnormal_analysis_text
            character_analysis_text = ""

        # 更新人物对象的属性值
        global person
        person.update_from_report(report)
        person.append_report(report)
        # 调试输出report
        # report.print_info()

        return is_content_pass, abnormal_analysis_text, indicator_analysis_text, character_analysis_text


    @staticmethod
    def check_json_format_1(report_json):
        """
        检查report_info.json文件的总体结构是否符合要求
        
        Args:
            report_json: 从report_info.json文件读取的JSON对象
            
        Returns:
            tuple: (is_valid, error_messages)
                is_valid: 布尔值，表示结构是否符合要求
                error_messages: 字符串列表，包含所有的结构问题错误信息
        """
        error_messages = []
        
        if 'info' not in report_json:
            error_messages.append("JSON格式错误：缺少必要的'info'键")
        # if 'number' not in report_json:
        #     error_messages.append("JSON格式错误：缺少必要的'number'键")
        
        # 检查info的值是否为列表
        if 'info' in report_json and not isinstance(report_json['info'], list):
            error_messages.append("JSON格式错误：'info'的值必须是一个列表")
        
        # 检查number的值是否为整型数字
        # if 'number' in report_json and not isinstance(report_json['number'], int):
        #     error_messages.append("JSON格式错误：'number'的值必须是一个整型数字")
        
        # 检查info内部结构
        # if 'info' in report_json and isinstance(report_json['info'], list) and 'number' in report_json and isinstance(report_json['number'], int):
        if 'info' in report_json and isinstance(report_json['info'], list):
            # 检查info列表中的每个元素是否都是字典
            for i, item in enumerate(report_json['info']):  
                if not isinstance(item, dict):
                    error_messages.append(f"JSON格式错误：'info'列表中的元素必须是字典类型")
            
            # 检查info列表的个数是否和number的数目相同
            # if len(report_json['info']) != report_json['number']:
            #     error_messages.append(f"JSON格式错误：'info'列表的长度({len(report_json['info'])})必须与'number'的值({report_json['number']})相同")

        is_valid = (len(error_messages) == 0)
        return is_valid, error_messages
    @staticmethod
    def check_json_format_2(report_data):
        """
        检查report_info.json文件的总体结构是否符合要求
        
        Args:
            report_data: info内的字典
            
        Returns:
            tuple: (is_valid, error_messages)
                is_valid: 布尔值，表示结构是否符合要求
                error_messages: 字符串列表，包含所有的结构问题错误信息
        """
        error_messages = []


        # 检查当前元素是否为字典
        if not isinstance(report_data, dict):
            error_messages.append(f"JSON格式错误：'info'列表中的元素必须是字典类型")
        
        # 1. 检查report_type
        if 'report_type' in report_data:
            if not isinstance(report_data['report_type'], int):
                error_messages.append(f"INFO格式错误：报告的'report_type'的值必须是整型数")
            elif not (1 <= report_data['report_type'] <= 30):
                error_messages.append(f"INFO格式错误：报告的'report_type'的值必须在1-30之间（包含区间端点）")
        
        # 2. 检查sex
        if 'sex' in report_data:
            if not isinstance(report_data['sex'], int):
                error_messages.append(f"INFO格式错误：个报告的'sex'的值必须是整型数")
            elif report_data['sex'] not in [0, 1, -1]:
                error_messages.append(f"INFO格式错误：报告的'sex'的值必须为0、1或-1")
        
        # 3. 检查age
        if 'age' in report_data:
            if not isinstance(report_data['age'], (int, float)):
                error_messages.append(f"INFO格式错误：报告的'age'的值必须是数字（整型数或浮点数）") 
        
        # 辅助函数：检查时间列表
        def check_time_list(time_list, field_name):
            if not isinstance(time_list, list):
                error_messages.append(f"INFO格式错误：报告的'{field_name}'的值必须是列表类型")
                return
            if len(time_list) != 3:
                error_messages.append(f"INFO格式错误：报告的'{field_name}'列表长度必须为3")
                return
            for j, num in enumerate(time_list):
                if not isinstance(num, int):
                    error_messages.append(f"INFO格式错误：报告的'{field_name}'列表中元素必须是整型数")
            # # 检查是否为无效时间标记[-1,-1,-1]
            # if time_list != [-1, -1, -1]:
            #     # 这里可以添加更严格的日期验证，如月份范围1-12，日期根据月份验证等
            #     # 目前只检查基本格式
            #     pass
        
        # 4. 检查report_time
        if 'report_time' in report_data:
            check_time_list(report_data['report_time'], 'report_time')
        
        # 5. 检查user_time
        if 'user_time' in report_data:
            check_time_list(report_data['user_time'], 'user_time')
        
        # 6. 检查period_info
        if 'period_info' in report_data:
            if not isinstance(report_data['period_info'], int):
                error_messages.append(f"INFO格式错误：报告的'period_info'的值必须是整数")
            elif report_data['period_info'] not in [0, 1, 2, 3, 4, 5]:
                error_messages.append(f"INFO格式错误：报告的'period_info'的值必须为0-5之间的整数（0：无 1：月经期 2：卵泡期 3：排卵期 4：黄体期 5：绝经期）")
        
        # 7. 检查preg_info
        if 'preg_info' in report_data:
            if not isinstance(report_data['preg_info'], int):
                error_messages.append(f"INFO格式错误：报告的'preg_info'的值必须是整数")
            elif report_data['preg_info'] not in [0, 1, 2, 3]:
                error_messages.append(f"INFO格式错误：报告的'preg_info'的值必须为0-3之间的整数（0：无 1：妊娠期 2：哺乳期 3：不明）")

        # 8. 检查report_data格式

        is_valid = (len(error_messages) == 0)
        return is_valid, error_messages

    def print_info(self):
        """
        打印report信息
        """
        print("======== 报告信息 ========")
        print(f"报告类型: {self.report_type}")
        print(f"性别: {self.sex}")
        print(f"年龄: {self.age}")
        print(f"报告时间: {self.report_time}")
        print(f"用户时间: {self.user_time}")
        print(f"月经阶段信息: {self.period_info}")
        print(f"孕期信息: {self.preg_info}")
        print(f"报告数据: {self.report_data}")
        print(f"是否怀孕: {self.ispregnant}")
        print(f"怀孕周数: {self.pregnancy}")
        print(f"指标列表: {self.indicators_list}")
        print(f"targets列表: {self.targets}")
        print(f"conditions列表: {self.conditions}")

        for indicator in self.indicators_list:
            indicator.print_info()
        
class Person:
    """Person类 - 管理个人身份信息属性"""
    
    def __init__(self):
        """空初始化Person对象"""
        # 基本身份信息
        self.sex = -1          # 性别：0=未知，1=男性，2=女性
        self.age = 0.0        # 年龄
        self.period_info = 0  # 经期信息：0=默认值，1=经期，2=卵泡期，3=排卵期，4=黄体期，5=绝经期
        self.preg_info = 0    # 妊娠信息
        
        # 怀孕相关
        self.ispregnant = 0   # 是否怀孕：1=是，0=否
        self.pregnancy = 0    # 孕期第几周

        # 目标人群和条件
        self.targets = []     # 目标人群标签列表（如"女性"、"大于35岁"等）
        self.conditions = []  # 检测异常条件列表

        # Report_gather
        self.report_list = []

    def append_report(self, report):
        """
        将Report实例添加到Person类中
        
        参数:
            report: Report实例，包含要添加的检测报告信息
            
        返回:
            self: 添加成功后的Person实例本身，支持链式调用
        """
        self.report_list.append(report)
    def print_report_list(self):
        """
        打印所有报告列表
        """
        for report in self.report_list:
            report.print_info()

    def update_json(self, json_data):
        """
        更新JSON数据，为每个info字典添加indicator_analysis和character_analysis字段
        
        参数:
            json_data: 包含原始JSON数据的字典
            
        返回:
            dict: 更新后的JSON数据字典
        """
        # 创建新的JSON数据副本
        new_json_data = json_data.copy()
        
        # 检查是否有info字段
        if 'info' in new_json_data and isinstance(new_json_data['info'], list):
            # 遍历每个info字典
            for i, info_dict in enumerate(new_json_data['info']):
                # 初始化indicator_analysis和character_analysis列表
                indicator_analysis = []
                character_analysis = []
                
                # 查找对应的report实例
                report_type = info_dict.get('report_type')
                for report in self.report_list:
                    if report.report_type == report_type:
                        if report_type == 4 or report_type == 5:
                            # 遍历report中的每个indicator
                            for i, indicator in enumerate(report.indicators_list):
                                if i == 0:
                                    # 输出子宫内膜的尺寸和结果
                                    # 是否是双子宫内膜
                                    analysis_double_endometrial = "正常女性为单子宫，该报告显示双子宫参数，可能存在子宫畸形，建议进一步检查。"
                                    if indicator.value[0] == -1 and indicator.value[1] == -1:
                                        indicator_analysis += [-1, -1]
                                        character_analysis += [-1, -1]
                                    elif indicator.value[1] != -1:
                                        indicator_analysis += ["异常", "异常"]
                                        character_analysis += [analysis_double_endometrial, "异常"]
                                    else:
                                        indicator_analysis += ["正常", indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][0]]]
                                        character_analysis += ["正常", indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][0]]]
                                        # indicator_analysis.append("正常")
                                        # character_analysis.append("正常")                
                                elif i == 1:
                                    if indicator.value != [-1] and indicator.value != [0] and indicator.value != []:
                                        # 输出卵泡的尺寸和结果
                                        # 循环遍历所有卵泡
                                        indicator_analysis.append([])
                                        character_analysis.append([])
                                        for n, size in enumerate(indicator.value):
                                            indicator_analysis[2].append(indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][n]])
                                            character_analysis[2].append(indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1][n]])
                                    else:
                                        indicator_analysis.append([-1])
                                        character_analysis.append([-1])
                            # #输出卵泡各自的尺寸
                            # part_3_abnormal_analysis_text += ('卵泡' + str(n+1) + sign1[self.report_data[3]] + str(size) + indicator.unit + ": " 
                            #                             + indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1][n]] + "。" )
                        
                        else:
                            # 遍历report中的每个indicator
                            for indicator in report.indicators_list:
                                # 获取indicator_analysis
                                if indicator.result_xy[0] != -1 and indicator.result_xy[1] != -1:
                                    ind_analysis = indicator.size_category[indicator.result_xy[0]][indicator.result_xy[1]]
                                    char_analysis = indicator.analysis[indicator.result_xy[0]][indicator.result_xy[1]]
                                    indicator_analysis.append(ind_analysis)
                                    character_analysis.append(char_analysis)
                                else:
                                    indicator_analysis.append(-1)
                                    character_analysis.append(-1)
                            break
                
                # 添加到info字典中
                if indicator_analysis:
                    info_dict['indicator_analysis'] = indicator_analysis
                else:
                    info_dict['indicator_analysis'] = info_dict['report_data']
                if character_analysis:
                    info_dict['character_analysis'] = character_analysis
                else:
                    info_dict['character_analysis'] = info_dict['report_data']
        
        return new_json_data
    def update_from_report(self, report):
        """
        从Report实例更新Person类属性
        
        参数:
            report: Report实例，包含要提取的个人信息
            
        返回:
            self: 返回Person实例本身，支持链式调用
        """
        # 更新基本信息
        if hasattr(report, 'sex') and report.sex is not None:
            self.sex = report.sex
            
        if hasattr(report, 'age') and report.age is not None:
            self.age = report.age
            
        if hasattr(report, 'period_info') and report.period_info is not None:
            self.period_info = report.period_info
            
        if hasattr(report, 'preg_info') and report.preg_info is not None:
            self.preg_info = report.preg_info
            
        if hasattr(report, 'ispregnant') and report.ispregnant is not None:
            self.ispregnant = report.ispregnant
            
        if hasattr(report, 'pregnancy') and report.pregnancy is not None:
            self.pregnancy = report.pregnancy
        
        # 更新targets（去重追加）
        if hasattr(report, 'targets') and report.targets:
            for target in report.targets:
                if target not in self.targets:  # 去重
                    self.targets.append(target)
        
        # 更新conditions（去重追加）
        if hasattr(report, 'conditions') and report.conditions:
            for condition in report.conditions:
                if condition not in self.conditions:  # 去重
                    self.conditions.append(condition)
        
        return self
    
    def __str__(self):
        """返回Person对象的字符串表示"""
        return (f"Person(sex={self.sex}, age={self.age}, period_info={self.period_info}, "
                f"preg_info={self.preg_info}, targets={len(self.targets)} items, "
                f"conditions={len(self.conditions)} items, ispregnant={self.ispregnant}, "
                f"pregnancy={self.pregnancy})")
    
    def __repr__(self):
        """返回Person对象的详细表示"""
        return (f"Person(sex={self.sex}, age={self.age}, period_info={self.period_info}, "
                f"preg_info={self.preg_info}, targets={self.targets}, "
                f"conditions={self.conditions}, ispregnant={self.ispregnant}, "
                f"pregnancy={self.pregnancy})")
    @staticmethod
    def fill_sentence(sentence, filler_list):
        """
        根据 sentence 中 {} 的数量从 filler_list 中随机选择对应数量的元素进行填充

        参数:
            sentence (str): 待填充的句子
            filler_list (list): 填充元素列表

        返回:
            str: 填充后的句子   
        """
        num_placeholders = sentence.count("{}")
        if not filler_list or num_placeholders == 0:
            return sentence
        
        # 从 filler_list 中随机选取 num_placeholders 个不重复元素
        selected_items = random.sample(filler_list, min(num_placeholders, len(filler_list)))
        
        # 补足缺失的参数（如果 filler 不够用）
        while len(selected_items) < num_placeholders:
            selected_items.append("")

        return sentence.format(*selected_items)
    def get_adjustment(self):
        """
        得到对应的建议
        """
        # 初始化
        adjustment_text = ""

        # 获取知识库的建议字典
        jianyi_dict = adjustment_dict_summarizing

        # 建议抬头字典
        adjustment_title = {
            "diet_advices": "【一、饮食调整】",
            "workout_advices": "【二、适度运动】",
            "life_advices": "【三、日常生活】",
            "hcp_advices": "【四、保健品服用】",
            "targeted_advices": "【五、异常指标建议】"
        }
        # 循环获取建议
        for advice_types in ["diet_advices", "workout_advices", "life_advices", "hcp_advices", "targeted_advices"]:
            advices = jianyi_dict.get(advice_types, [])
            # 添加抬头
            adjustment_text += adjustment_title[advice_types] + "\n"
            idx = 1
            for advice in advices:
                advice_target = advice.get('target')
                advice_condition = advice.get('condition')
                # 该条advice有约束
                if advice_target or advice_condition:
                    # 检查advice_target中的所有元素是否都包含在self.targets中
                    target_met = all(x in self.targets for x in advice_target) if advice_target else True
                    # 检查advice_condition中的所有元素是否都包含在self.conditions中
                    condition_met = all(x in self.conditions for x in advice_condition) if advice_condition else True
                    
                    if target_met and condition_met:
                        sentence = random.choice(advice["sentence_structure"])
                        # 填充建议
                        filled_sentence = Person.fill_sentence(sentence, advice["filler"])

                        adjustment_text += f"{idx}. " + filled_sentence  + "\n"
                        idx += 1

                # 没有约束，从 sentence_structure 中再随机选取一条建议
                else:
                    sentence = random.choice(advice["sentence_structure"])
                    # 填充建议
                    filled_sentence = Person.fill_sentence(sentence, advice["filler"])

                    adjustment_text += f"{idx}. " + filled_sentence  + "\n"
                    idx += 1
            if idx == 1:
                adjustment_text += "暂无建议\n"
        return adjustment_text

def save_result(abnormal_path, indicator_analysis_path, character_analysis_path, output_jianyi_path, log_file_path, all_abnormal_text, all_indicator_analysis_text, all_character_analysis_text, all_jianyi_text, error_text):
    """
    保存结果
    :param abnormal_path: 异常报告结果文件路径
    :param indicator_analysis_path: 指标数值结论文件路径
    :param character_analysis_path: 指标文本结论文件路径
    :param output_jianyi_path: 建议文件路径
    :param log_file_path: 日志文件路径

    :param all_abnormal_text: 异常报告结果文本
    :param all_indicator_analysis_text: 指标数值结论文本
    :param all_character_analysis_text: 指标文本结论文本
    :param all_jianyi_text: 建议文本
    :param error_text: 错误信息文本
    """

    with open(abnormal_path, 'w', encoding='utf-8') as f:
        f.write(all_abnormal_text)
    
    # 写入指标数值结论文件, indicator的结果用于直接输出未找到
    with open(indicator_analysis_path, 'w', encoding='utf-8') as f:
        f.write(all_indicator_analysis_text)

    # 写入指标文本结论文件
    with open(character_analysis_path, 'w', encoding='utf-8') as f:
        f.write(all_character_analysis_text)

    # 写入建议文件
    with open(output_jianyi_path, 'w', encoding='utf-8') as f:
        f.write(all_jianyi_text)

    # 写入日志文件
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(error_text)


parser = argparse.ArgumentParser(description="Script for generating suggestions from OCR results")
parser.add_argument("--input_path", type=str, required=True, help="Path to task_record.json")
parser.add_argument("--abnormal_path", type=str, required=True, help="Path to report_abnormal.txt")
parser.add_argument("--indicator_analysis_path", type=str, required=True, help="Path to indicator_analysis_text.txt")
parser.add_argument("--character_analysis_path", type=str, required=True, help="Path to character_analysis_text.txt")
parser.add_argument("--output_jianyi_path", type=str, required=True, help="Path to save final suggestions")
args = parser.parse_args()

# 使用传入的参数
input_path = args.input_path
abnormal_path = args.abnormal_path
output_jianyi_path = args.output_jianyi_path
indicator_analysis_path = args.indicator_analysis_path
character_analysis_path = args.character_analysis_path

# 1. 提取input_path所在的文件夹路径（也可替换为其他参数，如output_jianyi_path）
# target_dir = os.path.dirname(input_path)
# 2. 拼接log.txt文件名，生成完整路径
log_file_path = os.path.join(os.path.dirname(input_path), "log.txt")

################################################################
# # 单机调试
# input_path = r"./test_data/test.json"
# abnormal_path = r"./test_data/report_abnormal.txt"
# output_jianyi_path = r"./test_data/report_jianyi.txt"
# indicator_analysis_path = r"./test_data/indicator_analysis_text.txt"
# character_analysis_path = r"./test_data/character_analysis_text.txt"
# log_file_path = r"./test_data/log.txt"

# 初始化Person类对象
person = Person()

if __name__ == '__main__':
    # 数据准备阶段
    # 加载报告信息文件
    with open(input_path, "r", encoding="utf-8") as f:
        reports_data = json.load(f)

    new_reports_data = process_reports_data(reports_data)
    
    # 保存新的json文件(新版json文件保存放在程序结尾)
    # output_json_path = os.path.join(os.path.dirname(input_path), "1_processed_" + os.path.basename(input_path))
    # with open(output_json_path, "w", encoding="utf-8") as f:
    #     json.dump(new_reports_data, f, ensure_ascii=False, indent=2)
    # print(f"新的json文件已保存到: {output_json_path}")

    # 初始化累积的文本变量
    all_abnormal_text = ""
    all_indicator_analysis_text = ""
    all_character_analysis_text = ""
    all_jianyi_text = ""
    error_text = ""
    # log_text = ""
    # all_log_text = ""
    log_text_list = ["程序执行记录:"]
    error_flag = False  # 错误标志，True表明报告内部是否存在错误或空值

    # 文件规则验证与内容验证
    # 检验报告是否符合规定的标准json格式（格式审查1）(外部：info 和 number)
    is_format_pass_1, format_error_messages_list_1 = Report.check_json_format_1(new_reports_data)

    # 格式审查1通过
    if not is_format_pass_1:
        # 格式1审查未通过
        error_text += "\n".join([f"{msg}" for msg in format_error_messages_list_1])
        # 输出异常信息结果文件
        print("JSON文件异常，请检查文件")
        log_text_list.append("【格式1】审查【未通过】")
        error_text += "\n".join([f"{msg}" for msg in log_text_list])
        save_result(abnormal_path, indicator_analysis_path, character_analysis_path, output_jianyi_path, log_file_path, all_abnormal_text, all_indicator_analysis_text, all_character_analysis_text, all_jianyi_text, error_text)

    else:
        # 格式1审查通过，开始格式审查2(内循环)
        log_text_list.append("【格式1】审查【通过】")
        for i, report in enumerate(new_reports_data['info']):
            # 验证报告数据结构
            is_format_pass_2, format_error_messages_list_2 = Report.check_json_format_2(new_reports_data)  

            if not is_format_pass_2:
                # 将错误信息列表转换为字符串
                error_text += "\n".join([f"{msg}" for msg in format_error_messages_list_2])
                error_flag = True

        # 格式审查2未通过
        if error_flag:
            # 记录Log
            log_text_list.append("【格式2】审查【未通过】")
            error_text += "\n".join([f"{msg}" for msg in log_text_list])
            save_result(abnormal_path, indicator_analysis_path, character_analysis_path, output_jianyi_path, log_file_path, all_abnormal_text, all_indicator_analysis_text, all_character_analysis_text, all_jianyi_text, error_text)
        else:
            # 格式审查2通过
            log_text_list.append("【格式2】审查【通过】")
            # 若通过格式验证与内容验证，走进正常流程
            # 循环处理内部info
            adjustment_flag = False
            for i, report in enumerate(new_reports_data['info']):
                # 初始化
                abnormal_text = ""
                indicator_analysis_text = ""
                character_analysis_text = ""
                all_jianyi_text = ""

                # 处理单份报告
                is_content_pass, abnormal_text, indicator_analysis_text, character_analysis_text = Report.process_single_report(report)
                if is_content_pass == True:
                    adjustment_flag = True
                # 如果不是第一份报告，添加两个空行作为分隔
                if i > 0:
                    all_abnormal_text += "\n\n"
                    all_indicator_analysis_text += "\n\n"
                    all_character_analysis_text += "\n\n"

                all_abnormal_text += abnormal_text
                all_indicator_analysis_text += indicator_analysis_text
                all_character_analysis_text += character_analysis_text
            # 判断是否存在结果输出
            str_text = all_indicator_analysis_text

            # 判断报告数据是否存在错误
            if adjustment_flag == True and str_text.strip() != "":
                # 记录Log
                log_text_list.append("【报告内容】审查【通过】")
                # 构建建议文本
                all_jianyi_text = person.get_adjustment()

                # 为需要后续处理的文件写入前缀
                all_jianyi_text = "need_post_process\n" + all_jianyi_text
                all_character_analysis_text = "need_post_process\n" + all_character_analysis_text
            else:
                # 记录Log
                log_text_list.append("【报告内容】审查【未通过】")
                print("JSON文件异常，请检查文件")
            error_text += "\n".join([f"{msg}" for msg in log_text_list])
            # 如果存在图片冗余，提供返回语句：【存在*张非医学报告图片，请仔细核查】
            if new_reports_data['unsolvable_img_number'] > 0:
                all_indicator_analysis_text += f"\n\n存在{new_reports_data['unsolvable_img_number']}张非医学报告图片，请仔细核查。"
                all_abnormal_text += f"\n\n存在{new_reports_data['unsolvable_img_number']}张非医学报告图片，请仔细核查。"
            else:
                pass
            save_result(abnormal_path, indicator_analysis_path, character_analysis_path, output_jianyi_path, log_file_path, all_abnormal_text, all_indicator_analysis_text, all_character_analysis_text, all_jianyi_text, error_text)
            # 打印所有报告列表
            # person.print_report_list()
            
            # 生成带有indicator_analysis和character_analysis的新JSON文件
            new_new_reports_data = person.update_json(new_reports_data)
            
            # 保存更新后的JSON文件
            output_json_path = os.path.join(os.path.dirname(input_path), "processed_" + os.path.basename(input_path))
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(new_new_reports_data, f, ensure_ascii=False, indent=2)
            print(f"新的json文件已保存到: {output_json_path}")


