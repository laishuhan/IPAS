# === Python代码文件: evaluator.py ===

import config_eval
# 引入新拆分的规则模块
from medical_logic import adjust_severity_contextually

class PatientEvaluator:
    def __init__(self):
        self.default_severity = config_eval.DEFAULT_SEVERITY_MAP
        self.report_configs = config_eval.REPORT_CONFIGS
        self.score_rules = getattr(config_eval, 'CRITICAL_SCORE_RULES', [])

    def _get_indicator_config(self, report_type, index):
        """
        获取特定指标的配置对象 (权重和严重度覆盖表)
        """
        report_conf_list = self.report_configs.get(report_type)
        if not report_conf_list:
            report_conf_list = self.report_configs.get("default", [])

        # B超特殊处理：Index >= 2 的后续项复用 Index 2
        if report_type in [4, 5] and index >= 2:
            if len(report_conf_list) > 2:
                return report_conf_list[2]

        if index < len(report_conf_list):
            return report_conf_list[index]
        
        return {"weight": 1, "severity_override": {}}

    def _calculate_item_score(self, report_type, idx, status_code, indicator_config, age, sex):
        """
        计算单个指标的得分
        """
        # 1. 初始严重度 (配置表)
        override_map = indicator_config.get("severity_override", {})
        if status_code in override_map:
            base_severity = override_map[status_code]
        else:
            base_severity = self.default_severity.get(status_code, 0)

        # 2. 动态调整严重度 (调用外部规则模块)
        # 注意：这里直接调用导入的函数，不再需要 self
        final_severity = adjust_severity_contextually(
            report_type, 
            idx, 
            status_code, 
            base_severity, 
            age, 
            sex
        )

        # 3. 获取权重
        weight = indicator_config.get("weight", 1)

        # 4. 计算最终得分
        score = final_severity * weight
        
        return score, final_severity, weight
    
    # ==========================================
    # [新增] 辅助判断逻辑：什么是异常？
    # ==========================================
    def _is_abnormal(self, status_code):
        """
        判断状态码是否属于“异常”范畴
        定义：排除 -1(不存在), 0(正常), 9(阴性) 之外的都是异常
        """
        if status_code == -1: return False
        if status_code == 0: return False  # 正常
        if status_code == 9: return False  # 阴性
        return True

    def _apply_critical_score_rules(self, need_info):
        """
        [新增] 检查组合加分规则
        :param need_info: 原始数据结构
        :return: (extra_score, hit_rules_names)
        """
        extra_score = 0
        hit_rules = []

        # 1. 为了快速查找，先将 need_info 转换为字典映射
        # 结构: data_map[report_type] = [code_idx0, code_idx1, ...]
        data_map = {}
        # 跳过 Index 0 (全局信息)
        for sub_report in need_info[1:]:
            r_type = sub_report[0]
            r_codes = sub_report[1]
            data_map[r_type] = r_codes

        # 2. 遍历每一条规则
        for rule in self.score_rules:
            conditions = rule.get("conditions", [])
            is_rule_met = True
            
            # 检查规则内的每一个条件 (AND 关系)
            for cond in conditions:
                target_type = cond["report_type"]
                target_idx = cond["index"]
                target_codes = cond["status_codes"]

                # 检查: 报告是否存在
                if target_type not in data_map:
                    is_rule_met = False
                    break
                
                # 检查: 索引是否越界
                current_codes = data_map[target_type]
                if target_idx >= len(current_codes):
                    is_rule_met = False
                    break
                
                # 检查: 状态码是否匹配
                current_status = current_codes[target_idx]
                if current_status not in target_codes:
                    is_rule_met = False
                    break
            
            # 如果所有条件都满足
            if is_rule_met and len(conditions) > 0:
                extra_score += rule.get("score_add", 0)
                hit_rules.append(f"{rule['name']} (+{rule['score_add']})")

        return extra_score, hit_rules

    def evaluate(self, need_info):
        """
        计算基础加权分 + 组合规则加分
        """
        total_score = 0
        score_details = [] 

        # 1. 基础信息提取
        current_age = -1
        current_sex = -1
        if need_info and len(need_info) > 0:
            global_info = need_info[0]
            if len(global_info) >= 1: current_age = global_info[0]
            if len(global_info) >= 2: current_sex = global_info[1]

        # 2. 计算常规单项得分 (Standard Weighted Score)
        for sub_report in need_info[1:]:
            report_type = sub_report[0]
            abnormal_data_codes = sub_report[1]

            for idx, status_code in enumerate(abnormal_data_codes):
                config = self._get_indicator_config(report_type, idx)
                item_score, used_severity, used_weight = self._calculate_item_score(
                    report_type, idx, status_code, config, current_age, current_sex
                )
                
                if item_score > 0:
                    total_score += item_score
                    score_details.append({
                        "report_type": report_type,
                        "indicator_idx": idx,
                        "status_code": status_code,
                        "severity": used_severity,
                        "weight": used_weight,
                        "item_score": item_score
                    })

        # 3. [新增] 计算规则强制加分 (Rule-Based Extra Score)
        extra_score, hit_rules_desc = self._apply_critical_score_rules(need_info)
        
        # 将加分计入总分
        total_score += extra_score

        return {
            "total_score": total_score,
            "breakdown": score_details,
            "extra_score": extra_score,     # 返回加分数值
            "rule_hits": hit_rules_desc     # 返回触发的规则名称列表
        }

    
    # ==========================================
    # [新方法] 专心负责统计 (Calculate Stats)
    # ==========================================
    def calculate_statistics(self, need_info):
        """
        只计算统计指标 (异常数/总数)
        """
        count_existing = 0
        count_abnormal = 0

        # 跳过 Index 0 (全局信息)，从子报告开始
        for sub_report in need_info[1:]:
            abnormal_data_codes = sub_report[1]

            for status_code in abnormal_data_codes:
                if status_code != -1:
                    count_existing += 1
                    if self._is_abnormal(status_code):
                        count_abnormal += 1
        
        ratio = 0.0
        if count_existing > 0:
            ratio = count_abnormal / count_existing

        return {
            "existing": count_existing,
            "abnormal": count_abnormal,
            "ratio": ratio
        }
