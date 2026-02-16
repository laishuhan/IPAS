

# 特殊模板特征指纹列表 
# 作用：用于 patch.py 识别报告类型
# 结构: [ [该类型必须包含的关键词列表], ... ]
# 索引 + 1 即为 特殊模版ID

TEMPLATE_FINGERPRINT_LIST = [

    ["精子分类统计","世界卫生组织人类精液检查与处理"], #第0个ID为1
    
]

# =============================
# Extra OCR rules (Unified)
# =============================
# 规则语义统一为：
#   template_ids 条件 AND report_types 条件 同时成立 → 需要 OCR
# 其中：
#   template_ids 为空 = 不限制模板（任何 template_id 都算满足）
#   report_types 为空 = 不限制报告类型（任何 report_type 都算满足）
#
# 注意：不要写 template_ids=[] 且 report_types=[]，那会对所有情况都命中。
NEED_EXTRA_OCR_RULES = [
    
    # 某些报告类型在某些模板下都需要 OCR,若某个列表为空则对应为任何
    # {
    #     "template_ids": [xxx,xxx],
    #     "report_types": [xxx,xxx],
    #     "reason": "xxx"
    # },

    {
        "template_ids": [1],
        "report_types": [],
        "reason": "模板1在纯图片场景下结构不稳定，且左上角内容有概率提取不到。需要OCR文本辅助提取"
    },


]

# 特殊模板关键词映射表
# TEMPLATE_KEYWORD_MAP = {
#    # ID 1 的模版配置 假设此种模板的性激素六项中LH,FSH；精子中精液量，PH值有特殊叫法
#     1: { 
        
#         # === 针对 性激素六项 (ID 1) 的特殊配置 index从0开始 ===
#         1: {
#             0: ["xxxLH"],
#             1: ["xxxFSH"]
#         },

#          # === 针对 精子 (ID 3) 的特殊配置 index从0开始===
#         3: {
#             0: ["xxx精液量"],
#             2: ["xxxPH"] 
#         }
#     },
    
#     # ID 2 的模版配置...
#     2: {
#         # ...
#     }
# }
TEMPLATE_KEYWORD_MAP = {
    9999: { 
        1: {
            0: [""]
        }
    }
    
}

#整个提取模块无法处理的报告类型ID
UNCLASSIFIED_REPORT_ID = 999

#本模型无法处理类型数字，不提取也不组装
UNABLE_TO_PROCESS_REPORT_LIST = [
    5,  #泰国b超
    UNCLASSIFIED_REPORT_ID,  #无法分类报告
]

SPECIAL_EXTRACTOR_LIST = [
    4,   # 中文 B 超
    14,  # 支原体
    28,  # TCT（新增）
    33,  # 淋球菌
]

SPECIAL_EXTRACTOR_MAP = {
    4: "find_b_info_in_text",
    14: "find_mycoplasma_info_in_vision",
    28: "find_tct_info_in_vision",
    33: "find_neisseria_gonorrhoeae_culture_info_in_vision",
}


# 报告类型关键词列表
REPORT_TYPE_LIST = [
                    ["sex_hormone", 1],             # 性激素六项
                    ["amh", 2],                     # AMH
                    ["sperm_status", 3],            # 精子
                    ["ultrasound", 4],              # 中文B超 
                    ["ultrasound_tai", 5],          #泰国b超
                    ["immuno_five", 6],             # 免疫五项
                    ["coag_function", 7],           # 凝血功能
                    ["renal_function", 8],          # 肾功能
                    ["blood_type", 9],              # 血型检测
                    ["blood_routine", 10],          # 血常规
                    ["ct_dna", 11],                 # 衣原体
                    ["infectious_disease", 12],     # 传染病四项
                    ["torch", 13],                  # 优生五项TORCH
                    ["mycoplasma", 14],             # 支原体
                    ["hcg_pregnancy", 15],          # HCG妊娠诊断
                    ["thalassemia", 16],            # 地中海贫血
                    ["anemia_four", 17],            # 贫血四项
                    ["liver_function", 18],         # 肝功五项
                    ["thyroid_function", 19],       # 甲功
                    ["preconception_health", 20],   # 孕前基础健康评估(维D)
                    ["urine_routine", 21],          # 尿常规
                    ["nuclear_medicine", 22],       # 核医学(CA199)
                    ["tb_tcell", 23],               # 结核感染T细胞检测
                    ["rf_typing", 24],              # RF分型(类风湿因子)
                    ["blood_lipid", 25],            # 血脂
                    ["blood_glucose", 26],          # 血糖
                    ["homocysteine", 27],           # 同型半胱氨酸
                    ["tct", 28],                    # TCT(宫颈细胞学)
                    ["y_microdeletion", 29],        # Y染色体微缺失
                    ["lupus", 30],                  # 狼疮(LA)
                    ["leukorrhea_routine_report", 31],     # 白带常规
                    ["tumor_marker_report", 32],            # 肿瘤标志物
                    ["neisseria_gonorrhoeae_culture", 33], #淋球菌 
                    ["membrane_potential", 34], #精子线粒体膜电位检测 
                    ["dna_fragmentation_index", 35], #dna碎片率
                    ["unclassified", UNCLASSIFIED_REPORT_ID], #无法分类报告
]

#对 指标名称 的一些筛除和替换
KEY_NORMALIZE_CONFIG = {
    # 1️⃣ 直接删除的字符（不承载医学语义）
    "remove_chars": [
        # —— 空白类 ——
        " ", "\t", "\n", "\r", "\u3000",

        # —— 装饰 / 标记符 ——
        "★", "☆", "※", "●", "○", "■", "□",
        "▲", "△", "▼", "▽",

        # —— 序号 / 说明符 ——
        "#", "No.", "NO.", "no.",

    ],

    # 2️⃣ 字符统一替换（同形异码）
    "char_convert_map": {
        # —— 中英文括号 ——
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "「": "[",
        "」": "]",

        # —— 中文标点 → 英文标点 ——
        "，": ",",
        "。": ".",
        "：": ":",
        "；": ";",
        "／": "/",
        "＼": "\\",

        # —— 数学 / 医学常用符号 ——
        "＋": "+",
        "－": "-",
        "×": "*",
        "÷": "/",

        # —— 全角数字 ——
        "０": "0",
        "１": "1",
        "２": "2",
        "３": "3",
        "４": "4",
        "５": "5",
        "６": "6",
        "７": "7",
        "８": "8",
        "９": "9"
    }
}

# 仅女性报告
FEMALE_ONLY_REPORT = {
    2,  # "AMH（抗缪勒管激素）"
    4,  # "国内B超"
    6,  # "免疫五项"
    13, # "优生四项"
    20, # "25羟维生素D"
    28, # "TCT"
    31, # "白带常规"
}

# 仅男性报告
MALE_ONLY_REPORT = {
    3,  #"精子报告"
    29, #"Y - 染色体微缺失"
    35, #"DFI DNA碎片率"
}

#不确定性为mid所包含的关键词
UNCERTAINTY_MID = [

    # ===== 可能性 / 推测类 =====
    "考虑", "可能", "疑似", "倾向", "倾向于",
    "可疑", "考虑为", "考虑有", "考虑存在",
    "不除外", "不能排除", "不排除",

    # ===== 鉴别 / 排除类 =====
    "待排除", "需排除", "需要排除",
    "有待排除", "有待明确",
    "需进一步鉴别", "建议鉴别",
    "鉴别", "鉴别诊断",

    # ===== 建议进一步检查 / 随访 =====
    "建议复查", "建议复诊",
    "建议随访", "建议定期复查",
    "建议结合", "建议进一步",
    "建议完善", "建议完善相关检查",
    "建议结合临床", "建议结合病史",
    "建议结合临床及病史",
    "建议必要时", "必要时复查",

    # ===== 提示 / 印象类 =====
    "提示", "影像提示", "印象提示",
    "提示可能", "提示考虑",
    "影像学提示",

    # ===== 程度 / 范围不确定 =====
    "部分", "轻度", "轻微",
    "早期", "初步考虑",
    "阶段性改变",
    "局部", "局灶性",

    # ===== 需结合其他信息 =====
    "需结合", "需结合临床",
    "需结合病史", "需结合其他检查",
    "结合临床判断",

    # ===== 非定论性表述 =====
    "不典型",
    "表现不特异",
    "意义不明",
    "临床意义不明确",
    "暂不能明确",
    "尚不能明确",
    "目前不能明确",
]

# 二次提取的配置表
SECONDARY_EXTRACTION_CONFIG = {
    3: {  # 精子报告
        "positions": {
            0: {  # 精液量
                "default_value": -1,
                "default_unit": "ml",
                "aliases": [],
                "special_instruction": ""
            },
            1: {  # 液化时间
                "default_value": -1,
                "default_unit": "min",
                "aliases": [],
                "special_instruction": ""
            },
            2: {  # 酸碱度(PH)
                "default_value": -1,
                "default_unit": "",
                "aliases": [],
                "special_instruction": ""
            },
            3: {  # 白细胞浓度
                "default_value": -1,
                "default_unit": "10^6/ml",
                "aliases": [],
                "special_instruction": ""
            },
            4: {  # 精子浓度
                "default_value": -1,
                "default_unit": "10^6/ml",
                "aliases": [],
                "special_instruction": ""
            },
            5: {  # 总精子数
                "default_value": -1,
                "default_unit": "10^6",
                "aliases": ["检测精子数"],
                "special_instruction": "个也是一个单位"
            },
            6: {  # 精子总活力
                "default_value": -1,
                "default_unit": "%",
                "aliases": [],
                "special_instruction": ""
            },
            7: {  # 前向精子百分率
                "default_value": -1,
                "default_unit": "%",
                "aliases": [],
                "special_instruction": ""
            },
            8: {  # 精子正常形态率
                "default_value": -1,
                "default_unit": "%",
                "aliases": [],
                "special_instruction": ""
            },
            9: {  # 血红蛋白A
                "default_value": -1,
                "default_unit": "%",
                "aliases": [],
                "special_instruction": ""
            },
            10: {  # 快速前向运动精子(A)
                "default_value": -1,
                "default_unit": "%",
                "aliases": [],
                "special_instruction": ""
            },
            11: {  # 快速前向运动精子(B)
                "default_value": -1,
                "default_unit": "%",
                "aliases": [],
                "special_instruction": ""
            },
        }
    },
    35: {  # 报告类型
        "positions": {
            0: {
                "default_value": -1,   # 数值兜底
                "default_unit": "%",   # 单位兜底
                "aliases": [],
                "special_instruction": ""
            },
        }
    },
}






