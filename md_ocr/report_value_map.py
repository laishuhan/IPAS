# 2025/2/3-1:34 laishuhan
#======================================= 指标值筛除部分 =======================================
# 通用：逐字符无脑删除（不承载医学语义）
# 通用逐字符删除规则（不承载医学语义，只用于标记/比较）
COMMON_REMOVE_CHARS = [
    # 异常标记
    "↑", "↓",

    # 比较符号
    "≥", "≤",
    ">", "<", "=",

    #空格
    " ",
]

# 阴性阳性指标可能需要删除的字符
NEGATIVE_POSITIVE_REMOVE_CHARS = ["(", ")", "（", "）", "+", "-"]



# REPORT_VALUE_FILTER_MAP = {
#     报告类型: {
#         指标位置: {"remove_chars": ["↑", "↓"], #字符剔除
#                     "remove_regex": [r"\(H\)$", r"\(L\)$", r"（高）$", r"（低）$"] #正则剔除
#                     },
#     }
# }
REPORT_VALUE_FILTER_MAP = {

    # ===== 报告类型 1 : 性激素六项 =====
    1: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 黄体生成素(LH)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 卵泡刺激素(FSH)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 孕酮(P4)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 睾酮(T)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 雌二醇(E2)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 催乳素(PRL)
    },

    # ===== 报告类型 2 : amh =====
    2: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 抗缪勒氏管激素(AMH)
    },

    # ===== 报告类型 3 : 精子 =====
    3: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 精液量(Volume)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 液化时间(Liquefaction)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 酸碱度(PH)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 白细胞浓度(WBC)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 精子浓度(Concentration)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 总精子数(Total Concentration)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 精子总活力(Motility)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # 前向精子百分率(Progressive motile)
        8: {"remove_chars": COMMON_REMOVE_CHARS},  # 精子正常形态率(Morphology)
        9: {"remove_chars": COMMON_REMOVE_CHARS},  # 血红蛋白A(HBA)
        # 10: {"remove_chars": COMMON_REMOVE_CHARS},  # DNA碎片化指数(DFI)
        10: {"remove_chars": COMMON_REMOVE_CHARS},  # 快速前向运动精子（A）
        11: {"remove_chars": COMMON_REMOVE_CHARS},  # 慢速前向运动精子（B）
    },

    # ===== 报告类型 4 : 中文B超 =====
    4: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 子宫内膜厚度
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 卵泡总数
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 最大卵泡尺寸
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 卵泡发育趋势
    },

    # ===== 报告类型 6 : 免疫五项 =====
    6: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 免疫球蛋白G(IgG)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 免疫球蛋白A(IgA)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 免疫球蛋白M(IgM)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 补体C3(C3)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 补体C4(C4)
    },

    # ===== 报告类型 7 : 凝血功能 =====
    7: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 凝血酶原时间(PT)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 凝血酶原时间比值(PT-R)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 凝血酶原时间活动度(PT%)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 凝血酶原国际标准化比值(PT-INR)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 活化部分凝血活酶时间(APTT)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 凝血酶时间(TT)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 抗凝血酶III(AT-III)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # D-二聚体(D-D)
        8: {"remove_chars": COMMON_REMOVE_CHARS},  # 纤维蛋白原(FIB)
        9: {"remove_chars": COMMON_REMOVE_CHARS},  # 国际标准化比值(INR)
    },

    # ===== 报告类型 8 : 肾功能 =====
    8: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 尿素(Urea)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 尿酸(UA)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 肌酐(Cr)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 胱抑素C(CysC)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 二氧化碳结合力(CO2-CP)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 葡萄糖(GLU)
    },

    # ===== 报告类型 9 : 血型检测 =====
    9: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # ABO血型(ABO)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # Rh血型(Rh)
    },

    # ===== 报告类型 10 : 血常规 =====
    10: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 白细胞计数(WBC)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 红细胞分布宽度变异系数(RDW-CV)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 红细胞计数(RBC)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 血小板计数(PLT)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 血小板分布宽度(PDW)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 血小板压积(PCT)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 中性粒细胞百分比(NEU%)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # 中性粒细胞计数(NEU#)
        8: {"remove_chars": COMMON_REMOVE_CHARS},  # 平均血小板体积(MPV)
        9: {"remove_chars": COMMON_REMOVE_CHARS},  # 单核细胞计数(MON#)
        10: {"remove_chars": COMMON_REMOVE_CHARS},  # 单核细胞百分比(MON%)
        11: {"remove_chars": COMMON_REMOVE_CHARS},  # 平均红细胞体积(MCV)
        12: {"remove_chars": COMMON_REMOVE_CHARS},  # 平均红细胞血红蛋白浓度(MCHC)
        13: {"remove_chars": COMMON_REMOVE_CHARS},  # 红细胞血红蛋白含量(HCH)
        14: {"remove_chars": COMMON_REMOVE_CHARS},  # 淋巴细胞计数(LYM#)
        15: {"remove_chars": COMMON_REMOVE_CHARS},  # 淋巴细胞百分比(LYM%)
        16: {"remove_chars": COMMON_REMOVE_CHARS},  # 血红蛋白(HGB)
        17: {"remove_chars": COMMON_REMOVE_CHARS},  # 红细胞比容(HCT)
        18: {"remove_chars": COMMON_REMOVE_CHARS},  # 嗜酸性粒细胞百分比(EO%)
        19: {"remove_chars": COMMON_REMOVE_CHARS},  # 嗜酸性粒细胞计数(EO#)
        20: {"remove_chars": COMMON_REMOVE_CHARS},  # 嗜碱性粒细胞计数(BAS#)
        21: {"remove_chars": COMMON_REMOVE_CHARS},  # 嗜碱性粒细胞百分比(BAS%)
        22: {"remove_chars": COMMON_REMOVE_CHARS},  # 未成熟粒细胞百分比(IG%)
        23: {"remove_chars": COMMON_REMOVE_CHARS},  # 未成熟粒细胞计数(IG)
        24: {"remove_chars": COMMON_REMOVE_CHARS},  # 平均红细胞血红蛋白量(MCH)
        25: {"remove_chars": COMMON_REMOVE_CHARS},  # 红细胞分布宽度(RDW)
        26: {"remove_chars": COMMON_REMOVE_CHARS},  # 有核红细胞计数(NRBC#)
        27: {"remove_chars": COMMON_REMOVE_CHARS},  # 有核红细胞百分比(NRBC%)
    },

    # ===== 报告类型 11 : 衣原体 =====
    11: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 衣原体DNA(CT-DNA)
    },

    # ===== 报告类型 12 : 传染病四项 =====
    12: {
        0: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 乙肝表面抗原(HBsAg)
        1: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 乙肝表面抗体(HBsAb)
        2: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 乙肝e抗原(HBeAg)
        3: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 乙肝e抗体(HBeAb)
        4: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 乙肝核心抗体(HBcAb)
        5: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 丙肝抗体(Anti-HCV)
        6: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 艾滋抗体(Anti-HIV)
        7: {"remove_chars": COMMON_REMOVE_CHARS + NEGATIVE_POSITIVE_REMOVE_CHARS},  # 梅毒螺旋体抗体(TPAb)
    },

    # ===== 报告类型 13 : 优生五项TORCH =====
    13: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 巨细胞病毒IgM(CMV-IgM)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 巨细胞病毒IgG(CMV-IgG)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 弓形虫IgM(TOX-IgM)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 弓形虫IgG(TOX-IgG)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 风疹病毒IgM(RV-IgM)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 风疹病毒IgG(RV-IgG)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 单纯疱疹病毒1型IgM(HSV-1-IgM)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # 单纯疱疹病毒1型IgG(HSV-1-IgG)
        8: {"remove_chars": COMMON_REMOVE_CHARS},  # 单纯疱疹病毒2型IgM(HSV-2-IgM)
        9: {"remove_chars": COMMON_REMOVE_CHARS},  # 单纯疱疹病毒2型IgG(HSV-2-IgG)
        10: {"remove_chars": COMMON_REMOVE_CHARS},  # B19细小病毒IgM(B19-IgM)
        11: {"remove_chars": COMMON_REMOVE_CHARS},  # B19细小病毒IgG(B19-IgG)
    },

    # ===== 报告类型 14 : 支原体 =====
    14: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 解脲支原体(Uu)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 人型支原体(Mh)
    },

    # ===== 报告类型 15 : HCG妊娠诊断报告 =====
    15: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 绒毛膜促性腺激素(HCG)
    },

    # ===== 报告类型 16 : 地中海贫血症 =====
    16: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # a-地贫基因检测(3种缺失型)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # a-地贫基因检测(3种非缺失型)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # β-地贫基因检测(17种突变)
    },

    # ===== 报告类型 17 : 贫血四项 =====
    17: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 铁蛋白(Fer)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 叶酸(Folate)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 维生素B12(VitB12)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 转铁蛋白(TRF)
    },

    # ===== 报告类型 18 : 肝功五项 =====
    18: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 白蛋白(ALB)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 丙氨酸氨基转移酶(ALT)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 天门冬氨酸氨基转移酶(AST)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 天门冬氨酸/丙氨酸氨基转移酶比值(AST/ALT)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 总胆红素(T-BiL)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 直接胆红素(D-BiL)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 间接胆红素(I-BiL)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # 谷氨酰转肽酶(GGT)
        8: {"remove_chars": COMMON_REMOVE_CHARS},  # 总蛋白(TP)
        9: {"remove_chars": COMMON_REMOVE_CHARS},  # 球蛋白(GLO)
        10: {"remove_chars": COMMON_REMOVE_CHARS},  # 白蛋白/球蛋白比值(A/G)
        11: {"remove_chars": COMMON_REMOVE_CHARS},  # 碱性磷酸酶(ALP)
        12: {"remove_chars": COMMON_REMOVE_CHARS},  # 胆碱脂酶(ChE)
        13: {"remove_chars": COMMON_REMOVE_CHARS},  # α-L-岩藻糖苷酶(AFU)
        14: {"remove_chars": COMMON_REMOVE_CHARS},  # 腺苷脱氨酶(ADA)
        15: {"remove_chars": COMMON_REMOVE_CHARS},  # 总胆汁酸(TBA)
    },

    # ===== 报告类型 19 : 甲功 =====
    19: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 促甲状腺激素(TSH)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 总三碘甲状腺原氨酸(TT3)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 总甲状腺素(TT4)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 游离三碘甲状腺原氨酸(FT3)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 游离甲状腺素(FT4)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 甲状腺过氧化物酶抗体(TPOAb)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 甲状腺球蛋白抗体(TGAb)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # 促甲状腺激素受体抗体(TSHRAb)
    },

    # ===== 报告类型 20 : 孕前基础健康评估报告 =====
    20: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 25-羟维生素D(25-OH-VD)
    },

    # ===== 报告类型 21 : 尿常规 =====
    21: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 尿蛋白(PRO)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 尿糖(GLU)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 酮体(KET)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 尿胆红素(BIL)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 尿胆原(URO)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 亚硝酸盐(NIT)
    },

    # ===== 报告类型 22 : 核医学 =====
    22: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 糖类抗原19-9(CA199)
    },

    # ===== 报告类型 23 : 结核感染T细胞检测 =====
    23: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # IFN-(N)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 结核感染T细胞检测
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # IFN-Y(T)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # IFN-V(T-N)
    },

    # ===== 报告类型 24 : RF分型(类风湿因子) =====
    24: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 类风湿因子IgA
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 类风湿因子IgG
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 类风湿因子IgM
    },

    # ===== 报告类型 25 : 血脂 =====
    25: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 总胆固醇(TC)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 甘油三酯(TG)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 低密度脂蛋白胆固醇(LDL-C)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 高密度脂蛋白胆固醇(HDL-C)
    },

    # ===== 报告类型 26 : 血糖 =====
    26: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 空腹血糖(FPG)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 餐后2小时血糖(2hPG)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 糖化血红蛋白(HbA1c)
    },

    # ===== 报告类型 27 : 同型半胱氨酸报告 =====
    27: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 同型半胱氨酸(HCY)
    },

    # ===== 报告类型 28 : TCT =====
    28: {
        # 0: {"remove_chars": COMMON_REMOVE_CHARS},  # 未见上皮内病变或恶性细胞(tct-NILM)
        # 1: {"remove_chars": COMMON_REMOVE_CHARS},  # 炎症细胞改变(tct-Inflammatory cell changes)
        # 2: {"remove_chars": COMMON_REMOVE_CHARS},  # 意义不明的非典型鳞状细胞(ASC-US)
        # 3: {"remove_chars": COMMON_REMOVE_CHARS},  # 低度鳞状上皮内病变(LSIL)
        # 4: {"remove_chars": COMMON_REMOVE_CHARS},  # 高度鳞状上皮内病变(HSIL)
        # 5: {"remove_chars": COMMON_REMOVE_CHARS},  # 鳞状细胞癌/腺细胞癌(squamous_cell_carcinoma)
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 占位
    },

    # ===== 报告类型 29 : Y - 染色体微缺失 =====
    29: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # Y-染色体微缺失(sY84)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # Y-染色体微缺失(sY86)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # Y-染色体微缺失(sY127)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # Y-染色体微缺失(sY134)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # Y-染色体微缺失(sY254)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # Y-染色体微缺失(sY255)
    },

    # ===== 报告类型 30 : 狼疮 =====
    30: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 狼疮抗凝物初筛试验1(LA1)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 狼疮抗凝物确定试验(LA2)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 狼疮初筛/狼疮确定(LA1/LA2)
    },

    # ===== 报告类型 31 : 白带常规 =====
    31: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 阴道清洁度(Cleanliness)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 白细胞(WBC)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 红细胞(RBC)
        3: {"remove_chars": COMMON_REMOVE_CHARS},  # 滴虫(TV)
        4: {"remove_chars": COMMON_REMOVE_CHARS},  # 霉菌(FV)
        5: {"remove_chars": COMMON_REMOVE_CHARS},  # 细菌性阴道病(BV)
        6: {"remove_chars": COMMON_REMOVE_CHARS},  # 酸碱度(pH)
        7: {"remove_chars": COMMON_REMOVE_CHARS},  # 线索细胞(Clue Cells)
        8: {"remove_chars": COMMON_REMOVE_CHARS},  # 过氧化氢(H2O2)
        9: {"remove_chars": COMMON_REMOVE_CHARS},  # β-葡萄糖醛酸酶(GUS)
        10: {"remove_chars": COMMON_REMOVE_CHARS},  # 唾液酸苷酶(SNA)
        11: {"remove_chars": COMMON_REMOVE_CHARS},  # 乙酰氨基葡萄糖苷酶(NAG)
        12: {"remove_chars": COMMON_REMOVE_CHARS},  # 白细胞酯酶(LE)
    },

    # ===== 报告类型 32 : 肿瘤标记物 =====
    32: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 糖类抗原125(CA125)
        1: {"remove_chars": COMMON_REMOVE_CHARS},  # 甲胎蛋白(AFP)
        2: {"remove_chars": COMMON_REMOVE_CHARS},  # 癌胚抗原(CEA)
    },

    # ===== 报告类型 33 : 淋球菌 =====
    33: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 淋球菌

    },
    # ===== 报告类型 34 : 精子线粒体膜电位 =====
    34: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # 精子线粒体膜电位(MMP)

    },
    # ===== 报告类型 35 : DNA碎片化指数 =====
    35: {
        0: {"remove_chars": COMMON_REMOVE_CHARS},  # DNA碎片化指数(DFI)
        
    }
}

#======================================= 指标值映射部分 =======================================
# 将 0 映射为空
ZERO_TO_NULL_RULE = {
    "from": [
        0, 0.0, 0.00,
        "0", "0.0", "0.00"
    ],
    "to": -1
}
# 将常见空白占位符映射为空
TEXT_PLACEHOLDER_RULE = {
    "from": [
        "N/A", "NA", "n/a", "N.A.",
        "空", "无", "不存在"
    ],
    "to": -1
}

NEGATIVE_POSITIVE_RULE = [
    {
        "from": ["不存在", '未检出', '-'],
        "to": '阴性'
    },
    {
        "from": ["存在", '已检出', '+'],
        "to": '阳性'
    }
]
Number2Roman_RULE = [
    {
        "from": [1, '1', 'Ⅰ'],
        "to": "I"
    },
    {
        "from": [2, '2', 'Ⅱ'],
        "to": "II"
    },
    {
        "from": [3, '3', 'Ⅲ'],
        "to": "III"
    },
    {
        "from": [4, '4', 'Ⅳ'],
        "to": "IV"
    },
    {
        "from": [5, '5', 'Ⅴ'],
        "to": "V"
    },
    {
        "from": [6, '6', 'Ⅵ'],
        "to": "VI"
    },
    {
        "from": [7, '7', 'Ⅶ'],
        "to": "VII"
    },
    {
        "from": [8, '8', 'Ⅷ'],
        "to": "VIII"
    },
    {
        "from": [9, '9', 'Ⅸ'],
        "to": "IX"
    },
    {
        "from": [10, '10', 'Ⅹ'],
        "to": "X"
    },

]



# RULE_NAME = {"from": [XX], "to": xx}
# REPORT_VALUE_MAP = {
#     报告类型: {
#     指标位置: [RULE_NAME],
#     }
# }
REPORT_VALUE_MAP = {

    # ===== 报告类型 1 : 性激素六项 =====
    1: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 黄体生成素(LH)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 卵泡刺激素(FSH)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 孕酮(P4)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 睾酮(T)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 雌二醇(E2)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 催乳素(PRL)
    },

    # ===== 报告类型 2 : amh =====
    2: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 抗缪勒氏管激素(AMH)
    },

    # ===== 报告类型 3 : 精子 =====
    3: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 精液量(Volume)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 液化时间(Liquefaction)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 酸碱度(PH)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 白细胞浓度(WBC)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 精子浓度(Concentration)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总精子数(Total Concentration)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 精子总活力(Motility)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 前向精子百分率(Progressive motile)
        8: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 精子正常形态率(Morphology)
        9: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 血红蛋白A(HBA)
        # 10: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # DNA碎片化指数(DFI)
        10: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 快速前向运动精子（A）
        11: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 慢速前向运动精子（B）
    },

    # ===== 报告类型 4 : B超（妇科） =====
    4: {
        0: [],  # 子宫内膜厚度(列表)
        1: [],  # 卵泡总数
        2: [],  # 最大卵泡尺寸(列表)
        3: [],  # 大于/严格等于/小于
    },

    # ===== 报告类型 6 : 免疫五项 =====
    6: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 免疫球蛋白G(IgG)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 免疫球蛋白A(IgA)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 免疫球蛋白M(IgM)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 补体C3(C3)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 补体C4(C4)
    },

    # ===== 报告类型 7 : 凝血功能 =====
    7: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 凝血酶原时间(PT)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 凝血酶原时间比值(PT-R)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 凝血酶原时间活动度(PT%)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 凝血酶原国际标准化比值(PT-INR)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 活化部分凝血活酶时间(APTT)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 凝血酶时间(TT)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 抗凝血酶III(AT-III)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # D-二聚体(D-D)
        8: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 纤维蛋白原(FIB)
        9: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 国际标准化比值(INR)
    },

    # ===== 报告类型 8 : 肾功能 =====
    8: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 尿素(Urea)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 尿酸(UA)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 肌酐(Cr)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 胱抑素C(CysC)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 二氧化碳结合力(CO2-CP)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 葡萄糖(GLU)
    },

    # ===== 报告类型 9 : 血型检测 =====
    9: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # ABO血型(ABO)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Rh血型(Rh)
    },

    # ===== 报告类型 10 : 血常规 =====
    10: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 白细胞计数(WBC)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 红细胞分布宽度变异系数(RDW-CV)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 红细胞计数(RBC)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 血小板计数(PLT)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 血小板分布宽度(PDW)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 血小板压积(PCT)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 中性粒细胞百分比(NEU%)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 中性粒细胞计数(NEU#)
        8: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 平均血小板体积(MPV)
        9: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 单核细胞计数(MON#)
        10: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 单核细胞百分比(MON%)
        11: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 平均红细胞体积(MCV)
        12: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 平均红细胞血红蛋白浓度(MCHC)
        13: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 红细胞血红蛋白含量(HCH)
        14: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 淋巴细胞计数(LYM#)
        15: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 淋巴细胞百分比(LYM%)
        16: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 血红蛋白(HGB)
        17: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 红细胞比容(HCT)
        18: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 嗜酸性粒细胞百分比(EO%)
        19: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 嗜酸性粒细胞计数(EO#)
        20: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 嗜碱性粒细胞计数(BAS#)
        21: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 嗜碱性粒细胞百分比(BAS%)
        22: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 未成熟粒细胞百分比(IG%)
        23: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 未成熟粒细胞计数(IG)
        24: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 平均红细胞血红蛋白量(MCH)
        25: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 红细胞分布宽度(RDW)
        26: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 有核红细胞计数(NRBC#)
        27: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 有核红细胞百分比(NRBC%)
    },

    # ===== 报告类型 11 : 衣原体 =====
    11: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 衣原体DNA(CT-DNA)
    },

    # ===== 报告类型 12 : 传染病四项 =====
    12: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 乙肝表面抗原(HBsAg)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 乙肝表面抗体(HBsAb)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 乙肝e抗原(HBeAg)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 乙肝e抗体(HBeAb)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 乙肝核心抗体(HBcAb)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 丙肝抗体(Anti-HCV)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 艾滋抗体(Anti-HIV)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 梅毒螺旋体抗体(TPAb)
    },

    # ===== 报告类型 13 : 优生五项TORCH =====
    13: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 巨细胞病毒IgM(CMV-IgM)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 巨细胞病毒IgG(CMV-IgG)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 弓形虫IgM(TOX-IgM)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 弓形虫IgG(TOX-IgG)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 风疹病毒IgM(RV-IgM)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 风疹病毒IgG(RV-IgG)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 单纯疱疹病毒1型IgM(HSV-1-IgM)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 单纯疱疹病毒1型IgG(HSV-1-IgG)
        8: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 单纯疱疹病毒2型IgM(HSV-2-IgM)
        9: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 单纯疱疹病毒2型IgG(HSV-2-IgG)
        10: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # B19细小病毒IgM(B19-IgM)
        11: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # B19细小病毒IgG(B19-IgG)
    },

    # ===== 报告类型 14 : 支原体 =====
    14: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 解脲支原体(Uu)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 人型支原体(Mh)
    },

    # ===== 报告类型 15 : HCG妊娠诊断报告 =====
    15: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 绒毛膜促性腺激素(HCG)
    },

    # ===== 报告类型 16 : 地中海贫血症 =====
    16: {
        0: [] + NEGATIVE_POSITIVE_RULE ,  # a-地贫基因检测(3种缺失型)
        1: [] + NEGATIVE_POSITIVE_RULE,  # a-地贫基因检测(3种非缺失型)
        2: [] + NEGATIVE_POSITIVE_RULE,  # β-地贫基因检测(17种突变)
    },

    # ===== 报告类型 17 : 贫血四项 =====
    17: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 铁蛋白(Fer)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 叶酸(Folate)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 维生素B12(VitB12)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 转铁蛋白(TRF)
    },

    # ===== 报告类型 18 : 肝功五项 =====
    18: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 白蛋白(ALB)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 丙氨酸氨基转移酶(ALT)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 天门冬氨酸氨基转移酶(AST)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 天门冬氨酸/丙氨酸氨基转移酶比值(AST/ALT)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总胆红素(T-BiL)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 直接胆红素(D-BiL)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 间接胆红素(I-BiL)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 谷氨酰转肽酶(GGT)
        8: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总蛋白(TP)
        9: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 球蛋白(GLO)
        10: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 白蛋白/球蛋白比值(A/G)
        11: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 碱性磷酸酶(ALP)
        12: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 胆碱脂酶(ChE)
        13: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # α-L-岩藻糖苷酶(AFU)
        14: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 腺苷脱氨酶(ADA)
        15: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总胆汁酸(TBA)
    },

    # ===== 报告类型 19 : 甲功 =====
    19: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 促甲状腺激素(TSH)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总三碘甲状腺原氨酸(TT3)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总甲状腺素(TT4)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 游离三碘甲状腺原氨酸(FT3)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 游离甲状腺素(FT4)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 甲状腺过氧化物酶抗体(TPOAb)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 甲状腺球蛋白抗体(TGAb)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 促甲状腺激素受体抗体(TSHRAb)
    },

    # ===== 报告类型 20 : 孕前基础健康评估报告 =====
    20: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 25-羟维生素D(25-OH-VD)
    },

    # ===== 报告类型 21 : 尿常规 =====
    21: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 尿蛋白(PRO)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 尿糖(GLU)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 酮体(KET)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 尿胆红素(BIL)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 尿胆原(URO)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 亚硝酸盐(NIT)
    },

    # ===== 报告类型 22 : 核医学 =====
    22: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 糖类抗原19-9(CA199)
    },

    # ===== 报告类型 23 : 结核感染T细胞检测 =====
    23: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # IFN-（N）
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 结核感染T细胞检测
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # IFN-Y（T）
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # IFN-V（T-N）
    },

    # ===== 报告类型 24 : RF分型（类风湿因子） =====
    24: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 类风湿因子IgA
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 类风湿因子IgG
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 类风湿因子IgM
    },

    # ===== 报告类型 25 : 血脂 =====
    25: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 总胆固醇(TC)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 甘油三酯(TG)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 低密度脂蛋白胆固醇(LDL-C)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 高密度脂蛋白胆固醇(HDL-C)
    },

    # ===== 报告类型 26 : 血糖 =====
    26: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 空腹血糖(FPG)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 餐后2小时血糖(2hPG)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 糖化血红蛋白(HbA1c)
    },

    # ===== 报告类型 27 : 同型半胱氨酸报告 =====
    27: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 同型半胱氨酸(HCY)
    },

    # ===== 报告类型 28 : TCT =====
    28: {
        # 0: [],  # 未见上皮内病变或恶性细胞(tct-NILM)
        # 1: [],  # 炎症细胞改变(tct-Inflammatory cell changes)
        # 2: [],  # 意义不明的非典型鳞状细胞(ASC-US)
        # 3: [],  # 低度鳞状上皮内病变(LSIL)
        # 4: [],  # 高度鳞状上皮内病变(HSIL)
        # 5: [],  # 鳞状细胞癌/腺细胞癌(squamous_cell_carcinoma)
        0: [], # 占位
    },

    # ===== 报告类型 29 : Y - 染色体微缺失 =====
    29: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Y-染色体微缺失(sY84)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Y-染色体微缺失(sY86)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Y-染色体微缺失(sY127)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Y-染色体微缺失(sY134)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Y-染色体微缺失(sY254)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # Y-染色体微缺失(sY255)
    },

    # ===== 报告类型 30 : 狼疮 =====
    30: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 狼疮抗凝物初筛试验1(LA1)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 狼疮抗凝物确定试验(LA2)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 狼疮初筛/狼疮确定(LA1/LA2)
    },

    # ===== 报告类型 31 : 白带常规 =====
    31: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + Number2Roman_RULE,  # 阴道清洁度(Cleanliness)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 白细胞(WBC)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 红细胞(RBC)
        3: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 滴虫(TV)
        4: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 霉菌(FV)
        5: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 细菌性阴道病(BV)
        6: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 酸碱度(pH)
        7: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 线索细胞(Clue Cells)
        8: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 过氧化氢(H2O2)
        9: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # β-葡萄糖-aldo酶(GUS)
        10: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 唾液酸苷酶(SNA)
        11: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 乙酰氨基葡萄糖苷酶(NAG)
        12: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 白细胞酯酶(LE)
    },

    # ===== 报告类型 32 : 肿瘤标记物 =====
    32: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 糖类抗原125(CA125)
        1: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 甲胎蛋白(AFP)
        2: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 癌胚抗原(CEA)
    },
    # ===== 报告类型 33 : 淋球菌 =====
    33: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE] + NEGATIVE_POSITIVE_RULE,  # 淋球菌

    },
    # ===== 报告类型 34 : 精子线粒体膜电位 =====
    34: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # 精子线粒体膜电位(MMP)

    },
    # ===== 报告类型 35 : DNA碎片化指数 =====
    35: {
        0: [ZERO_TO_NULL_RULE, TEXT_PLACEHOLDER_RULE],  # DNA碎片化指数(DFI)

    }
}

#======================================= 指标值（仅）保留部分 =======================================
# 增加一个关键词  对应位置： COMMON_KEEP_KEYWORDS + ["aaa"] 
# 去掉一个关键词  对应位置 ：["阳性"] 
# 完全自定义     对应位置：["xxx","yyy"]           

COMMON_KEEP_KEYWORDS = ["阳性", "阴性"]

REPORT_VALUE_KEEP_MAP = {

    # ===== 报告类型 1 : 性激素六项 =====
    1: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 黄体生成素(LH)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 卵泡刺激素(FSH)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 孕酮(P4)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 睾酮(T)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 雌二醇(E2)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 催乳素(PRL)
    },

    # ===== 报告类型 2 : amh =====
    2: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 抗缪勒氏管激素(AMH)
    },

    # ===== 报告类型 3 : 精子 =====
    3: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 精液量(Volume)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 液化时间(Liquefaction)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 酸碱度(PH)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 白细胞浓度(WBC)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 精子浓度(Concentration)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 总精子数(Total Concentration)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 精子总活力(Motility)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 前向精子百分率(Progressive motile)
        8: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 精子正常形态率(Morphology)
        9: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 血红蛋白A(HBA)
        10: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 快速前向运动精子（A）
        11: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 慢速前向运动精子（B）
    },

    # ===== 报告类型 4 : 中文B超 =====
    4: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 子宫内膜厚度
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 卵泡总数
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 最大卵泡尺寸
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 卵泡发育趋势
    },

    # ===== 报告类型 6 : 免疫五项 =====
    6: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 免疫球蛋白G(IgG)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 免疫球蛋白A(IgA)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 免疫球蛋白M(IgM)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 补体C3(C3)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 补体C4(C4)
    },

    # ===== 报告类型 7 : 凝血功能 =====
    7: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 凝血酶原时间(PT)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 凝血酶原时间比值(PT-R)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 凝血酶原时间活动度(PT%)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 凝血酶原国际标准化比值(PT-INR)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 活化部分凝血活酶时间(APTT)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 凝血酶时间(TT)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 抗凝血酶III(AT-III)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # D-二聚体(D-D)
        8: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 纤维蛋白原(FIB)
        9: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 国际标准化比值(INR)
    },

    # ===== 报告类型 8 : 肾功能 =====
    8: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 尿素(Urea)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 尿酸(UA)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 肌酐(Cr)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 胱抑素C(CysC)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 二氧化碳结合力(CO2-CP)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 葡萄糖(GLU)
    },

    # ===== 报告类型 9 : 血型检测 =====
    9: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # ABO血型(ABO)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Rh血型(Rh)
    },

    # ===== 报告类型 10 : 血常规 =====
    10: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 白细胞计数(WBC)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 红细胞分布宽度变异系数(RDW-CV)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 红细胞计数(RBC)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 血小板计数(PLT)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 血小板分布宽度(PDW)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 血小板压积(PCT)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 中性粒细胞百分比(NEU%)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 中性粒细胞计数(NEU#)
        8: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 平均血小板体积(MPV)
        9: {"keep_contains": COMMON_KEEP_KEYWORDS},   # 单核细胞计数(MON#)
        10: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 单核细胞百分比(MON%)
        11: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 平均红细胞体积(MCV)
        12: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 平均红细胞血红蛋白浓度(MCHC)
        13: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 红细胞血红蛋白含量(HCH)
        14: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 淋巴细胞计数(LYM#)
        15: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 淋巴细胞百分比(LYM%)
        16: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 血红蛋白(HGB)
        17: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 红细胞比容(HCT)
        18: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 嗜酸性粒细胞百分比(EO%)
        19: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 嗜酸性粒细胞计数(EO#)
        20: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 嗜碱性粒细胞计数(BAS#)
        21: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 嗜碱性粒细胞百分比(BAS%)
        22: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 未成熟粒细胞百分比(IG%)
        23: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 未成熟粒细胞计数(IG)
        24: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 平均红细胞血红蛋白量(MCH)
        25: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 红细胞分布宽度(RDW)
        26: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 有核红细胞计数(NRBC#)
        27: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 有核红细胞百分比(NRBC%)
    },

    # ===== 报告类型 11 : 衣原体 =====
    11: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 衣原体DNA(CT-DNA)
    },

    # ===== 报告类型 12 : 传染病四项 =====
    12: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 乙肝表面抗原(HBsAg)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 乙肝表面抗体(HBsAb)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 乙肝e抗原(HBeAg)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 乙肝e抗体(HBeAb)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 乙肝核心抗体(HBcAb)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 丙肝抗体(Anti-HCV)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 艾滋抗体(Anti-HIV)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 梅毒螺旋体抗体(TPAb)
    },

    # ===== 报告类型 13 : 优生五项TORCH =====
    13: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 巨细胞病毒IgM(CMV-IgM)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 巨细胞病毒IgG(CMV-IgG)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 弓形虫IgM(TOX-IgM)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 弓形虫IgG(TOX-IgG)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 风疹病毒IgM(RV-IgM)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 风疹病毒IgG(RV-IgG)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 单纯疱疹病毒1型IgM(HSV-1-IgM)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 单纯疱疹病毒1型IgG(HSV-1-IgG)
        8: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 单纯疱疹病毒2型IgM(HSV-2-IgM)
        9: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 单纯疱疹病毒2型IgG(HSV-2-IgG)
        10: {"keep_contains": COMMON_KEEP_KEYWORDS}, # B19细小病毒IgM(B19-IgM)
        11: {"keep_contains": COMMON_KEEP_KEYWORDS}, # B19细小病毒IgG(B19-IgG)
    },

    # ===== 报告类型 14 : 支原体 =====
    14: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 解脲支原体(Uu)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 人型支原体(Mh)
    },

    # ===== 报告类型 15 : HCG妊娠诊断报告 =====
    15: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 绒毛膜促性腺激素(HCG)
    },

    # ===== 报告类型 16 : 地中海贫血症 =====
    16: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # a-地贫基因检测(3种缺失型)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # a-地贫基因检测(3种非缺失型)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # β-地贫基因检测(17种突变)
    },

    # ===== 报告类型 17 : 贫血四项 =====
    17: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 铁蛋白(Fer)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 叶酸(Folate)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 维生素B12(VitB12)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 转铁蛋白(TRF)
    },

    # ===== 报告类型 18 : 肝功五项 =====
    18: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 白蛋白(ALB)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 丙氨酸氨基转移酶(ALT)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 天门冬氨酸氨基转移酶(AST)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 天门冬氨酸/丙氨酸氨基转移酶比值(AST/ALT)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 总胆红素(T-BiL)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 直接胆红素(D-BiL)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 间接胆红素(I-BiL)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 谷氨酰转肽酶(GGT)
        8: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 总蛋白(TP)
        9: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 球蛋白(GLO)
        10: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 白蛋白/球蛋白比值(A/G)
        11: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 碱性磷酸酶(ALP)
        12: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 胆碱脂酶(ChE)
        13: {"keep_contains": COMMON_KEEP_KEYWORDS}, # α-L-岩藻糖苷酶(AFU)
        14: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 腺苷脱氨酶(ADA)
        15: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 总胆汁酸(TBA)
    },

    # ===== 报告类型 19 : 甲功 =====
    19: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 促甲状腺激素(TSH)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 总三碘甲状腺原氨酸(TT3)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 总甲状腺素(TT4)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 游离三碘甲状腺原氨酸(FT3)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 游离甲状腺素(FT4)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 甲状腺过氧化物酶抗体(TPOAb)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 甲状腺球蛋白抗体(TGAb)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 促甲状腺激素受体抗体(TSHRAb)
    },

    # ===== 报告类型 20 : 孕前基础健康评估报告 =====
    20: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 25-羟维生素D(25-OH-VD)
    },

    # ===== 报告类型 21 : 尿常规 =====
    21: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 尿蛋白(PRO)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 尿糖(GLU)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 酮体(KET)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 尿胆红素(BIL)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 尿胆原(URO)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 亚硝酸盐(NIT)
    },

    # ===== 报告类型 22 : 核医学 =====
    22: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 糖类抗原19-9(CA199)
    },

    # ===== 报告类型 23 : 结核感染T细胞检测 =====
    23: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # IFN-(N)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 结核感染T细胞检测
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # IFN-Y(T)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # IFN-V(T-N)
    },

    # ===== 报告类型 24 : RF分型(类风湿因子) =====
    24: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 类风湿因子IgA
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 类风湿因子IgG
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 类风湿因子IgM
    },

    # ===== 报告类型 25 : 血脂 =====
    25: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 总胆固醇(TC)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 甘油三酯(TG)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 低密度脂蛋白胆固醇(LDL-C)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 高密度脂蛋白胆固醇(HDL-C)
    },

    # ===== 报告类型 26 : 血糖 =====
    26: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 空腹血糖(FPG)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 餐后2小时血糖(2hPG)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 糖化血红蛋白(HbA1c)
    },

    # ===== 报告类型 27 : 同型半胱氨酸报告 =====
    27: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 同型半胱氨酸(HCY)
    },

    # ===== 报告类型 28 : TCT =====
    28: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 占位
    },

    # ===== 报告类型 29 : Y - 染色体微缺失 =====
    29: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Y-染色体微缺失(sY84)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Y-染色体微缺失(sY86)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Y-染色体微缺失(sY127)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Y-染色体微缺失(sY134)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Y-染色体微缺失(sY254)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # Y-染色体微缺失(sY255)
    },

    # ===== 报告类型 30 : 狼疮 =====
    30: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 狼疮抗凝物初筛试验1(LA1)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 狼疮抗凝物确定试验(LA2)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 狼疮初筛/狼疮确定(LA1/LA2)
    },

    # ===== 报告类型 31 : 白带常规 =====
    31: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 阴道清洁度(Cleanliness)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 白细胞(WBC)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 红细胞(RBC)
        3: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 滴虫(TV)
        4: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 霉菌(FV)
        5: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 细菌性阴道病(BV)
        6: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 酸碱度(pH)
        7: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 线索细胞(Clue Cells)
        8: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 过氧化氢(H2O2)
        9: {"keep_contains": COMMON_KEEP_KEYWORDS},  # β-葡萄糖醛酸酶(GUS)
        10: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 唾液酸苷酶(SNA)
        11: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 乙酰氨基葡萄糖苷酶(NAG)
        12: {"keep_contains": COMMON_KEEP_KEYWORDS}, # 白细胞酯酶(LE)
    },

    # ===== 报告类型 32 : 肿瘤标记物 =====
    32: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 糖类抗原125(CA125)
        1: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 甲胎蛋白(AFP)
        2: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 癌胚抗原(CEA)
    },

    # ===== 报告类型 33 : 淋球菌 =====
    33: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 淋球菌
    },

    # ===== 报告类型 34 : 精子线粒体膜电位 =====
    34: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # 精子线粒体膜电位(MMP)
    },

    # ===== 报告类型 35 : DNA碎片化指数 =====
    35: {
        0: {"keep_contains": COMMON_KEEP_KEYWORDS},  # DNA碎片化指数(DFI)
    },
}
