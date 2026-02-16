# === 配置文件: config_eval.py ===

# ==========================================
# 1. 全局基础映射 (文本 -> ID)
# ==========================================
status_map = {
    
    "正常": 0,
    
    "偏低": 1, 
    "过低": 2,
    "偏高": 3, 
    "过高": 4,
    
    "偏小": 5, 
    "过小": 6,
    "偏大": 7,
    "过大": 8,
    
    "阴性": 9, 
    "阳性": 10,
    
    "异常": 11,
    
    "不存在": -1,
}

# ==========================================
# 2. 全局默认严重度 (Default Severity)
# ==========================================
DEFAULT_SEVERITY_MAP = {
    
    0: 0, # 正常
    1: 1, # 偏低
    2: 2, # 过低
    3: 1, # 偏高
    4: 2, # 过高
    
    5: 1, # 偏小
    6: 2, # 过小
    7: 1, # 偏大
    8: 2, # 过大
    
    9: 0,  # 阴性
    10: 3, # 阳性
    
    11: 3,  # 异常
    -1: 0   # 不存在
}

# ==========================================
# 3. 报告指标精细化配置 (Report Configurations)
# ==========================================
# 说明: 
# weight: 默认1
# severity_override: 默认 {}。仅需填写与全局默认不同的项。
# 例如：{"severity_override": {3: 3}} 表示仅将状态3(偏高)改为严重度3，其他照旧。
# "severity_override": { 状态ID: 新的分数, 状态ID: 新的分数 }

REPORT_CONFIGS = {
    "default": [], 

    # === 1. 性激素六项 ===
    1: [
        {"weight": 1, "severity_override": {}}, # Index 0: 黄体生成素(LH)
        {"weight": 1, "severity_override": {}}, # Index 1: 卵泡刺激素(FSH)
        {"weight": 1, "severity_override": {}}, # Index 2: 孕酮(P4)
        {"weight": 1, "severity_override": {}}, # Index 3: 睾酮(T)
        {"weight": 1, "severity_override": {}}, # Index 4: 雌二醇(E2)
        {"weight": 1, "severity_override": {}}, # Index 5: 催乳素(PRL)
    ],

    # === 2. AMH ===
    2: [
        {"weight": 1, "severity_override": {}}, # Index 0: 抗缪勒氏管激素(AMH)
    ],

    # === 3. 精子 ===
    3: [
        {"weight": 1, "severity_override": {}}, # Index 0: 精液量(Volume)
        {"weight": 1, "severity_override": {}}, # Index 1: 液化时间(Liquefaction)
        {"weight": 1, "severity_override": {}}, # Index 2: 酸碱度(PH)
        {"weight": 1, "severity_override": {}}, # Index 3: 白细胞浓度(WBC)
        {"weight": 1, "severity_override": {}}, # Index 4: 精子浓度(Concentration)
        {"weight": 1, "severity_override": {}}, # Index 5: 总精子数(Total Concentration)
        {"weight": 1, "severity_override": {}}, # Index 6: 精子总活力(Motility)
        {"weight": 1, "severity_override": {}}, # Index 7: 前向精子百分率(Progressive motile)
        {"weight": 1, "severity_override": {}}, # Index 8: 精子正常形态率(Morphology)
        {"weight": 1, "severity_override": {}}, # Index 9: 血红蛋白A(HBA)
        {"weight": 1, "severity_override": {}}, # Index 10: 快速前向运动精子(A)
        {"weight": 1, "severity_override": {}}, # Index 11: 快速前向运动精子(B)
    ],

    # === 4. 国内B超 ===
    4: [
        {"weight": 1, "severity_override": {}}, # Index 0: 子宫内膜厚度
        {"weight": 1, "severity_override": {}}, # Index 1: 卵泡总数
        {"weight": 1, "severity_override": {}}, # Index 2: 卵泡尺寸列表
    ],

    # === 5. 泰国B超 ===
    5: [
        {"weight": 1, "severity_override": {}}, # Index 0: 子宫内膜厚度
        {"weight": 1, "severity_override": {}}, # Index 1: 卵泡总数
        {"weight": 1, "severity_override": {}}, # Index 2: 卵泡尺寸列表
    ],

    # === 6. 免疫五项 ===
    6: [
        {"weight": 1, "severity_override": {}}, # Index 0: 免疫球蛋白G(IgG)
        {"weight": 1, "severity_override": {}}, # Index 1: 免疫球蛋白A(IgA)
        {"weight": 1, "severity_override": {}}, # Index 2: 免疫球蛋白M(IgM)
        {"weight": 1, "severity_override": {}}, # Index 3: 补体C3(C3)
        {"weight": 1, "severity_override": {}}, # Index 4: 补体C4(C4)
    ],

    # === 7. 凝血功能 ===
    7: [
        {"weight": 1, "severity_override": {}}, # Index 0: 凝血酶原时间(PT)
        {"weight": 1, "severity_override": {}}, # Index 1: 凝血酶原时间比值(PT-R)
        {"weight": 1, "severity_override": {}}, # Index 2: 凝血酶原时间活动度(PT%)
        {"weight": 1, "severity_override": {}}, # Index 3: 凝血酶原国际标准化比值(PT-INR)
        {"weight": 1, "severity_override": {}}, # Index 4: 活化部分凝血活酶时间(APTT)
        {"weight": 1, "severity_override": {}}, # Index 5: 凝血酶时间(TT)
        {"weight": 1, "severity_override": {}}, # Index 6: 抗凝血酶III(AT-III)
        {"weight": 1, "severity_override": {}}, # Index 7: D-二聚体(D-D)
        {"weight": 1, "severity_override": {}}, # Index 8: 纤维蛋白原(FIB)
        {"weight": 1, "severity_override": {}}, # Index 9: 国际标准化比值(INR)
    ],

    # === 8. 肾功能 ===
    8: [
        {"weight": 1, "severity_override": {}}, # Index 0: 尿素(Urea)
        {"weight": 1, "severity_override": {}}, # Index 1: 尿酸(UA)
        {"weight": 1, "severity_override": {}}, # Index 2: 肌酐(Cr)
        {"weight": 1, "severity_override": {}}, # Index 3: 胱抑素C(CysC)
        {"weight": 1, "severity_override": {}}, # Index 4: 二氧化碳结合力(CO2-CP)
        {"weight": 1, "severity_override": {}}, # Index 5: 葡萄糖(GLU)
    ],

    # === 9. 血型检测 ===
    9: [
        {"weight": 1, "severity_override": {}}, # Index 0: ABO血型(ABO)
        {"weight": 1, "severity_override": {}}, # Index 1: Rh血型(Rh)
    ],

    # === 10. 血常规 ===
    10: [
        {"weight": 1, "severity_override": {}}, # Index 0: 白细胞计数(WBC)
        {"weight": 1, "severity_override": {}}, # Index 1: RDW-CV
        {"weight": 1, "severity_override": {}}, # Index 2: 红细胞计数(RBC)
        {"weight": 1, "severity_override": {}}, # Index 3: 血小板计数(PLT)
        {"weight": 1, "severity_override": {}}, # Index 4: 血小板分布宽度(PDW)
        {"weight": 1, "severity_override": {}}, # Index 5: 血小板压积(PCT)
        {"weight": 1, "severity_override": {}}, # Index 6: 中性粒细胞百分比(NEU%)
        {"weight": 1, "severity_override": {}}, # Index 7: 中性粒细胞计数(NEU#)
        {"weight": 1, "severity_override": {}}, # Index 8: 平均血小板体积(MPV)
        {"weight": 1, "severity_override": {}}, # Index 9: 单核细胞计数(MON#)
        {"weight": 1, "severity_override": {}}, # Index 10: 单核细胞百分比(MON%)
        {"weight": 1, "severity_override": {}}, # Index 11: 平均红细胞体积(MCV)
        {"weight": 1, "severity_override": {}}, # Index 12: MCHC
        {"weight": 1, "severity_override": {}}, # Index 13: HCH
        {"weight": 1, "severity_override": {}}, # Index 14: 淋巴细胞计数(LYM#)
        {"weight": 1, "severity_override": {}}, # Index 15: 淋巴细胞百分比(LYM%)
        {"weight": 1, "severity_override": {}}, # Index 16: 血红蛋白(HGB)
        {"weight": 1, "severity_override": {}}, # Index 17: 红细胞比容(HCT)
        {"weight": 1, "severity_override": {}}, # Index 18: 嗜酸性粒细胞百分比(EO%)
        {"weight": 1, "severity_override": {}}, # Index 19: 嗜酸性粒细胞计数(EO#)
        {"weight": 1, "severity_override": {}}, # Index 20: 嗜碱性粒细胞计数(BAS#)
        {"weight": 1, "severity_override": {}}, # Index 21: 嗜碱性粒细胞百分比(BAS%)
        {"weight": 1, "severity_override": {}}, # Index 22: 未成熟粒细胞百分比(IG%)
        {"weight": 1, "severity_override": {}}, # Index 23: 未成熟粒细胞计数(IG)
        {"weight": 1, "severity_override": {}}, # Index 24: MCH
        {"weight": 1, "severity_override": {}}, # Index 25: 红细胞分布宽度(RDW)
        {"weight": 1, "severity_override": {}}, # Index 26: 有核红细胞计数(NRBC#)
        {"weight": 1, "severity_override": {}}, # Index 27: 有核红细胞百分比(NRBC%)
    ],

    # === 11. 衣原体 ===
    11: [
        {"weight": 1, "severity_override": {}}, # Index 0: 衣原体DNA(CT-DNA)
    ],

    # === 12. 传染病四项 ===
    12: [
        {"weight": 1, "severity_override": {}}, # Index 0: 乙肝表面抗原(HBsAg)
        {"weight": 1, "severity_override": {}}, # Index 1: 乙肝表面抗体(HBsAb)
        {"weight": 1, "severity_override": {}}, # Index 2: 乙肝e抗原(HBeAg)
        {"weight": 1, "severity_override": {}}, # Index 3: 乙肝e抗体(HBeAb)
        {"weight": 1, "severity_override": {}}, # Index 4: 乙肝核心抗体(HBcAb)
        {"weight": 1, "severity_override": {}}, # Index 5: 丙肝抗体(Anti-HCV)
        {"weight": 1, "severity_override": {}}, # Index 6: 艾滋抗体(Anti-HIV)
        {"weight": 1, "severity_override": {}}, # Index 7: 梅毒螺旋体抗体(TPAb)
    ],

    # === 13. 优生五项TORCH ===
    13: [
        {"weight": 1, "severity_override": {}}, # Index 0: 巨细胞病毒IgM
        {"weight": 1, "severity_override": {}}, # Index 1: 巨细胞病毒IgG
        {"weight": 1, "severity_override": {}}, # Index 2: 弓形虫IgM
        {"weight": 1, "severity_override": {}}, # Index 3: 弓形虫IgG
        {"weight": 1, "severity_override": {}}, # Index 4: 风疹病毒IgM
        {"weight": 1, "severity_override": {}}, # Index 5: 风疹病毒IgG
        {"weight": 1, "severity_override": {}}, # Index 6: HSV-1-IgM
        {"weight": 1, "severity_override": {}}, # Index 7: HSV-1-IgG
        {"weight": 1, "severity_override": {}}, # Index 8: HSV-2-IgM
        {"weight": 1, "severity_override": {}}, # Index 9: HSV-2-IgG
        {"weight": 1, "severity_override": {}}, # Index 10: B19-IgM
        {"weight": 1, "severity_override": {}}, # Index 11: B19-IgG
    ],

    # === 14. 支原体 ===
    14: [
        {"weight": 1, "severity_override": {}}, # Index 0: 解脲支原体(Uu)
        {"weight": 1, "severity_override": {}}, # Index 1: 人型支原体(Mh)
    ],

    # === 15. HCG妊娠诊断报告 ===
    15: [
        {"weight": 1, "severity_override": {}}, # Index 0: 绒毛膜促性腺激素(HCG)
    ],

    # === 16. 地中海贫血症 ===
    16: [
        {"weight": 1, "severity_override": {}}, # Index 0: a-地贫基因检测(3种缺失型)
        {"weight": 1, "severity_override": {}}, # Index 1: a-地贫基因检测(3种非缺失型)
        {"weight": 1, "severity_override": {}}, # Index 2: β-地贫基因检测(17种突变)
    ],

    # === 17. 贫血四项 ===
    17: [
        {"weight": 1, "severity_override": {}}, # Index 0: 铁蛋白(Fer)
        {"weight": 1, "severity_override": {}}, # Index 1: 叶酸(Folate)
        {"weight": 1, "severity_override": {}}, # Index 2: 维生素B12(VitB12)
        {"weight": 1, "severity_override": {}}, # Index 3: 转铁蛋白(TRF)
    ],

    # === 18. 肝功五项 ===
    18: [
        {"weight": 1, "severity_override": {}}, # Index 0: 白蛋白(ALB)
        {"weight": 1, "severity_override": {}}, # Index 1: ALT
        {"weight": 1, "severity_override": {}}, # Index 2: AST
        {"weight": 1, "severity_override": {}}, # Index 3: AST/ALT
        {"weight": 1, "severity_override": {}}, # Index 4: 总胆红素(T-BiL)
        {"weight": 1, "severity_override": {}}, # Index 5: 直接胆红素(D-BiL)
        {"weight": 1, "severity_override": {}}, # Index 6: 间接胆红素(I-BiL)
        {"weight": 1, "severity_override": {}}, # Index 7: 谷氨酰转肽酶(GGT)
        {"weight": 1, "severity_override": {}}, # Index 8: 总蛋白(TP)
        {"weight": 1, "severity_override": {}}, # Index 9: 球蛋白(GLO)
        {"weight": 1, "severity_override": {}}, # Index 10: 白蛋白/球蛋白比值(A/G)
        {"weight": 1, "severity_override": {}}, # Index 11: 碱性磷酸酶(ALP)
        {"weight": 1, "severity_override": {}}, # Index 12: 胆碱脂酶(ChE)
        {"weight": 1, "severity_override": {}}, # Index 13: α-L-岩藻糖苷酶(AFU)
        {"weight": 1, "severity_override": {}}, # Index 14: 腺苷脱氨酶(ADA)
        {"weight": 1, "severity_override": {}}, # Index 15: 总胆汁酸(TBA)
        {"weight": 1, "severity_override": {}}, # Index 16: 脂蛋白(a)
    ],

    # === 19. 甲功 ===
    19: [
        {"weight": 1, "severity_override": {}}, # Index 0: 促甲状腺激素(TSH)
        {"weight": 1, "severity_override": {}}, # Index 1: 总三碘甲状腺原氨酸(TT3)
        {"weight": 1, "severity_override": {}}, # Index 2: 总甲状腺素(TT4)
        {"weight": 1, "severity_override": {}}, # Index 3: 游离三碘甲状腺原氨酸(FT3)
        {"weight": 1, "severity_override": {}}, # Index 4: 游离甲状腺素(FT4)
        {"weight": 1, "severity_override": {}}, # Index 5: 甲状腺过氧化物酶抗体(TPOAb)
        {"weight": 1, "severity_override": {}}, # Index 6: 甲状腺球蛋白抗体(TGAb)
        {"weight": 1, "severity_override": {}}, # Index 7: 促甲状腺激素受体抗体(TSHRAb)
    ],

    # === 20. 孕前基础健康 ===
    20: [
        {"weight": 1, "severity_override": {}}, # Index 0: 25-羟维生素D
    ],

    # === 21. 尿常规 ===
    21: [
        {"weight": 1, "severity_override": {}}, # Index 0: 尿蛋白(PRO)
        {"weight": 1, "severity_override": {}}, # Index 1: 尿糖(GLU)
        {"weight": 1, "severity_override": {}}, # Index 2: 酮体(KET)
        {"weight": 1, "severity_override": {}}, # Index 3: 尿胆红素(BIL)
        {"weight": 1, "severity_override": {}}, # Index 4: 尿胆原(URO)
        {"weight": 1, "severity_override": {}}, # Index 5: 亚硝酸盐(NIT)
    ],

    # === 22. 核医学 ===
    22: [
        {"weight": 1, "severity_override": {}}, # Index 0: 糖类抗原19-9(CA199)
    ],

    # === 23. 结核感染T细胞检测 ===
    23: [
        {"weight": 1, "severity_override": {}}, # Index 0: IFN-(N)
        {"weight": 1, "severity_override": {}}, # Index 1: 结核感染T细胞检测
        {"weight": 1, "severity_override": {}}, # Index 2: IFN-Y(T)
        {"weight": 1, "severity_override": {}}, # Index 3: IFN-V(T-N)
    ],

    # === 24. RF分型(类风湿因子) ===
    24: [
        {"weight": 1, "severity_override": {}}, # Index 0: 类风湿因子IgA
        {"weight": 1, "severity_override": {}}, # Index 1: 类风湿因子IgG
        {"weight": 1, "severity_override": {}}, # Index 2: 类风湿因子IgM
    ],

    # === 25. 血脂 ===
    25: [
        {"weight": 1, "severity_override": {}}, # Index 0: 总胆固醇(TC)
        {"weight": 1, "severity_override": {}}, # Index 1: 甘油三酯(TG)
        {"weight": 1, "severity_override": {}}, # Index 2: LDL-C
        {"weight": 1, "severity_override": {}}, # Index 3: HDL-C
    ],

    # === 26. 血糖 ===
    26: [
        {"weight": 1, "severity_override": {}}, # Index 0: 空腹血糖(FPG)
        {"weight": 1, "severity_override": {}}, # Index 1: 餐后2小时血糖(2hPG)
        {"weight": 1, "severity_override": {}}, # Index 2: 糖化血红蛋白(HbA1c)
    ],

    # === 27. 同型半胱氨酸 ===
    27: [
        {"weight": 1, "severity_override": {}}, # Index 0: 同型半胱氨酸(HCY)
    ],

    # === 28. TCT ===
    28: [
        {"weight": 1, "severity_override": {}}, # Index 0: tct-NILM
        {"weight": 1, "severity_override": {}}, # Index 1: 炎症细胞改变
        {"weight": 1, "severity_override": {}}, # Index 2: ASC-US
        {"weight": 1, "severity_override": {}}, # Index 3: LSIL
        {"weight": 1, "severity_override": {}}, # Index 4: HSIL
        {"weight": 1, "severity_override": {}}, # Index 5: 鳞状细胞癌/腺细胞癌
    ],

    # === 29. Y-染色体微缺失 ===
    29: [
        {"weight": 1, "severity_override": {}}, # Index 0: sY84
        {"weight": 1, "severity_override": {}}, # Index 1: sY86
        {"weight": 1, "severity_override": {}}, # Index 2: sY127
        {"weight": 1, "severity_override": {}}, # Index 3: sY134
        {"weight": 1, "severity_override": {}}, # Index 4: sY254
        {"weight": 1, "severity_override": {}}, # Index 5: sY255
    ],

    # === 30. 狼疮 ===
    30: [
        {"weight": 1, "severity_override": {}}, # Index 0: LA1
        {"weight": 1, "severity_override": {}}, # Index 1: LA2
        {"weight": 1, "severity_override": {}}, # Index 2: LA1/LA2
    ],

    # === 31. 白带常规 ===
    31: [
        {"weight": 1, "severity_override": {}}, # Index 0: 阴道清洁度
        {"weight": 1, "severity_override": {}}, # Index 1: 白细胞(WBC)
        {"weight": 1, "severity_override": {}}, # Index 2: 红细胞(RBC)
        {"weight": 1, "severity_override": {}}, # Index 3: 滴虫(TV)
        {"weight": 1, "severity_override": {}}, # Index 4: 霉菌(FV)
        {"weight": 1, "severity_override": {}}, # Index 5: 细菌性阴道病(BV)
        {"weight": 1, "severity_override": {}}, # Index 6: 酸碱度(pH)
        {"weight": 1, "severity_override": {}}, # Index 7: 线索细胞
        {"weight": 1, "severity_override": {}}, # Index 8: 过氧化氢(H2O2)
        {"weight": 1, "severity_override": {}}, # Index 9: β-葡萄糖醛酸酶
        {"weight": 1, "severity_override": {}}, # Index 10: 唾液酸苷酶
        {"weight": 1, "severity_override": {}}, # Index 11: 乙酰氨基葡萄糖苷酶
        {"weight": 1, "severity_override": {}}, # Index 12: 白细胞酯酶(LE)
    ],

    # === 32. 肿瘤标记物 ===
    32: [
        {"weight": 1, "severity_override": {}}, # Index 0: 糖类抗原125(CA125)
        {"weight": 1, "severity_override": {}}, # Index 1: 甲胎蛋白(AFP)
        {"weight": 1, "severity_override": {}}, # Index 2: 癌胚抗原(CEA)
    ],

    # === 33. 淋球菌 ===
    33: [
        {"weight": 1, "severity_override": {}}, # Index 0: 淋球菌培养
    ],

    # === 34. 精子线粒体膜电位检测 ===
    34: [
        {"weight": 1, "severity_override": {}}, # Index 0: 精子线粒体膜电位(MMP)
    ],

    # === 35. DNA碎片化指数 ===
    35: [
        {"weight": 1, "severity_override": {}}, # Index 0: DNA碎片化指数(DFI)
    ],

}

# ==========================================
# 4. [修改] 组合条件加分规则 (Compound Score Rules)
# ==========================================
# 逻辑: 
# 每个规则包含一个 conditions 列表。
# 只有当列表中 *所有* 条件(AND逻辑) 都满足时，总分才会加上 score_add。
# 可以用于单项(列表长度1)或多项组合(列表长度>1)。

CRITICAL_SCORE_RULES = [
    # --- 示例 1: 单项高风险 (HIV阳性) ---
    {
        "name": "HIV阳性-极高风险",
        "score_add": 100,  # 直接加100分，确保进入最高风险区
        "conditions": [
            # 传染病(12) - 艾滋抗体(6) - 阳性(10)
            {"report_type": 12, "index": 6, "status_codes": [10]} 
        ]
    },
    
    # --- 示例 2: 组合风险 (PCOS多囊典型特征) ---
    # 假设逻辑: 激素(1)中LH(0)偏高 + 激素(1)中T(3)偏高
    {
        "name": "疑似多囊卵巢综合征(PCOS)风险",
        "score_add": 20, # 追加 20 分
        "conditions": [
            {"report_type": 1, "index": 0, "status_codes": [3, 4]}, # LH 偏高/过高
            {"report_type": 1, "index": 3, "status_codes": [3, 4]}  # 睾酮 偏高/过高
        ]
    },

    # --- 示例 3: 肝肾功能同时受损 (Systemic Failure Risk) ---
    # 假设逻辑: 肝功(18)ALT(1)异常 + 肾功(8)肌酐(2)异常
    {
        "name": "肝肾功能联合异常",
        "score_add": 30,
        "conditions": [
            {"report_type": 18, "index": 1, "status_codes": [3, 4, 11]}, # ALT
            {"report_type": 8,  "index": 2, "status_codes": [3, 4, 11]}  # Cr
        ]
    },
    
    # --- 示例 4: TCT癌变风险 ---
    {
        "name": "宫颈癌变高风险",
        "score_add": 80,
        "conditions": [
            {"report_type": 28, "index": 5, "status_codes": [10, 11]} # 鳞状细胞癌
        ]
    }
]


# ==========================================
# 5. 急迫等级综合配置 (使用前闭后开区间)
# ==========================================
# 逻辑说明: [min, max) -> min <= score < max
# Level 5 的上限设为 float('inf') 确保所有高分都被捕获
URGENCY_CONFIG = [
    {
        "level": 1, 
        "range": (0, 10),      # 0 <= score < 10
        "desc": "健康正常 (Health/Normal)"
    },
    {
        "level": 2, 
        "range": (10, 25),     # 10 <= score < 25
        "desc": "低风险/需关注 (Low Risk)"
    },
    {
        "level": 3, 
        "range": (25, 45),     # 25 <= score < 45
        "desc": "中风险/建议复查 (Medium Risk)"
    },
    {
        "level": 4, 
        "range": (45, 70),     # 45 <= score < 70
        "desc": "高风险/尽快就医 (High Risk)"
    },
    {
        "level": 5, 
        "range": (70, float('inf')), # 70 <= score < ∞
        "desc": "极高风险/急诊 (Severe Risk)"
    }
]
