#0127-19:10 laishuhan add "脂蛋白(a)"
#0127-jww add keywords(3,16,19)
#0128-13:17 jww correct the keyword for tt3(reomve "甲状腺激素定量")(type 19)
#0130-10:25 jww add "前向运动精子(PR)"(type 3)
#0130-14:09 jww add "乙型肝炎病毒表面抗原","乙型肝炎病毒表面抗体","乙型肝炎病毒e抗原","乙型肝炎病毒e抗体","乙型肝炎病毒核心抗体","人类免疫缺陷病毒抗原抗体"(type 12)
#0202-09:05 jww add "α25羟基维生素D" (type 20)
#0202-09:51 jww add "精子DNA碎片指数(DFI)" (type 35)
#0203-15:52 jww add "碎片分析检测值"(type 35)
#0210-15:36 jww add "dRVVT-R","dRVVT-S","dRVVT-C"(type 20)
#0210-16:03 jww add "精子DNA碎片指数检测(SDFI)" (type35)
#0211-16:39 jww add "类风湿因子" (type 24)
class General_Report: 
    def __init__(self, report_name, index, unit_conversions, keywords=None):
        self.report_name = report_name
        self.index = index
        self.unit_conversions = unit_conversions
        
        # 强制使用 keywords，若未传入则默认使用 index 名称
        if keywords:
            self.keywords = keywords
        else:
            self.keywords = [[name] for name in index]

# 1. 性激素六项
six_sex_hormone_report = General_Report(
    report_name = "性激素六项",
    index = [
        "黄体生成素(LH)", 
        "卵泡刺激素(FSH)", 
        "孕酮(P4)", 
        "睾酮(T)", 
        "雌二醇(E2)", 
        "催乳素(PRL)"
    ],
    keywords = [
        ["黄体生成素(LH)", "黄体生成素", "LH", "促黄体生成素", "血清促黄体生成素测定(LH)", "促黄体生成素(LH)测定", "促黄体生成素(LH)", "促黄体生成素(LH)","LH(LuteinizingHormone)","促黄体生成激素"],
        ["卵泡刺激素(FSH)", "卵泡刺激素", "FSH", "促卵泡生成素", "血清促卵泡刺激素测定(FSH)", "卵泡刺激素(FSH)测定","促卵泡生成素(FSH)","FSH(Follicle-StimulatingHormone)","血清促卵泡刺激素","卵泡生成素"],
        ["孕酮(P4)", "孕酮", "P4", "Prog", "孕酮测定", "血清孕酮(PROG)测定", "血清孕酮(PROG)", "PRGE", "孕酮(P)","Progesterone","Progesterone(P4)"],
        ["睾酮(T)", "睾酮", "Testo", "总睾酮","睾酮测定(TEST)", "血清睾酮(TESTO)测定", "血清睾酮(TESTO)", "TSTO", "TEST"],
        ["雌二醇(E2)", "雌二醇", "E2", "二羟基雌特酮", "雌二醇(E2)测定", "eE2", "雌二醇(E2)","Estradiol","Estradiol(E2)"],
        ["催乳素(PRL)", "催乳素", "PRL", "泌乳素", "垂体泌乳素", "垂体泌乳素(PRL)", "催乳素(PRL)","Prolactin","血清垂体泌乳素","泌乳素测定"]
    ],
    unit_conversions = [
        [], # 黄体生成素(LH)
        [], # 卵泡刺激素(FSH)
        [["ng/mL"], ["nmol"], [0.314]], # 孕酮(P4)
        [["ng/mL"], ["nmol", "dL"], [0.288, 0.01]], # 睾酮(T)
        [["pg/mL"], ["pmol"], [0.272]], # 雌二醇(E2)
        [["ng/mL"], ["mIU", "uIU"], [0.0472, 0.0472]] # 催乳素(PRL)
    ]
)

# 2. AMH
amh_report = General_Report(
    report_name = "amh",
    index = [
        "抗缪勒氏管激素(AMH)"
    ],
    keywords = [
        ["抗缪勒氏管激素(AMH)", "抗缪勒氏管激素", "AMH", "抗苗勒氏管激素","抗缪勒氏激素(AMH)","AMH(Anti-MullerianHormone)",
        "8抗缪勒管激素","人抗缪勒氏管激素(AMH)","抗缪勒氏管激素(AMH)","抗缪勒管激素(AMH)","抗缪勒管激素"]
    ],
    unit_conversions = [
        [["ng/mL"], ["pmol"], [0.14]] # 抗缪勒氏管激素(AMH)
    ]
)

# 3. 精子
sperm_report = General_Report(
    report_name = "精子",
    index = [
        "精液量(Volume)", 
        "液化时间(Liquefaction)", 
        "酸碱度(PH)", 
        "白细胞浓度(WBC)",
        "精子浓度(Concentration)", 
        "总精子数(Total Concentration)", 
        "精子总活力(Motility)",
        "前向精子百分率(Progressive motile)", 
        "精子正常形态率(Morphology)", 
        "血红蛋白A(HBA)", 
        "快速前向运动精子(A)",
        "快速前向运动精子(B)"
    ],
    keywords = [
        ["精液量(Volume)", "精液量", "Volume", "Vol", "精液体积"],
        ["液化时间(Liquefaction)", "液化时间", "Liquefaction",],
        ["酸碱度(PH)", "酸碱度", "PH", "pH值","pH", "PH值", "精液酸碱度"],
        ["白细胞浓度(WBC)", "白细胞浓度", "WBC", "圆细胞", "白细胞"],
        ["精子浓度(Concentration)", "精子浓度", "Concentration", "密度","精子浓度", ],
        ["总精子数(Total Concentration)", "总精子数", "Total Concentration", "精子总数", "精子总数"],
        ["精子总活力(Motility)", "精子总活力", "Motility", "PR+NP", "总活力(PR+NP)","总精子活动率","精子总活动率"],
        ["前向精子百分率(Progressive motile)", "前向精子百分率", "Progressive motile", "a+b", "前向运动", "前向运动精子百分率(PR)","前向运动率","前向运动精子(PR)"],
        ["精子正常形态率(Morphology)", "精子正常形态率", "Morphology", "正常形态率", "正常形态精子"],
        ["血红蛋白A(HBA)", "血红蛋白A", "HBA", "精子头部结合水解酶"],
        ["快速前向运动精子(A)"],
        ["快速前向运动精子(B)"]
    ],
    unit_conversions = [
        [], # 精液量(Volume)
        [], # 液化时间(Liquefaction)
        [], # 酸碱度(PH)
        [], # 白细胞浓度(WBC)
        [], # 精子浓度(Concentration)
        [], # 总精子数(Total Concentration)
        [], # 精子总活力(Motility)
        [], # 前向精子百分率(Progressive motile)
        [], # 精子正常形态率(Morphology)
        [], # 血红蛋白A(HBA)
        [], # 快速前向运动精子(A)
        []  # 快速前向运动精子(B)
    ]
)
# 4. B超（妇科）
b_ultrasound_report = General_Report(
    report_name="中文B超",
    index=[
        "子宫内膜厚度",        # [左, 右]
        "卵泡总数",            # int
        "最大卵泡尺寸",        # [max]
        "卵泡发育趋势"         # -1 / 0 / 1
    ],
    keywords=[
        [],  # 特殊提取，不参与关键词匹配
        [],
        [],
        []
    ],
    unit_conversions=[
        [],
        [],
        [],
        []
    ]
)

# 6. 免疫五项
immuno_five_report = General_Report(
    report_name = "免疫五项",
    index = [
        "免疫球蛋白G(IgG)", 
        "免疫球蛋白A(IgA)", 
        "免疫球蛋白M(IgM)", 
        "补体C3(C3)", 
        "补体C4(C4)"
    ],
    keywords = [
        ["免疫球蛋白G(IgG)", "免疫球蛋白G", "IgG"],
        ["免疫球蛋白A(IgA)", "免疫球蛋白A", "IgA"],
        ["免疫球蛋白M(IgM)", "免疫球蛋白M", "IgM"],
        ["补体C3(C3)", "补体C3", "C3","血清补体3","血清补体3[C3]"],
        ["补体C4(C4)", "补体C4", "C4","血清补体4","血清补体4[C4]"]
    ],
    unit_conversions = [
        [], # 免疫球蛋白G(IgG)
        [], # 免疫球蛋白A(IgA)
        [], # 免疫球蛋白M(IgM)
        [], # 补体C3(C3)
        []  # 补体C4(C4)
    ]
)

# 7. 凝血功能
coag_function_report = General_Report(
    report_name = "凝血功能",
    index = [
        "凝血酶原时间(PT)", 
        "凝血酶原时间比值(PT-R)", 
        "凝血酶原时间活动度(PT%)",
        "凝血酶原国际标准化比值(PT-INR)", 
        "活化部分凝血活酶时间(APTT)", 
        "凝血酶时间(TT)",
        "抗凝血酶III(AT-III)", 
        "D-二聚体(D-D)", 
        "纤维蛋白原(FIB)", 
        "国际标准化比值(INR)"
    ],
    keywords = [
        ["凝血酶原时间(PT)", "凝血酶原时间", "PT","血浆凝血酶原时间"],
        ["凝血酶原时间比值(PT-R)", "凝血酶原时间比值", "PT-R", "PTR"],
        ["凝血酶原时间活动度(PT%)", "凝血酶原时间活动度", "PT%", "PTA","凝血酶原活动度"],
        ["凝血酶原国际标准化比值(PT-INR)", "凝血酶原国际标准化比值", "PT-INR","凝血酶原时间国际标准化比值"],
        ["活化部分凝血活酶时间(APTT)", "活化部分凝血活酶时间", "APTT"],
        ["凝血酶时间(TT)", "凝血酶时间", "TT"],
        ["抗凝血酶III(AT-III)", "抗凝血酶III", "AT-III", "AT3"],
        ["D-二聚体(D-D)", "D-二聚体", "D-D", "D-Dimer","血浆D-二聚体"],
        ["纤维蛋白原(FIB)", "纤维蛋白原", "FIB", "Fbg"],
        ["国际标准化比值(INR)", "国际标准化比值", "INR"]
    ],
    unit_conversions = [
        [], # 凝血酶原时间(PT)
        [], # 凝血酶原时间比值(PT-R)
        [], # 凝血酶原时间活动度(PT%)
        [], # 凝血酶原国际标准化比值(PT-INR)
        [], # 活化部分凝血活酶时间(APTT)
        [], # 凝血酶时间(TT)
        [], # 抗凝血酶III(AT-III)
        [], # D-二聚体(D-D)
        [], # 纤维蛋白原(FIB)
        []  # 国际标准化比值(INR)
    ]
)

# 8. 肾功能
renal_function_report = General_Report(
    report_name = "肾功能",
    index = [
        "尿素(Urea)", 
        "尿酸(UA)", 
        "肌酐(Cr)", 
        "胱抑素C(CysC)", 
        "二氧化碳结合力(CO2-CP)", 
        "葡萄糖(GLU)"
    ],
    keywords = [
        ["尿素(Urea)", "尿素", "Urea", "BUN", "尿素氮"],
        ["尿酸(UA)", "尿酸", "UA", "URIC"],
        ["肌酐(Cr)", "肌酐", "Cr", "CREA", "CRE"],
        ["胱抑素C(CysC)", "胱抑素C", "CysC"],
        ["二氧化碳结合力(CO2-CP)", "二氧化碳结合力", "CO2-CP", "CO2CP","血清二氧化碳(CO2)"],
        ["葡萄糖(GLU)", "葡萄糖", "GLU", "血糖"]
    ],
    unit_conversions = [
        [], # 尿素(Urea)
        [], # 尿酸(UA)
        [], # 肌酐(Cr)
        [], # 胱抑素C(CysC)
        [], # 二氧化碳结合力(CO2-CP)
        []  # 葡萄糖(GLU)
    ]
)

# 9. 血型检测
blood_type_report = General_Report(
    report_name = "血型检测",
    index = [
        "ABO血型(ABO)", 
        "Rh血型(Rh)"
    ],
    keywords = [
        ["ABO血型(ABO)", "ABO血型", "ABO","ABO血型鉴定","ABOGroup"],
        ["Rh血型(Rh)", "Rh血型", "Rh", "Rh(D)血型","Rh(D)血型鉴定","RHGroup"]
    ],
    unit_conversions = [
        [], # ABO血型(ABO)
        []  # Rh血型(Rh)
    ]
)

# 10. 血常规
blood_routine_report = General_Report(
    report_name = "血常规",
    index = [
        "白细胞计数(WBC)", 
        "红细胞分布宽度变异系数(RDW-CV)", 
        "红细胞计数(RBC)", 
        "血小板计数(PLT)",
        "血小板分布宽度(PDW)", 
        "血小板压积(PCT)", 
        "中性粒细胞百分比(NEU%)", 
        "中性粒细胞计数(NEU#)",
        "平均血小板体积(MPV)", 
        "单核细胞计数(MON#)", 
        "单核细胞百分比(MON%)", 
        "平均红细胞体积(MCV)",
        "平均红细胞血红蛋白浓度(MCHC)", 
        "红细胞血红蛋白含量(HCH)", 
        "淋巴细胞计数(LYM#)", 
        "淋巴细胞百分比(LYM%)",
        "血红蛋白(HGB)", 
        "红细胞比容(HCT)", 
        "嗜酸性粒细胞百分比(EO%)", 
        "嗜酸性粒细胞计数(EO#)",
        "嗜碱性粒细胞计数(BAS#)", 
        "嗜碱性粒细胞百分比(BAS%)", 
        "未成熟粒细胞百分比(IG%)", 
        "未成熟粒细胞计数(IG)",
        "平均红细胞血红蛋白量(MCH)", 
        "红细胞分布宽度(RDW)", 
        "有核红细胞计数(NRBC#)", 
        "有核红细胞百分比(NRBC%)"
    ],
    keywords = [
        ["白细胞计数(WBC)", "白细胞计数", "WBC", "白细胞数目"],
        ["红细胞分布宽度变异系数(RDW-CV)", "红细胞分布宽度变异系数", "RDW-CV", "RDWCV"],
        ["红细胞计数(RBC)", "红细胞计数", "RBC", "红细胞数目"],
        ["血小板计数(PLT)", "血小板计数", "PLT", "血小板数目"],
        ["血小板分布宽度(PDW)", "血小板分布宽度", "PDW"],
        ["血小板压积(PCT)", "血小板压积", "PCT"],
        ["中性粒细胞百分比(NEU%)", "中性粒细胞百分比", "NEU%", "GR%", "中性粒细胞百分数"],
        ["中性粒细胞计数(NEU#)", "中性粒细胞计数", "NEU#", "GR#", "中性粒细胞绝对值"],
        ["平均血小板体积(MPV)", "平均血小板体积", "MPV"],
        ["单核细胞计数(MON#)", "单核细胞计数", "MON#", "MO#", "单核细胞绝对值"],
        ["单核细胞百分比(MON%)", "单核细胞百分比", "MON%", "MO%"],
        ["平均红细胞体积(MCV)", "平均红细胞体积", "MCV"],
        ["平均红细胞血红蛋白浓度(MCHC)", "平均红细胞血红蛋白浓度", "MCHC", "平均血红蛋白浓度"],
        ["红细胞血红蛋白含量(HCH)", "红细胞血红蛋白含量", "HCH", "MCH","平均红细胞血红蛋白含量", "平均血红蛋白含量"],
        ["淋巴细胞计数(LYM#)", "淋巴细胞计数", "LYM#", "淋巴细胞绝对值"],
        ["淋巴细胞百分比(LYM%)", "淋巴细胞百分比", "LYM%", "淋巴细胞百分数"],
        ["血红蛋白(HGB)", "血红蛋白", "HGB", "Hb"],
        ["红细胞比容(HCT)", "红细胞比容", "HCT", "红细胞压积","红细胞比积"],
        ["嗜酸性粒细胞百分比(EO%)", "嗜酸性粒细胞百分比", "EO%"],
        ["嗜酸性粒细胞计数(EO#)", "嗜酸性粒细胞计数", "EO#", "嗜酸性粒细胞绝对值","嗜酸粒细胞计数"],
        ["嗜碱性粒细胞计数(BAS#)", "嗜碱性粒细胞计数", "BAS#", "嗜碱性粒细胞绝对值","嗜碱粒细胞计数"],
        ["嗜碱性粒细胞百分比(BAS%)", "嗜碱性粒细胞百分比", "BAS%"],
        ["未成熟粒细胞百分比(IG%)", "未成熟粒细胞百分比", "IG%"],
        ["未成熟粒细胞计数(IG)", "未成熟粒细胞计数", "IG", "IG#"],
        ["平均红细胞血红蛋白量(MCH)", "平均红细胞血红蛋白量", "MCH"],
        ["红细胞分布宽度(RDW)", "红细胞分布宽度", "RDW", "RDW-SD"],
        ["有核红细胞计数(NRBC#)", "有核红细胞计数", "NRBC#", "NRBC"],
        ["有核红细胞百分比(NRBC%)", "有核红细胞百分比", "NRBC%"]
    ],
    unit_conversions = [
        [], # 白细胞计数(WBC)
        [], # 红细胞分布宽度变异系数(RDW-CV)
        [], # 红细胞计数(RBC)
        [], # 血小板计数(PLT)
        [], # 血小板分布宽度(PDW)
        [], # 血小板压积(PCT)
        [], # 中性粒细胞百分比(NEU%)
        [], # 中性粒细胞计数(NEU#)
        [], # 平均血小板体积(MPV)
        [], # 单核细胞计数(MON#)
        [], # 单核细胞百分比(MON%)
        [], # 平均红细胞体积(MCV)
        [], # 平均红细胞血红蛋白浓度(MCHC)
        [], # 红细胞血红蛋白含量(HCH)
        [], # 淋巴细胞计数(LYM#)
        [], # 淋巴细胞百分比(LYM%)
        [], # 血红蛋白(HGB)
        [], # 红细胞比容(HCT)
        [], # 嗜酸性粒细胞百分比(EO%)
        [], # 嗜酸性粒细胞计数(EO#)
        [], # 嗜碱性粒细胞计数(BAS#)
        [], # 嗜碱性粒细胞百分比(BAS%)
        [], # 未成熟粒细胞百分比(IG%)
        [], # 未成熟粒细胞计数(IG)
        [], # 平均红细胞血红蛋白量(MCH)
        [], # 红细胞分布宽度(RDW)
        [], # 有核红细胞计数(NRBC#)
        []  # 有核红细胞百分比(NRBC%)
    ]
)

# 11. 衣原体
ct_report = General_Report(
    report_name = "衣原体",
    index = [
        "衣原体DNA(CT-DNA)"
    ],
    keywords = [
        ["衣原体DNA(CT-DNA)", "衣原体DNA", "CT-DNA", "沙眼衣原体核酸","衣原体"]
    ],
    unit_conversions = [
        []  # 衣原体DNA(CT-DNA)
    ]
)

# 12. 传染病四项
infectious_disease_report = General_Report(
    report_name = "传染病四项",
    index = [
        "乙肝表面抗原(HBsAg)", 
        "乙肝表面抗体(HBsAb)", 
        "乙肝e抗原(HBeAg)", 
        "乙肝e抗体(HBeAb)",
        "乙肝核心抗体(HBcAb)", 
        "丙肝抗体(Anti-HCV)", 
        "艾滋抗体(Anti-HIV)", 
        "梅毒螺旋体抗体(TPAb)"
    ],
    keywords = [
        ["乙肝表面抗原(HBsAg)", "乙肝表面抗原", "HBsAg","乙型肝炎病毒表面抗原"],
        ["乙肝表面抗体(HBsAb)", "乙肝表面抗体", "HBsAb", "Anti-HBs","乙型肝炎病毒表面抗体"],
        ["乙肝e抗原(HBeAg)", "乙肝e抗原", "HBeAg","乙型肝炎病毒e抗原"],
        ["乙肝e抗体(HBeAb)", "乙肝e抗体", "HBeAb", "Anti-HBe","乙型肝炎病毒e抗体"],
        ["乙肝核心抗体(HBcAb)", "乙肝核心抗体", "HBcAb", "Anti-HBc","乙型肝炎病毒核心抗体"],
        ["丙肝抗体(Anti-HCV)", "丙肝抗体", "Anti-HCV", "HCV","AntiHCV","丙型肝炎病毒抗体"],
        ["艾滋抗体(Anti-HIV)", "艾滋抗体", "Anti-HIV", "HIV","AntiHIV","HIV抗原抗体","HIV(Ag/Ab)","人类免疫缺陷病毒抗原抗体"],
        ["梅毒螺旋体抗体(TPAb)", "梅毒螺旋体抗体", "TPAb", "Anti-TP", "梅毒特异性抗体","Syphilis(CLIA)"]
    ],
    unit_conversions = [
        [], # 乙肝表面抗原(HBsAg)
        [], # 乙肝表面抗体(HBsAb)
        [], # 乙肝e抗原(HBeAg)
        [], # 乙肝e抗体(HBeAb)
        [], # 乙肝核心抗体(HBcAb)
        [], # 丙肝抗体(Anti-HCV)
        [], # 艾滋抗体(Anti-HIV)
        []  # 梅毒螺旋体抗体(TPAb)
    ]
)

# 13. 优生五项TORCH
torch_report = General_Report(
    report_name = "优生五项TORCH",
    index = [
        "巨细胞病毒IgM(CMV-IgM)", 
        "巨细胞病毒IgG(CMV-IgG)", 
        "弓形虫IgM(TOX-IgM)", 
        "弓形虫IgG(TOX-IgG)",
        "风疹病毒IgM(RV-IgM)", 
        "风疹病毒IgG(RV-IgG)", 
        "单纯疱疹病毒1型IgM(HSV-1-IgM)", 
        "单纯疱疹病毒1型IgG(HSV-1-IgG)",
        "单纯疱疹病毒2型IgM(HSV-2-IgM)", 
        "单纯疱疹病毒2型IgG(HSV-2-IgG)", 
        "B19细小病毒IgM(B19-IgM)", 
        "B19细小病毒IgG(B19-IgG)"
    ],
    keywords = [
        ["巨细胞病毒IgM(CMV-IgM)", "巨细胞病毒IgM", "CMV-IgM"],
        ["巨细胞病毒IgG(CMV-IgG)", "巨细胞病毒IgG", "CMV-IgG"],
        ["弓形虫IgM(TOX-IgM)", "弓形虫IgM", "TOX-IgM"],
        ["弓形虫IgG(TOX-IgG)", "弓形虫IgG", "TOX-IgG"],
        ["风疹病毒IgM(RV-IgM)", "风疹病毒IgM", "RV-IgM", "Rubella-IgM"],
        ["风疹病毒IgG(RV-IgG)", "风疹病毒IgG", "RV-IgG", "Rubella-IgG"],
        ["单纯疱疹病毒1型IgM(HSV-1-IgM)", "单纯疱疹病毒1型IgM", "HSV-1-IgM", "HSV I IgM"],
        ["单纯疱疹病毒1型IgG(HSV-1-IgG)", "单纯疱疹病毒1型IgG", "HSV-1-IgG", "HSV I IgG"],
        ["单纯疱疹病毒2型IgM(HSV-2-IgM)", "单纯疱疹病毒2型IgM", "HSV-2-IgM", "HSV II IgM"],
        ["单纯疱疹病毒2型IgG(HSV-2-IgG)", "单纯疱疹病毒2型IgG", "HSV-2-IgG", "HSV II IgG"],
        ["B19细小病毒IgM(B19-IgM)", "B19细小病毒IgM", "B19-IgM"],
        ["B19细小病毒IgG(B19-IgG)", "B19细小病毒IgG", "B19-IgG"]
    ],
    unit_conversions = [
        [], # 巨细胞病毒IgM(CMV-IgM)
        [], # 巨细胞病毒IgG(CMV-IgG)
        [], # 弓形虫IgM(TOX-IgM)
        [], # 弓形虫IgG(TOX-IgG)
        [], # 风疹病毒IgM(RV-IgM)
        [], # 风疹病毒IgG(RV-IgG)
        [], # 单纯疱疹病毒1型IgM(HSV-1-IgM)
        [], # 单纯疱疹病毒1型IgG(HSV-1-IgG)
        [], # 单纯疱疹病毒2型IgM(HSV-2-IgM)
        [], # 单纯疱疹病毒2型IgG(HSV-2-IgG)
        [], # B19细小病毒IgM(B19-IgM)
        []  # B19细小病毒IgG(B19-IgG)
    ]
)

# 14. 支原体
mycoplasma_report = General_Report(
    report_name = "支原体",
    index = [
        "解脲支原体(Uu)", 
        "人型支原体(Mh)"
    ],
    keywords = [
        ["解脲支原体(Uu)", "解脲支原体", "Uu", "解脲脲原体"],
        ["人型支原体(Mh)", "人型支原体", "Mh"]
    ],
    unit_conversions = [
        [], # 解脲支原体(Uu)
        []  # 人型支原体(Mh)
    ]
)

# 15. HCG
hcg_pregnancy_report = General_Report(
    report_name = "HCG妊娠诊断报告",
    index = [
        "绒毛膜促性腺激素(HCG)"
    ],
    keywords = [
        ["绒毛膜促性腺激素(HCG)", "HCG", "绒毛膜促性腺激素", "β-HCG", "人绒毛膜促性腺激素","绒毛膜促性腺激素"]
    ],
    unit_conversions = [
        []  # 绒毛膜促性腺激素(HCG)
    ]
)

# 16. 地中海贫血
thalassemia_report = General_Report(
    report_name = "地中海贫血症",
    index = [
        "a-地贫基因检测(3种缺失型)", 
        "a-地贫基因检测(3种非缺失型)", 
        "β-地贫基因检测(17种突变)"
    ],
    keywords = [
        ["a-地贫基因检测(3种缺失型)", "a-地贫缺失型", "α-地贫缺失型","α-地贫基因检测(3种缺失型)"],
        ["a-地贫基因检测(3种非缺失型)", "a-地贫非缺失型", "α-地贫突变型","α-地贫基因检测(3种非缺失型)"],
        ["β-地贫基因检测(17种突变)", "β-地贫基因检测", "β-地贫突变"]
    ],
    unit_conversions = [
        [], # a-地贫基因检测(3种缺失型)
        [], # a-地贫基因检测(3种非缺失型)
        []  # β-地贫基因检测(17种突变)
    ]
)

# 17. 贫血四项
anemia_four_report = General_Report(
    report_name = "贫血四项",
    index = [
        "铁蛋白(Fer)", 
        "叶酸(Folate)", 
        "维生素B12(VitB12)", 
        "转铁蛋白(TRF)"
    ],
    keywords = [
        ["铁蛋白(Fer)", "铁蛋白", "Fer", "Ferritin"],
        ["叶酸(Folate)", "叶酸", "Folate", "FA"],
        ["维生素B12(VitB12)", "维生素B12", "VitB12", "VB12"],
        ["转铁蛋白(TRF)", "转铁蛋白", "TRF", "Tf"]
    ],
    unit_conversions = [
        [], # 铁蛋白(Fer)
        [], # 叶酸(Folate)
        [], # 维生素B12(VitB12)
        []  # 转铁蛋白(TRF)
    ]
)

# 18. 肝功五项
liver_function_report = General_Report(
    report_name = "肝功五项",
    index = [
        "白蛋白(ALB)", 
        "丙氨酸氨基转移酶(ALT)", 
        "天门冬氨酸氨基转移酶(AST)", 
        "天门冬氨酸/丙氨酸氨基转移酶比值(AST/ALT)",
        "总胆红素(T-BiL)", 
        "直接胆红素(D-BiL)", 
        "间接胆红素(I-BiL)", 
        "谷氨酰转肽酶(GGT)",
        "总蛋白(TP)", 
        "球蛋白(GLO)", 
        "白蛋白/球蛋白比值(A/G)", 
        "碱性磷酸酶(ALP)",
        "胆碱脂酶(ChE)", 
        "α-L-岩藻糖苷酶(AFU)", 
        "腺苷脱氨酶(ADA)", 
        "总胆汁酸(TBA)",
        "脂蛋白(a)"
    ],
    keywords = [
        ["白蛋白(ALB)", "白蛋白", "ALB"],
        ["丙氨酸氨基转移酶(ALT)", "丙氨酸氨基转移酶", "ALT", "谷丙转氨酶", "GPT"],
        ["天门冬氨酸氨基转移酶(AST)", "天门冬氨酸氨基转移酶", "AST", "谷草转氨酶", "GOT"],
        ["天门冬氨酸/丙氨酸氨基转移酶比值(AST/ALT)", "AST/ALT", "谷草/谷丙", "AST/ALT比值"],
        ["总胆红素(T-BiL)", "总胆红素", "T-BiL", "TBIL", "STB"],
        ["直接胆红素(D-BiL)", "直接胆红素", "D-BiL", "DBIL", "SDB"],
        ["间接胆红素(I-BiL)", "间接胆红素", "I-BiL", "IBIL", "SIB"],
        ["谷氨酰转肽酶(GGT)", "谷氨酰转肽酶", "GGT", "r-GT", "γ-GT","γ-谷氨酰转肽酶","γ-谷氨酰基转移酶","γ谷氨酰转移酶","γ-谷氨酰转移酶"],
        ["总蛋白(TP)", "总蛋白", "TP"],
        ["球蛋白(GLO)", "球蛋白", "GLO", "GLB"],
        ["白蛋白/球蛋白比值(A/G)", "白蛋白/球蛋白比值", "A/G", "A/G比值","白蛋白/球蛋白","白球比例","白蛋白/球蛋白(A/G)"],
        ["碱性磷酸酶(ALP)", "碱性磷酸酶", "ALP", "AKP"],
        ["胆碱脂酶(ChE)", "胆碱脂酶", "ChE", "CHE"],
        ["α-L-岩藻糖苷酶(AFU)", "α-L-岩藻糖苷酶", "AFU"],
        ["腺苷脱氨酶(ADA)", "腺苷脱氨酶", "ADA"],
        ["总胆汁酸(TBA)", "总胆汁酸", "TBA"],
        ["脂蛋白(a)"]
    ],
    unit_conversions = [
        [], # 白蛋白(ALB)
        [], # 丙氨酸氨基转移酶(ALT)
        [], # 天门冬氨酸氨基转移酶(AST)
        [], # 天门冬氨酸/丙氨酸氨基转移酶比值(AST/ALT)
        [], # 总胆红素(T-BiL)
        [], # 直接胆红素(D-BiL)
        [], # 间接胆红素(I-BiL)
        [], # 谷氨酰转肽酶(GGT)
        [], # 总蛋白(TP)
        [], # 球蛋白(GLO)
        [], # 白蛋白/球蛋白比值(A/G)
        [], # 碱性磷酸酶(ALP)
        [], # 胆碱脂酶(ChE)
        [], # α-L-岩藻糖苷酶(AFU)
        [], # 腺苷脱氨酶(ADA)
        [], # 总胆汁酸(TBA)
        [], # 脂蛋白(a)
    ]
)

# 19. 甲功
thyroid_function_report = General_Report(
    report_name = "甲功",
    index = [
        "促甲状腺激素(TSH)", 
        "总三碘甲状腺原氨酸(TT3)", 
        "总甲状腺素(TT4)", 
        "游离三碘甲状腺原氨酸(FT3)",
        "游离甲状腺素(FT4)", 
        "甲状腺过氧化物酶抗体(TPOAb)", 
        "甲状腺球蛋白抗体(TGAb)", 
        "促甲状腺激素受体抗体(TSHRAb)"
    ],
    keywords = [
        ["促甲状腺激素(TSH)", "促甲状腺激素", "TSH", "促甲状腺刺激激素(TSH)","促甲状腺激素TSH","促甲状腺素定量","促甲状腺素定量"],
        ["总三碘甲状腺原氨酸(TT3)", "总三碘甲状腺原氨酸", "TT3", "三碘甲状腺原氨酸","三碘甲状腺原氨酸定量"],
        ["总甲状腺素(TT4)", "总甲状腺素", "TT4", "甲状腺素","甲状腺激素定量"],
        ["游离三碘甲状腺原氨酸(FT3)", "游离三碘甲状腺原氨酸", "FT3","游离三碘甲状腺原氨酸定量"],
        ["游离甲状腺素(FT4)", "游离甲状腺素", "FT4","游离甲状腺激素定量"],
        ["甲状腺过氧化物酶抗体(TPOAb)", "甲状腺过氧化物酶抗体", "TPOAb", "TPO-Ab"],
        ["甲状腺球蛋白抗体(TGAb)", "甲状腺球蛋白抗体", "TGAb", "TG-Ab"],
        ["促甲状腺激素受体抗体(TSHRAb)", "促甲状腺激素受体抗体", "TSHRAb", "TRAb"]
    ],
    unit_conversions = [
        [], # 促甲状腺激素(TSH)
        [], # 总三碘甲状腺原氨酸(TT3)
        [], # 总甲状腺素(TT4)
        [], # 游离三碘甲状腺原氨酸(FT3)
        [], # 游离甲状腺素(FT4)
        [], # 甲状腺过氧化物酶抗体(TPOAb)
        [], # 甲状腺球蛋白抗体(TGAb)
        []  # 促甲状腺激素受体抗体(TSHRAb)
    ]
)

# 20. 孕前基础健康
preconception_basic_health_report = General_Report(
    report_name = "孕前基础健康评估报告",
    index = [
        "25-羟维生素D(25-OH-VD)"
    ],
    keywords = [
        ["25-羟维生素D(25-OH-VD)", "25-羟维生素D", "25-OH-VD", "维生素D","25OH维生素D","25羟基维生素D","α25羟基维生素D"]
    ],
    unit_conversions = [
        []  # 25-羟维生素D(25-OH-VD)
    ]
)

# 21. 尿常规
urine_routine_report = General_Report(
    report_name = "尿常规",
    index = [
        "尿蛋白(PRO)", 
        "尿糖(GLU)", 
        "酮体(KET)", 
        "尿胆红素(BIL)", 
        "尿胆原(URO)", 
        "亚硝酸盐(NIT)"
    ],
    keywords = [
        ["尿蛋白(PRO)", "尿蛋白", "PRO"],
        ["尿糖(GLU)", "尿糖", "GLU"],
        ["酮体(KET)", "酮体", "KET"],
        ["尿胆红素(BIL)", "尿胆红素", "BIL"],
        ["尿胆原(URO)", "尿胆原", "URO"],
        ["亚硝酸盐(NIT)", "亚硝酸盐", "NIT"]
    ],
    unit_conversions = [
        [], # 尿蛋白(PRO)
        [], # 尿糖(GLU)
        [], # 酮体(KET)
        [], # 尿胆红素(BIL)
        [], # 尿胆原(URO)
        []  # 亚硝酸盐(NIT)
    ]
)

# 22. 核医学
nuclear_medicine_report = General_Report(
    report_name = "核医学",
    index = [
        "糖类抗原19-9(CA199)"
    ],
    keywords = [
        ["糖类抗原19-9(CA199)", "糖类抗原19-9", "CA199"]
    ],
    unit_conversions = [
        []  # 糖类抗原19-9(CA199)
    ]
)

# 23. 结核感染T细胞检测
tb_tcell_report = General_Report(
    report_name = "结核感染T细胞检测",
    index = [
        "IFN-(N)", 
        "结核感染T细胞检测", 
        "IFN-Y(T)", 
        "IFN-V(T-N)"
    ],
    keywords = [
        ["IFN-(N)", "IFN-N", "阴性对照", "N"],
        ["结核感染T细胞检测", "结核感染T细胞", "TB-IGRA"],
        ["IFN-Y(T)", "IFN-T", "抗原A", "T"],
        ["IFN-V(T-N)", "IFN-T-N", "T-N"]
    ],
    unit_conversions = [
        [], # IFN-（N）
        [], # 结核感染T细胞检测
        [], # IFN-Y（T）
        []  # IFN-V（T-N）
    ]
)

# 24. RF分型
rf_typing_report = General_Report(
    report_name = "RF分型(类风湿因子)",
    index = [
        "类风湿因子IgA", 
        "类风湿因子IgG", 
        "类风湿因子IgM"
    ],
    keywords = [
        ["类风湿因子IgA", "RF-IgA","类风湿因子"],
        ["类风湿因子IgG", "RF-IgG","类风湿因子"],
        ["类风湿因子IgM", "RF-IgM","类风湿因子"]
    ],
    unit_conversions = [
        [], # 类风湿因子IgA
        [], # 类风湿因子IgG
        []  # 类风湿因子IgM
    ]
)

# 25. 血脂
blood_lipid_report = General_Report(
    report_name = "血脂",
    index = [
        "总胆固醇(TC)", 
        "甘油三酯(TG)", 
        "低密度脂蛋白胆固醇(LDL-C)", 
        "高密度脂蛋白胆固醇(HDL-C)"
    ],
    keywords = [
        ["总胆固醇(TC)", "总胆固醇", "TC", "CHOL"],
        ["甘油三酯(TG)", "甘油三酯", "TG", "TRIG"],
        ["低密度脂蛋白胆固醇(LDL-C)", "低密度脂蛋白胆固醇", "LDL-C", "LDL", "低密度脂蛋白"],
        ["高密度脂蛋白胆固醇(HDL-C)", "高密度脂蛋白胆固醇", "HDL-C", "HDL", "高密度脂蛋白"]
    ],
    unit_conversions = [
        [], # 总胆固醇(TC)
        [], # 甘油三酯(TG)
        [], # 低密度脂蛋白胆固醇(LDL-C)
        []  # 高密度脂蛋白胆固醇(HDL-C)
    ]
)

# 26. 血糖
blood_glucose_report = General_Report(
    report_name = "血糖",
    index = [
        "空腹血糖(FPG)", 
        "餐后2小时血糖(2hPG)", 
        "糖化血红蛋白(HbA1c)"
    ],
    keywords = [
        ["空腹血糖(FPG)", "空腹血糖", "FPG", "葡萄糖"],
        ["餐后2小时血糖(2hPG)", "餐后2小时血糖", "2hPG","餐后两小时血糖"],
        ["糖化血红蛋白(HbA1c)", "糖化血红蛋白", "HbA1c"]
    ],
    unit_conversions = [
        [], # 空腹血糖(FPG)
        [], # 餐后2小时血糖(2hPG)
        []  # 糖化血红蛋白(HbA1c)
    ]
)

# 27. 同型半胱氨酸
homocysteine_report = General_Report(
    report_name = "同型半胱氨酸报告",
    index = [
        "同型半胱氨酸(HCY)"
    ],
    keywords = [
        ["同型半胱氨酸(HCY)", "同型半胱氨酸", "HCY"]
    ],
    unit_conversions = [
        []  # 同型半胱氨酸(HCY)
    ]
)

# 28. TCT
tct_report = General_Report(
    report_name = "TCT",
    index = [
        "未见上皮内病变或恶性细胞(tct-NILM)", 
        "炎症细胞改变(tct-Inflammatory cell changes)", 
        "意义不明的非典型鳞状细胞(ASC-US)", 
        "低度鳞状上皮内病变(LSIL)", 
        "高度鳞状上皮内病变(HSIL)", 
        "鳞状细胞癌/腺细胞癌(squamous_cell_carcinoma)"
    ],
    keywords = [
        ["NILM", "未见上皮内病变", "无上皮内病变"],
        ["炎症反应性细胞改变", "炎症", "轻度炎症"],
        ["ASC-US", "意义不明确的非典型鳞状细胞", "ASCUS"],
        ["LSIL", "低度鳞状上皮内病变"],
        ["HSIL", "高度鳞状上皮内病变"],
        ["鳞状细胞癌 / 腺细胞癌", "鳞状细胞癌", "SCC", "腺癌"]
    ],
    unit_conversions = [
        [], # NILM
        [], # 炎症反应性细胞改变
        [], # ASC-US
        [], # LSIL
        [], # HSIL
        []  # 鳞状细胞癌 / 腺细胞癌
    ]
)

# 29. Y染色体微缺失
y_microdeletion_report = General_Report(
    report_name = "Y - 染色体微缺失",
    index = [
        "Y-染色体微缺失(sY84)", 
        "Y-染色体微缺失(sY86)", 
        "Y-染色体微缺失(sY127)",
        "Y-染色体微缺失(sY134)", 
        "Y-染色体微缺失(sY254)", 
        "Y-染色体微缺失(sY255)"
    ],
    keywords = [
        ["Y-染色体微缺失(sY84)", "sY84"],
        ["Y-染色体微缺失(sY86)", "sY86"],
        ["Y-染色体微缺失(sY127)", "sY127"],
        ["Y-染色体微缺失(sY134)", "sY134"],
        ["Y-染色体微缺失(sY254)", "sY254", "AZFcSY254"],
        ["Y-染色体微缺失(sY255)", "sY255", "AZFcSY255"]
    ],
    unit_conversions = [
        [], # Y-染色体微缺失(sY84)
        [], # Y-染色体微缺失(sY86)
        [], # Y-染色体微缺失(sY127)
        [], # Y-染色体微缺失(sY134)
        [], # Y-染色体微缺失(sY254)
        []  # Y-染色体微缺失(sY255)
    ]
)

# 30. 狼疮
lupus_report = General_Report(
    report_name = "狼疮",
    index = [
        "狼疮抗凝物初筛试验1(LA1)", 
        "狼疮抗凝物确定试验(LA2)", 
        "狼疮初筛/狼疮确定(LA1/LA2)"
    ],
    keywords = [
        ["LA1", "狼疮抗凝物筛选", "DRVVT Screen","dRVVT-S"],
        ["LA2", "狼疮抗凝物确证", "DRVVT Confirm","dRVVT-C"],
        ["LA1/LA2", "比值", "Ratio", "S/C","dRVVT-R"]
    ],
    unit_conversions = [
        [], # LA1
        [], # LA2
        []  # LA1/LA2
    ]
)

# 31. 白带常规
leukorrhea_routine_report = General_Report(
    report_name = "白带常规",
    index = [
        "阴道清洁度(Cleanliness)",
        "白细胞(WBC)",
        "红细胞(RBC)",
        "滴虫(TV)", 
        "霉菌(FV)", 
        "细菌性阴道病(BV)", 
        "酸碱度(pH)",
        "线索细胞(Clue Cells)",
        "过氧化氢(H2O2)",
        "β-葡萄糖醛酸酶(GUS)",
        "唾液酸苷酶(SNA)",
        "乙酰氨基葡萄糖苷酶(NAG)",
        "白细胞酯酶(LE)"
    ],
    keywords = [
        ["阴道清洁度(Cleanliness)", "阴道清洁度", "清洁度", "Cleanliness"], # 阴道清洁度
        ["白细胞(WBC)", "白细胞", "WBC", "白血球"], # 白细胞
        ["红细胞(RBC)", "红细胞", "RBC", "红血球"], # 红细胞
        ["滴虫(TV)", "滴虫", "TV", "阴道毛滴虫", "滴虫感染提示"], # 滴虫
        ["霉菌(FV)", "霉菌", "FV", "念珠菌", "假丝酵母菌", "真菌", "孢子", "菌丝", "霉菌感染提示"], # 霉菌
        ["细菌性阴道病(BV)", "细菌性阴道病", "BV", "加德纳菌"], # 细菌性阴道病
        ["酸碱度(pH)", "酸碱度", "pH", "pH值"], # 酸碱度
        ["线索细胞(Clue Cells)", "线索细胞", "Clue Cell", "Clue Cells"], # 线索细胞
        ["过氧化氢(H2O2)", "过氧化氢", "H2O2", "双氧水"], # 过氧化氢
        ["β-葡萄糖醛酸酶(GUS)", "β-葡萄糖醛酸酶", "β-G", "Beta-glucuronidase"], # β-葡萄糖醛酸酶
        ["唾液酸苷酶(SNA)", "唾液酸苷酶", "SNA", "神经氨酸酶", "Sialidase"], # 唾液酸苷酶
        ["乙酰氨基葡萄糖苷酶(NAG)", "乙酰氨基葡萄糖苷酶", "NAG", "N-乙酰-β-D-氨基葡萄糖苷酶"], # 乙酰氨基葡萄糖苷酶
        ["白细胞酯酶(LE)", "白细胞酯酶", "LE", "Leukocyte Esterase"] # 白细胞酯酶
    ],
    unit_conversions = [
        [], # 阴道清洁度
        [], # 白细胞
        [], # 红细胞
        [], # 滴虫
        [], # 霉菌
        [], # 细菌性阴道病
        [], # 酸碱度
        [], # 线索细胞
        [], # 过氧化氢
        [], # β-葡萄糖醛酸酶
        [], # 唾液酸苷酶
        [], # 乙酰氨基葡萄糖苷酶
        []  # 白细胞酯酶
    ]
)

# 32. 肿瘤标记物
tumor_marker_report = General_Report(
    report_name = "肿瘤标记物",
    index = [
        "糖类抗原125(CA125)", 
        "甲胎蛋白(AFP)", 
        "癌胚抗原(CEA)"
    ],
    keywords = [
        ["糖类抗原125(CA125)", "糖类抗原125", "CA125","糖链抗原125(CA125)"],
        ["甲胎蛋白(AFP)", "甲胎蛋白", "AFP"],
        ["癌胚抗原(CEA)", "癌胚抗原", "CEA"]
    ],
    unit_conversions = [
        [], # 糖类抗原125(CA125)
        [], # 甲胎蛋白(AFP)
        []  # 癌胚抗原(CEA)
    ]
)

# 33. 淋球菌
neisseria_gonorrhoeae_culture_report = General_Report(
    report_name = "淋球菌",
    index = ["淋球菌培养"],
    keywords = [
        [],
    ],
    unit_conversions = [
        [], 
    ]
)

# 34. 精子线粒体膜电位检测 
sperm_mitochondrial_membrane_potential_report = General_Report(
    report_name = "精子线粒体膜电位检测",
    index = ["精子线粒体膜电位(MMP)"],
    keywords = [
        ["精子线粒体膜电位(MMP)","MMP","精子线粒体膜电位"],
    ],
    unit_conversions = [
        [], 
    ]
)

# 35. DNA碎片化指数
dna_fragmentation_index_report = General_Report(
    report_name = "DNA碎片化指数",
    index = [
        "DNA碎片化指数(DFI)"
    ],
    keywords = [
        ["DNA碎片化指数(DFI)", "DNA碎片化指数", "DFI", "精子DNA碎片", "DNA fragmentation Index","精子DNA碎片指数(DFI)","碎片分析检测值","精子DNA碎片指数检测(SDFI)"]
    ],
    unit_conversions = [
        []  # DNA碎片化指数(DFI)
    ]
)

# 999. 泰国B超（虚空占位）
ultrasound_tai_report = General_Report(
    report_name="中文B超",
    index=[
        "子宫内膜厚度",        # [左, 右]
        "卵泡总数",            # int
        "最大卵泡尺寸",        # [max]
        "卵泡发育趋势"         # -1 / 0 / 1
    ],
    keywords=[
        [],  # 特殊提取，不参与关键词匹配
        [],
        [],
        []
    ],
    unit_conversions=[
        [],
        [],
        [],
        []
    ]
)



# 注册表：ID -> ReportObject
REPORT_REGISTRY = {
    1: six_sex_hormone_report,
    2: amh_report,
    3: sperm_report,
    4: b_ultrasound_report,
    6: immuno_five_report,
    7: coag_function_report,
    8: renal_function_report,
    9: blood_type_report,
    10: blood_routine_report,
    11: ct_report,
    12: infectious_disease_report,
    13: torch_report,
    14: mycoplasma_report,
    15: hcg_pregnancy_report,
    16: thalassemia_report,
    17: anemia_four_report,
    18: liver_function_report,
    19: thyroid_function_report,
    20: preconception_basic_health_report,
    21: urine_routine_report,
    22: nuclear_medicine_report,
    23: tb_tcell_report,
    24: rf_typing_report,
    25: blood_lipid_report,
    26: blood_glucose_report,
    27: homocysteine_report,
    28: tct_report,
    29: y_microdeletion_report,
    30: lupus_report,
    31: leukorrhea_routine_report,
    32: tumor_marker_report,
    33: neisseria_gonorrhoeae_culture_report,
    34: sperm_mitochondrial_membrane_potential_report,
    35: dna_fragmentation_index_report,
    999: ultrasound_tai_report
}
