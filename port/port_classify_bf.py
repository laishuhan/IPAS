import os
import re
import time
import uuid
import json
import threading
import requests
import tempfile
import errno
import hashlib
import cv2
import traceback
import logging
import pathlib
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from pdf2image import convert_from_path
from collections import Counter
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR
from datetime import datetime
from pyzbar import pyzbar
from port.pic_correction import correct_document
from yolo_xincc3.yolo_bf import yolo_cls
#from yolo_predict import run_yolo_inference
#gunicorn -w 10 --timeout 600 -b 0.0.0.0:8080 port_classify_bf:app
#gunicorn -w 1 -k gthread --threads 4 --timeout 600 -b 0.0.0.0:8080 port_classify_bf_time_b_tes2:app

# TESSDATA_PREFIX=/usr/share gunicorn -w 5 --timeout 600 -b 0.0.0.0:8080 port_classify_bf_tes2_85:app
# TESSDATA_PREFIX=/usr/share gunicorn --chdir /root/md_ocr_test2 -w 5 --timeout 600 -b 0.0.0.0:8080 port_classify_bf_tes2_85:app
# TESSDATA_PREFIX=/usr/share gunicorn -w 3 --timeout 600 -b 0.0.0.0:8080 port.port_classify_bf_tes2_85:app
# gunicorn --chdir /home/work/md_ocr_test2 -w 1 --timeout 600 -b 0.0.0.0:8080 --capture-output port_classify_bf_tes2_1128_30:app
# TESSDATA_PREFIX=/usr/share/tesseract/tessdata gunicorn --chdir /home/work/md_ocr_test2 -w 1 --timeout 600 -b 0.0.0.0:8080 --capture-output --log-file /home/work/md_ocr_test2/logs/server.log port_classify_bf_tes2_1128_30:app
#cd /home/work/md_ocr_test2
#python3.9 -m port.port_classify_bf_tes2_1017

base_path = r"/home/work/md_ocr_test2"

# 目录配置
output_base = os.path.join(base_path, "output_bf_test")
os.makedirs(output_base, exist_ok=True)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.pdf'}
app = Flask(__name__)
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=True, # 通过 use_doc_orientation_classify 参数指定使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=True, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
)

# ocr = PaddleOCR(lang="en") # 通过 lang 参数来使用英文模型
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # 通过 ocr_version 参数来使用 PP-OCR 其他版本
# ocr = PaddleOCR(device="gpu") # 通过 device 参数使得在模型推理时使用 GPU
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_server_det",
#     text_recognition_model_name="PP-OCRv5_server_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # 更换 PP-OCRv5_server 模型

Image.MAX_IMAGE_PIXELS = 400000000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("/home/work/md_ocr_test2/logs/server.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 分类规则
category_rules = {
    'china': {
        'sex_hormone': {
            'keywords': ['雌二醇', '睾酮', '黄体生成素','黄体生成激素', '卵泡刺激素', '促卵泡生成素','促卵泡成熟激素','泌乳素', '催乳素', '孕酮','孕测定',
                         'estradiol', 'testosterone', 'prolactin', 'progesterone', 'E2'],
            'must_have': []
        },
        'sperm_status': {
            'keywords': ['精液量', '液化时间', '精子浓度', '精子数',
                         '精子总活力', '前向运动精子百分率', '精子正常形态率'],
            'must_have': []
        },
        'amh': {
            'keywords': ['抗缪勒氏激素', '抗缪勒管激' ,'抗苗勒氏管激素', '抗缪勒氏管激素', '抗缪勒管素激素', 'AMH', '抗苗勒氏'],
            'must_have': []
        },
        'ultrasound': {
            'keywords': ['彩色超声检查', '彩超检查','彩色超声','多普勒超声','超声检查','超声所见'],
            'must_have': []
        },
        'immuno_five': {
            'keywords': ['免疫球蛋白IgG', '免疫球蛋白IgA','免疫球蛋白IgM','免疫球蛋白G', '免疫球蛋白A','免疫球蛋白M','补体C3','补体C4','补体3','补体4'],
            'must_have': []
        },
        'coag_function': {
            'keywords': ['凝血酶原时间','活化部分凝血活酶时间','凝血酶时间测定','二聚体','血浆纤维蛋白原'
                            ],
            'must_have': []
        },
        'renal_function': {
            'keywords': ['尿素', '尿酸', '葡萄糖'
                            ],
            'must_have': []
        },
        'blood_type': {
            'keywords': ['ABO血型鉴定', 'Rh(D)血型鉴定'
                            ],
            'must_have': []
        },
        'blood_routine': {
            'keywords': ['血小板', '血小板比积', '平均血小板体积','血红蛋白','中性粒','红细胞压积'
                            ],
            'must_have': []
        },
        'ct_dna': {
            'keywords': ['衣原体'
                            ],
            'must_have': []
        },
        'infectious_disease': {
            'keywords': ['乙肝表面抗原', '乙肝表面抗体', '乙型肝炎','乙肝e抗原','HCV抗体','HIV抗体','HIV抗原','梅毒螺旋体抗体','人类免疫缺陷病毒'
                            ],
            'must_have': []
        },
        'torch': {
            'keywords': ['巨细胞病毒','弓形体','弓形虫','风疹病毒','单纯疱疹'
                            ],
            'must_have': []
        },
        'mycoplasma': {
            'keywords': ['支原体' 
                            ],
            'must_have': []
        },        
        'hcg_pregnancy': {
            'keywords': ['人体绒毛促性腺激素', '人绒毛膜促性腺激素', '绒毛膜促性腺激素' 
                            ],
            'must_have': []
        },        
        'thalassemia': {
            'keywords': ['地贫基因检测', '地中海'
                            ],
            'must_have': []
        },        
        'anemia_four': {
            'keywords': ['铁蛋白', '叶酸(化学发光)', '维生素B12' , '血清转铁蛋白' 
                             ],
            'must_have': []
        },       
        'liver_function': {
            'keywords': ['白蛋白', '谷丙转氨酶','谷草转氨酶','总胆红素','总胆红素*','直接胆红素','转移酶'
                            ],
            'must_have': []
        },        
        'thyroid_function': {
            'keywords': ['促甲状腺激', '三碘甲状腺原氨酸','甲状腺素','游离三碘甲状腺原氨酸','游离甲状腺素'
                            ],
            'must_have': []
        },       
        'preconception_health': {
            'keywords': ['25 羟维生素 D','25-OH','基维生素'
                            ],
            'must_have': []
        },        
        'urine_routine': {
            'keywords': ['尿蛋白','尿胆原','尿胆红素'
                            ],
            'must_have': []
        },            
        'nuclear_medicine': {
            'keywords': ['糖类抗原'
                            ],
            'must_have': []
        },            
        'tb_tcell': {
            'keywords': ['阴性对照管', '结核感染T细胞检测','测试管','测试管-阴性对照管'
                            ],
            'must_have': []
        },               
        'rf_typing': {
            'keywords': ['类风湿因子'
                            ],
            'must_have': []
        },             
        'blood_lipid': {
            'keywords': ['总胆固醇', '甘油三酯'
                            ],
            'must_have': []
        },                
        'blood_glucose': {
            'keywords': ['空腹血糖', '餐后二小时血糖'
                            ],
            'must_have': []
        },              
        'homocysteine': {
            'keywords': ['同型半胱氨酸' , '同型半胱氨'
                            ],
            'must_have': []
        },              
        'tct': {
            'keywords': ['未见上皮内病变或恶性细胞', '炎症反应性细胞改变','意义不明的非典型鳞状细胞','低度鳞状上皮内病变','高度鳞状上皮内病变'
                            ],
            'must_have': []
        },                           
        'y_microdeletion': {
            'keywords': ['sY84', 'sY86','sY127','sY134','sY254','sY255'
                            ],
            'must_have': []
        },            
        'lupus': {
            'keywords': ['狼疮'
                            ],
            'must_have': []  
        },         
        'leukorrhea_routine_report': {
            'keywords': ['阴道清洁度','霉菌','细菌性阴道病'
                            ],
            'must_have': []
        },       
        'neisseria_gonorrhoeae_culture': {
            'keywords': ['未生长淋病'
                            ],
            'must_have': []
        },      
        'tumor_marker_report': {
            'keywords': ['糖链抗原', '甲胎蛋白', '癌胚抗原'
                            ],
            'must_have': []
        },        
        'dna_fragmentation_index': {
            'keywords': ['DFI'
                            ],
            'must_have': []
        },        
        'membrane_potential': {
            'keywords': ['线粒体膜电位'
                            ],
            'must_have': []
        }         
    },
    'thailand': {
        'sex_hormone': {
            'keywords': ['estradiol', 'testosterone', 'lh', 'fsh', 'prolactin', 'progesterone','P4'],
            'must_have': []
        },
        'sperm_status': {
            'keywords': ['volume', 'liquefaction', 'concentration',
                         'total concentration','motility','progressive motile','morphology','dna fragmentation index'],
            'must_have': []
        },
        'amh': {
            'keywords': ['amh', 'anti-mullerian hormone', 'antimullerian hormone','antimullerian'],
            'must_have': []
        },
        'ultrasound_tai': {
            'keywords': ['ultrasound', 'sheet', 'examdate', 'exam date'],
            'must_have': []
        },
        # 'immuno_five': {
        #     'keywords': ['IgG','IgA','IgM','C3','C4'],
        #     'must_have': ['result','results']
        # },
        # 'coag_function': {
        #     'keywords': [
        #                     'PT', 'PT-R','PT%','PT-INR','APTT','TT','AT-III','D-D','FIB'],
        #     'must_have': ['result','results']
        # },
        # 'renal_function': {
        #     'keywords': [
        #                     'Urea', 'UA','CYSC','CO2','GLU'],
        #     'must_have': ['result','results']
        # },
        'blood_type': {
            'keywords': [
                        'ABO Group', 'RH Group'],
            'must_have': []
        },
        # 'blood_routine': {
        #     'keywords': [
        #                     'WBC', 'RDW-CV','RBC','PLT','PDW','PCT','NEU%','NEU#','MPV','MON#','MON%','MCV','MCHC','HCH','LYM#','LYM%','HGB','HCT','EO%','EO#','BAS#','BAS%'],
        #     'must_have': ['result','results']
        # },
        # 'ct_dna': {
        #     'keywords': [ 
        #                     'CT-DNA'],
        #     'must_have': ['result','results']
        # },
        'infectious_disease': {
            'keywords': ['HBsAg', 'HBsAb','HBeAg','Anti-HCV','Anti-HIV','TPAb'],
            'must_have': []
        },
        # 'torch': {
        #     'keywords': [
        #                     'CMV-IgM', 'CMV-IgG','TOX-IgM','TOX-IgG','RV-IgM','RV-IgG','HSV-1-IgM','HSV-1-IgG','HSV-2-IgM','HSV-2-IgG','B19-IgM','B19-IgG'],
        #     'must_have': ['result','results']
        # },
        # 'mycoplasma': {
        #     'keywords': [
        #                     'Uu', 'Mh'],
        #     'must_have': ['result','results']
        # },        
        'hcg_pregnancy': {
            'keywords': [
                        'HCG','Beta-HCG', 'HCG_2'],
            'must_have': []
        }        
        # 'thalassemia': {
        #     'keywords': ['a-地贫基因检测(3种缺失型)', 'a-地贫基因检测(3种非缺失型)', 'β-地贫基因检测(17种突变)' 
        #                     ],
        #     'must_have': ['result','results']
        # },        
        # 'anemia_four': {
        #     'keywords': [ 
        #                     'Fer' , 'Folate' , 'VitB12' , 'TRF' ],
        #     'must_have': ['result','results']
        # },       
        # 'liver_function': {
        #     'keywords': [
        #                     'ALB', 'ALT','AST','T-BiL','D-BiL','I-BiL','GGT','TP','GLO','A/G','ALP','ChE','AFU','ADA','TBA'],
        #     'must_have': ['result','results']
        # },        
        # 'thyroid_function': {
        #     'keywords': [
        #                     'TSH', 'TT3','TT4','FT3','FT4','TGAb','TSHRAb'],
        #     'must_have': ['result','results']
        # },       
        # 'preconception_health': {
        #     'keywords': [
        #                     '25-OH-VD'],
        #     'must_have': ['result','results']
        # },        
        # 'urine_routine': {
        #     'keywords': [
        #                     'PRO', 'GLU','KET','BIL','URO','NIT'],
        #     'must_have': ['result','results']
        # },         
        # '肿瘤标记物': {
        #     'keywords': [
        #                     'CA125', 'AFP','CEA'],
        #     'must_have': ['result','results']
        # },        
        # 'nuclear_medicine': {
        #     'keywords': [
        #                     'CA199'],
        #     'must_have': ['result','results']
        # },            
        # 'tb_tcell': {
        #     'keywords': [
        #                     'IFN-(N)', 'IFN-Y(T)','IFN-V(T-N)'],
        #     'must_have': ['result','results']
        # },               
        # 'rf_typing': {
        #     'keywords': ['IgA', 'lgG','lgM'
        #                     ],
        #     'must_have': ['result','results']
        # },             
        # 'blood_lipid': {
        #     'keywords': [
        #                     'TC', 'TG','LDL-C','HDL-C'],
        #     'must_have': ['result','results']
        # },                
        # 'blood_glucose': {
        #     'keywords': [
        #                     'FPG', '2hPG','HbA1c'],
        #     'must_have': ['result','results']
        # },              
        # 'homocysteine': {
        #     'keywords': [
        #                     'HCY'],
        #     'must_have': ['result','results']
        # },              
        # 'tct': {
        #     'keywords': [
        #                     'NILM', 'ASC-US','LSIL','HSIL','鳞状细胞癌 / 腺细胞癌'],
        #     'must_have': ['result','results']
        # },                           
        # 'y_microdeletion': {
        #     'keywords': ['sY84', 'sY86','sY127','sY134','sY254','sY255'
        #                     ],
        #     'must_have': ['result','results']
        # },            
        # 'lupus': {
        #     'keywords': [
        #                     'LA1', 'LA2','LA1/LA2'],
        #     'must_have': ['result','results']
        # }          
    },
    'english': {  
        'sex_hormone': {
            'keywords': ['estradiol', 'testosterone', 'lh', 'fsh', 'prolactin', 'progesterone'],
            'must_have': []
        },
        'sperm_status': {
            'keywords': ['volume', 'liquefaction',  'concentration',
                         'total concentration','motility','progressive motile','morphology','dna fragmentation index'],
            'must_have': []
        },
        'amh': {
            'keywords': ['amh', 'anti-mullerian hormone', 'antimullerian hormone','antimullerian'],
            'must_have': []
        },
        'ultrasound_tai': {
            'keywords': ['ultrasound', 'sheet', 'examdate', 'exam date'],
            'must_have': []
        },
        # 'immuno_five': {
        #     'keywords': ['IgG','IgA','IgM','C3','C4'],
        #     'must_have': ['result','results']
        # },
        # 'coag_function': {
        #     'keywords': [
        #                     'PT', 'PT-R','PT%','PT-INR','APTT','TT','AT-III','D-D','FIB'],
        #     'must_have': ['result','results']
        # },
        # 'renal_function': {
        #     'keywords': [
        #                     'Urea', 'UA','CYSC','CO2','GLU'],
        #     'must_have': ['result','results']
        # },
        'blood_type': {
            'keywords': [
                        'ABO Group', 'RH Group'],
            'must_have': []
        },
        # 'blood_routine': {
        #     'keywords': [
        #                     'WBC', 'RDW-CV','RBC','PLT','PDW','PCT','NEU%','NEU#','MPV','MON#','MON%','MCV','MCHC','HCH','LYM#','LYM%','HGB','HCT','EO%','EO#','BAS#','BAS%'],
        #     'must_have': ['result','results']
        # },
        # 'ct_dna': {
        #     'keywords': [
        #                     'CT-DNA'],
        #     'must_have': ['result','results']
        # },
        'infectious_disease': {
            'keywords': ['HBsAg', 'HBsAb','HBeAg','Anti-HCV','Anti-HIV','TPAb'],
            'must_have': []
        },
        # 'torch': {
        #     'keywords': [
        #                     'CMV-IgM', 'CMV-IgG','TOX-IgM','TOX-IgG','RV-IgM','RV-IgG','HSV-1-IgM','HSV-1-IgG','HSV-2-IgM','HSV-2-IgG','B19-IgM','B19-IgG'],
        #     'must_have': ['result','results']
        # },
        # 'mycoplasma': {
        #     'keywords': [
        #                     'Uu', 'Mh'],
        #     'must_have': ['result','results']
        # },        
        'hcg_pregnancy': {
            'keywords': [
                            'HCG','Beta-HCG', 'HCG_2'],
            'must_have': []
        }        
        # 'thalassemia': {
        #     'keywords': ['a-地贫基因检测(3种缺失型)', 'a-地贫基因检测(3种非缺失型)', 'β-地贫基因检测(17种突变)' 
        #                     ],
        #     'must_have': ['result','results']
        # },        
        # 'anemia_four': {
        #     'keywords': [
        #                      'Fer' , 'Folate' , 'VitB12' , 'TRF' ],
        #     'must_have': ['result','results']
        # },       
        # 'liver_function': {
        #     'keywords': [
        #                     'ALB', 'ALT','AST','T-BiL','D-BiL','I-BiL','GGT','TP','GLO','A/G','ALP','ChE','AFU','ADA','TBA'],
        #     'must_have': ['result','results']
        # },        
        # 'thyroid_function': {
        #     'keywords': [
        #                     'TSH', 'TT3','TT4','FT3','FT4','TGAb','TSHRAb'],
        #     'must_have': ['result','results']
        # },       
        # 'preconception_health': {
        #     'keywords': [
        #                     '25-OH-VD'],
        #     'must_have': ['result','results']
        # },        
        # 'urine_routine': {
        #     'keywords': [
        #                     'PRO', 'GLU','KET','BIL','URO','NIT'],
        #     'must_have': ['result','results']
        # },         
        # '肿瘤标记物': {
        #     'keywords': [
        #                     'CA125', 'AFP','CEA'],
        #     'must_have': ['result','results']
        # },        
        # 'nuclear_medicine': {
        #     'keywords': [
        #                     'CA199'],
        #     'must_have': ['result','results']
        # },            
        # 'tb_tcell': {
        #     'keywords': [
        #                     'IFN-(N)', 'IFN-Y(T)','IFN-V(T-N)'],
        #     'must_have': ['result','results']
        # },               
        # 'rf_typing': {
        #     'keywords': ['IgA', 'lgG','lgM'
        #                     ],
        #     'must_have': ['result','results']
        # },             
        # 'blood_lipid': {
        #     'keywords': [
        #                     'TC', 'TG','LDL-C','HDL-C'],
        #     'must_have': ['result','results']
        # },                
        # 'blood_glucose': {
        #     'keywords': [
        #                     'FPG', '2hPG','HbA1c'],
        #     'must_have': ['result','results']
        # },              
        # 'homocysteine': {
        #     'keywords': [
        #                     'HCY'],
        #     'must_have': ['result','results']
        # },              
        # 'tct': {
        #     'keywords': [
        #                     'NILM', 'ASC-US','LSIL','HSIL','鳞状细胞癌 / 腺细胞癌'],
        #     'must_have': ['result','results']
        # },                           
        # 'y_microdeletion': {
        #     'keywords': ['sY84', 'sY86','sY127','sY134','sY254','sY255'
        #                     ],
        #     'must_have': ['result','results']
        # },            
        # 'lupus': {
        #     'keywords': [
        #                     'LA1', 'LA2','LA1/LA2'],
        #     'must_have': ['result','results']
        # }          
    }
}

def contains_chinese(text):
    return any('\u4e00' <= ch <= '\u9fff' for ch in text)

def contains_thai(text):
    return any('\u0E00' <= ch <= '\u0E7F' for ch in text)

def chinese_ratio(text):
    chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return chinese_chars / len(text) if text else 0

def reliable_imwrite(path, image, max_retries=5, delay=0.5):
    for i in range(max_retries):
        success = cv2.imwrite(path, image)
        if success and os.path.exists(path) and os.path.getsize(path) > 0:
            return True
        print(f"[WARN] 第{i+1}次写入失败，重试中: {path}")
        time.sleep(delay)
    raise IOError(f"[ERROR] 图像写入失败: {path}")

def atomic_write_bytes(final_path: str, data: bytes, replace_retries: int = 5, replace_delay: float = 0.05):
    """Write bytes to final_path via a temp file then os.replace (atomic on POSIX).

    Extra hardening:
    - replace retry for OverlayFS/NFS transient visibility issues
    - existence checks + directory diagnostics on failure
    """
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = final_path + f".tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    last_err = None
    for i in range(replace_retries):
        try:
            # Some FS layers may briefly not see the freshly created temp file.
            if not os.path.exists(tmp_path):
                time.sleep(replace_delay)
                if not os.path.exists(tmp_path):
                    raise FileNotFoundError(f"tmp_path not found before replace: {tmp_path}")
            os.replace(tmp_path, final_path)
            return
        except FileNotFoundError as e:
            last_err = e
            time.sleep(replace_delay)

    # Best-effort diagnostics
    parent = os.path.dirname(final_path)
    logging.error(
        f"atomic_write_bytes failed after retries. final={final_path} tmp={tmp_path} "
        f"parent_exists={os.path.exists(parent)} parent_list={safe_listdir(parent)} err={last_err}"
    )
    raise last_err

def atomic_save_filestorage(file_storage, final_path: str, chunk_size: int = 1024 * 1024,
                            replace_retries: int = 5, replace_delay: float = 0.05,
                            min_bytes: int = 1):
    """Stream-save a Werkzeug FileStorage to disk safely with atomic replace + retries."""
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    tmp_path = final_path + f".tmp.{uuid.uuid4().hex}"
    
    file_storage.stream.seek(0)  # ✅ 兜底，确保从头读
    written = 0
    with open(tmp_path, "wb") as f:
        while True:
            chunk = file_storage.stream.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)

        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    if written < min_bytes:
        # ensure tmp is removed if created
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise ValueError(f"上传文件为空或写入失败: {final_path} (written={written})")

    last_err = None
    for i in range(replace_retries):
        try:
            if not os.path.exists(tmp_path):
                time.sleep(replace_delay)
                if not os.path.exists(tmp_path):
                    raise FileNotFoundError(f"tmp_path not found before replace: {tmp_path}")
            os.replace(tmp_path, final_path)
            return written
        except FileNotFoundError as e:
            last_err = e
            time.sleep(replace_delay)

    parent = os.path.dirname(final_path)
    logging.error(
        f"atomic_save_filestorage replace failed after retries. final={final_path} tmp={tmp_path} "
        f"written={written} parent_exists={os.path.exists(parent)} parent_list={safe_listdir(parent)} err={last_err}"
    )
    raise last_err



def atomic_imwrite_png(final_path: str, bgr_img, png_compression=3):
    # 统一用 PNG，避免 jpg/png 混乱引发的偶发解码问题
    ok, buf = cv2.imencode(".png", bgr_img, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
    if not ok:
        raise IOError(f"cv2.imencode 失败: {final_path}")
    atomic_write_bytes(final_path, buf.tobytes())
    
def _fsync_file(path: str):
    try:
        with open(path, "rb") as f:
            os.fsync(f.fileno())
    except Exception:
        # 有些文件系统/容器环境 fsync 可能失败，允许继续，但后面还有可读性校验兜底
        pass

def wait_until_readable_image(path, timeout=5.0, interval=0.2):
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return img
        time.sleep(interval)
    raise FileNotFoundError(f"[严重] 超时无法读取图像: {path}")

def safe_listdir(dir_path: str, limit=40):
    try:
        names = os.listdir(dir_path)
        names = sorted(names)[:limit]
        return names
    except Exception as e:
        return [f"<listdir failed: {e}>"]

def run_correct_document_with_retry(inp_path: str, out_path: str, retries=2):
    last_err = None
    for attempt in range(retries + 1):
        try:
            correct_document(inp_path, out_path)
            wait_until_readable(out_path, timeout=10.0)
            # 再做一次“能读”确认
            img = cv2.imread(out_path)
            if img is None:
                raise IOError(f"旋转后图片不可读: {out_path}")
            return
        except Exception as e:
            last_err = e
            logging.error(f"[correct_document] 第{attempt+1}次失败: {e}")
            time.sleep(0.1 * (attempt + 1))
    raise last_err

def wait_until_readable(path: str, timeout=6.0, interval=0.05, min_size=64):
    start = time.time()
    last_size = -1
    stable = 0
    while time.time() - start < timeout:
        if os.path.exists(path):
            try:
                size = os.path.getsize(path)
            except OSError:
                time.sleep(interval)
                continue
            if size >= min_size and size == last_size:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
                last_size = size
        time.sleep(interval)
    raise TimeoutError(f"文件未就绪/不稳定: {path}")

def classify_report_multi(lines, country):
    text = ''.join(lines).lower()
    rules = category_rules.get(country, {})
    category_scores = {}

    for category, rule in rules.items():
        matched_keywords = [kw for kw in rule['keywords'] if kw.lower() in text]
        keyword_hits = len(matched_keywords)

        must_have_keywords = rule.get('must_have', [])
        matched_must = [kw for kw in must_have_keywords if kw.lower() in text]
        
        must_have_passed = not must_have_keywords or bool(matched_must)

        if must_have_passed:
            category_scores[category] = keyword_hits
            print(f"[{category}] 命中 must_have: {matched_must}")
            print(f"[{category}] 命中的关键词: {matched_keywords}")
            print(f"[{category}] 关键词得分: {keyword_hits}")
    
    if not category_scores:
        print("未匹配到任何分类")
        return []
    
    if all(score == 0 for score in category_scores.values()):
        print("所有分类关键词得分为0，判定为未分类")
        return []

    print("分类得分统计：", category_scores)

    # 获取所有非0得分的分类
    matched_categories = [category for category, score in category_scores.items() if score > 0]
    
    print(f"匹配的分类: {matched_categories}")
    return matched_categories

# OCR识别
def extract_text_from_ocr_result(ocr_result):
        lines = []
        for block in ocr_result:
            if not block:
                continue
            for line in block:
                if len(line) >= 2 and isinstance(line[1], tuple) and isinstance(line[1][0], str):
                    lines.append(line[1][0].strip())
        return lines

def _safe_json(val):
    try:
        json.dumps(val)
        return val
    except TypeError:
        return str(val)

def get_request_info():
    info = {}

    core_attrs = [
        "method", "url", "base_url", "url_root", "path", "full_path",
        "remote_addr", "scheme", "is_secure", "is_json", "content_type",
        "content_length", "user_agent", "cookies", "headers",
        "args", "form", "files", "data", "get_json"
    ]

    for attr in core_attrs:
        try:
            if attr == "headers":
                info[attr] = dict(request.headers)
            elif attr == "cookies":
                info[attr] = request.cookies.to_dict()
            elif attr == "args":
                info[attr] = request.args.to_dict()
            elif attr == "form":
                info[attr] = request.form.to_dict()
            elif attr == "files":
                info[attr] = {k: v.filename for k, v in request.files.items()}
            elif attr == "data":
                info[attr] = request.get_data().decode("utf-8", errors="ignore")
            elif attr == "get_json":
                info["json"] = request.get_json(silent=True)
            elif attr == "user_agent":
                info[attr] = str(request.user_agent)
            else:
                info[attr] = _safe_json(getattr(request, attr))
        except Exception as e:
            info[attr] = f"<error: {str(e)}>"

    for attr in dir(request):
        if not attr.startswith("_") and attr not in info:
            try:
                val = getattr(request, attr)
                if callable(val):
                    continue
                info[attr] = _safe_json(val)
            except Exception as e:
                info[attr] = f"<error: {str(e)}>"

    return info

def rotate_by_ccw_angle(img, ccw_angle: int):
    """按 0/90/180/270 逆时针旋转图像。"""
    a = int(ccw_angle) % 360
    if a == 0:
        return img
    if a == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if a == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if a == 270:
        # 逆时针270 == 顺时针90
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    raise ValueError(f"Unsupported angle: {ccw_angle}")

def load_ccw_angles_from_ocr_result(ocr_result_file: str):
    with open(ocr_result_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    angles = []
    for obj in data:
        angle = None
        if isinstance(obj, dict):
            angle = (obj.get("doc_preprocessor_res") or {}).get("angle")
        angles.append(angle)
    return angles

def get_doc_angle(res):
    """
    兼容不同版本 res：优先用 res.to_dict()，不行再尝试属性读取。
    """
    # 1) 如果 res 能转 dict（你 save_to_json 的结果就是这种结构）
    if hasattr(res, "to_dict"):
        d = res.to_dict()
        return (d.get("doc_preprocessor_res") or {}).get("angle")

    # 2) 尝试属性结构（不同版本字段可能不同）
    if hasattr(res, "doc_preprocessor_res"):
        dpr = getattr(res, "doc_preprocessor_res")
        if isinstance(dpr, dict):
            return dpr.get("angle")
        if hasattr(dpr, "angle"):
            return getattr(dpr, "angle")

    return None

def save_oriented_images_and_overwrite_paths_by_angles(all_processed_paths, angles, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    new_paths = []
    for src_path, angle in zip(all_processed_paths, angles):
        img = cv2.imread(src_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {src_path}")

        if angle is not None:
            img = rotate_by_ccw_angle(img, angle)

        base = os.path.splitext(os.path.basename(src_path))[0]
        out_path = os.path.join(save_dir, f"{base}_oriented.png")
        cv2.imwrite(out_path, img)
        new_paths.append(out_path)

    all_processed_paths[:] = new_paths

    # 原地覆盖：外部拿到的 list 引用也会更新
    all_processed_paths[:] = new_paths

def sort_lines_by_boxes(rec_texts, rec_boxes):
    """按(y1, x1)做近似阅读顺序排序。"""
    if not rec_boxes or len(rec_boxes) != len(rec_texts):
        return rec_texts
    idxs = list(range(len(rec_texts)))
    idxs.sort(key=lambda i: (rec_boxes[i][1], rec_boxes[i][0]))  # y1, x1
    return [rec_texts[i] for i in idxs]

def extract_lines_from_ocr_json(obj):
    """
    从单个OCR结果json(dict)提取行文本 list[str]
    适配你现在 json 格式里常见的 rec_texts / rec_boxes
    """
    rec_texts = obj.get("rec_texts") or []
    rec_boxes = obj.get("rec_boxes") or []
    lines = sort_lines_by_boxes(rec_texts, rec_boxes)
    # 可选：去掉空行
    lines = [s for s in lines if isinstance(s, str) and s.strip()]
    return lines

def build_all_lines_per_image_from_dir(ocr_result_dir, all_processed_paths, ocr_result_file=None):
    """
    返回 all_lines_per_image，长度= len(all_processed_paths)，顺序严格对齐。
    通过解析文件名 out_{i}_..._res.json 的 i 来定位到对应图片。
    """
    json_files = glob(os.path.join(ocr_result_dir, "*.json"))
    # 初始化：每张图一个空列表
    all_lines_per_image = [[] for _ in range(len(all_processed_paths))]
    # 额外：用于整合写出
    merged_results = [None for _ in range(len(all_processed_paths))]

    # out_0_xxx_res.json -> index=0
    pat = re.compile(r"(?:^|[/\\])out_(\d+)_.*_res\.json$", re.IGNORECASE)

    used = set()
    for fp in json_files:
        m = pat.search(fp)
        if not m:
            continue
        idx = int(m.group(1))
        if idx < 0 or idx >= len(all_processed_paths):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)

        all_lines_per_image[idx] = extract_lines_from_ocr_json(obj)
        if isinstance(obj, dict):
            obj = dict(obj)  # 拷贝，避免意外修改原引用
        merged_results[idx] = obj
        used.add(idx)

    # 兜底提示：有些 idx 没找到对应 json
    missing = [i for i in range(len(all_processed_paths)) if i not in used]
    if missing:
        print(f"[WARN] Missing OCR json for indices: {missing}")

    # 把 None 填成空 dict，避免 dump 出现 null（看你喜好）
    merged_results = [r if isinstance(r, dict) else {} for r in merged_results]

    # 写出整合文件（可选）
    if ocr_result_file:
        os.makedirs(os.path.dirname(ocr_result_file), exist_ok=True)
        with open(ocr_result_file, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)
    return all_lines_per_image

def classify_per_image(all_lines_per_image):
    matched_categories_per_image = []
    country_per_image = []
    th_keywords = ['harmonicare', 'deep']

    for all_lines in all_lines_per_image:
        full_text = ''.join(all_lines)
        lower_text = full_text.lower()

        if contains_thai(full_text):
            country = 'thailand'
        elif any(k in lower_text for k in th_keywords):
            country = 'thailand'
        elif chinese_ratio(full_text) > 0.01:
            country = 'china'
        else:
            country = 'english'

        if "互认项目" in full_text:
            cutoff_index = full_text.index("互认项目")
            full_text = full_text[:cutoff_index]
            # 重新构建 all_lines，保持原有格式
            all_lines = [full_text]

        matched_categories = classify_report_multi(all_lines, country)
        matched_categories_per_image.append(matched_categories)
        country_per_image.append(country)

    return matched_categories_per_image, country_per_image
def create_sandbox_directory(output_base, task_id):
    try:
        sandbox_dir = os.path.join(output_base, "sandbox", task_id)
        os.makedirs(sandbox_dir, exist_ok=True)
        logging.info(f"Directory created or already exists: {sandbox_dir}")
        return sandbox_dir 
    except Exception as e:
        logging.error(f"Failed to create directory {sandbox_dir}: {e}")
        raise e

def ensure_rgb(images):
    """
    确保所有图像都是 RGB 模式
    
    Args:
        images: PIL.Image 对象列表
    
    Returns:
        list: 转换为 RGB 模式的图像列表
    """
    rgb_images = []
    for img in images:
        if not isinstance(img, Image.Image):
            logging.warning(f"跳过非 PIL.Image 对象: {type(img)}")
            continue
        
        if img.mode != 'RGB':
            logging.info(f"转换图像模式: {img.mode} -> RGB")
            rgb_images.append(img.convert('RGB'))
        else:
            rgb_images.append(img)
    
    return rgb_images

def get_report_time_safely(sandbox_dir):
    """
    尝试从 sandbox_dir 下的 report_info.json 读取 report_time。
    如果不存或读取失败，返回 None。
    """
    report_info_path = os.path.join(sandbox_dir, 'report_info.json')
    if os.path.exists(report_info_path):
        try:
            # 简单的重试机制，防止文件正在写入时读取报错
            for _ in range(3):
                try:
                    with open(report_info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    return info.get('report_time')
                except json.JSONDecodeError:
                    time.sleep(0.1)
                    continue
        except Exception as e:
            logging.error(f"读取 report_time 失败: {e}")
    return None

def wait_for_template_and_send(task_id, callback_url, sandbox_dir, timeout=360, check_interval=1):
    txt_path = os.path.join(sandbox_dir, 'report_polished_merge.txt')
    start_time = time.time()
    
    # 获取初始大小
    old_size = os.path.getsize(txt_path) if os.path.exists(txt_path) else None
    success = False

    logging.info(f"[监听开始] TaskID: {task_id}, 初始 size: {old_size}")

    while time.time() - start_time < timeout:
        time.sleep(check_interval)
        
        if not os.path.exists(txt_path):
            # 文件还未生成，继续等待
            continue

        new_size = os.path.getsize(txt_path)
        
        # 检测到文件大小发生变化，或者之前不存在现在存在了
        if old_size is None or new_size != old_size:
            logging.info(f"[变更检测] TaskID: {task_id}, size 变化: {old_size} -> {new_size}")

            # 等待文件写入稳定 (防抖动)
            stable = False
            stable_wait_start = time.time()
            last_size = -1
            
            while time.time() - stable_wait_start < 5: # 最多防抖5秒
                curr_size = os.path.getsize(txt_path)
                if curr_size == last_size and curr_size > 0:
                    stable = True
                    break
                last_size = curr_size
                time.sleep(0.5)

            if not stable:
                logging.info(f"[文件未稳定] TaskID: {task_id}，跳过本次轮询")
                old_size = new_size
                continue

            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    logging.info(f"[文件为空] TaskID: {task_id}, 继续等待")
                    old_size = new_size
                    continue

                # === 关键修改：在发送前，尝试获取 report_time ===
                # 此时报告已生成，report_info.json 大概率也已生成
                report_time = get_report_time_safely(sandbox_dir)
                logging.info(f"[数据准备] TaskID: {task_id}, 获取到的 report_time: {report_time}")

                payload = {
                    "taskId": task_id,
                    "result": content,
                    "report_time": report_time 
                }

                response = requests.post(callback_url, json=payload, timeout=20)
                logging.info(f"[推送成功] TaskID: {task_id}, 状态: {response.status_code}")
                
                success = True
                break # 成功后退出循环
                
            except Exception as e:
                logging.error(f"[推送失败] TaskID: {task_id}, 错误: {e}")
                old_size = new_size
                time.sleep(2) # 稍作延迟后重试
                continue

        # 更新 old_size 以便下一次比较
        if os.path.exists(txt_path):
             old_size = os.path.getsize(txt_path)

    if not success:
        logging.warning(f"[超时] TaskID: {task_id}, 在 {timeout}s 内未成功推送")
        try:
            requests.post(callback_url, json={
                "taskId": task_id,
                "result": "报告解读失败，请重试"
            }, timeout=20)
        except Exception as e:
            logging.error(f"[推送超时通知失败] TaskID: {task_id}, 错误: {e}")

@app.route('/diagnosis/classify-image', methods=['POST'])
def classify_image_api():
    """
    【支持多文件上传版本】
    修复核心问题：通过唯一的 TaskID 隔离进程,防止僵尸进程误删文件。
    """
    # ==================== 1. 基础验证 ====================
    # ✅ 改用 getlist 获取多个文件
    files = request.files.getlist('file')
    if not files:
        return jsonify({"code": 400, "message": "未上传文件", "data": {}})
    
    valid_files = []
    for file in files:
        filename = os.path.basename(file.filename)
        if not filename:
            continue
            
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            return jsonify({
                "code": 400, 
                "message": f"不支持的文件类型: {filename}", 
                "data": {}
            })
        valid_files.append(file)
    if not valid_files:
        return jsonify({"code": 400, "message": "没有有效的文件", "data": {}})
    # 获取参数
    callback_url = request.form.get('callbackUrl') or request.args.get('callbackUrl')
    boolean_requirevalidation = request.form.get('Boolean requireValidation') or request.args.get('Boolean requireValidation')
    logging.info(f"收到 {len(valid_files)} 个文件, callback: {callback_url}")

    # ==================== 2. 创建目录 (核心修复) ====================
    # ❌ 旧代码: task_id = str(uuid.uuid4()) -> 容易被僵尸进程碰撞
    # ✅ 新代码: 增加 进程ID(pid) 和 纳秒时间戳 -> 绝对唯一，没人能删你的目录
    task_id = f"{uuid.uuid4()}-{os.getpid()}-{time.time_ns()}"
    
    # 你的 output_base 应该已经在外部定义好了
    sandbox_dir = os.path.join(output_base, "sandbox", task_id)
    yolo_result_dir = os.path.join(sandbox_dir, "yolo_result")
    
    try:
        os.makedirs(yolo_result_dir, exist_ok=True)
    except Exception as e:
        logging.error(f"Dir creation failed: {e}")
        return jsonify({'error': 'Dir creation failed'}), 500

    # ==================== 3. 保存文件 (防御性写入) ====================
    all_file_paths = []
    for idx, file in enumerate(valid_files):
        filename = os.path.basename(file.filename)
        # ✅ 为每个文件添加索引避免重名
        safe_filename = f"{idx:03d}_{filename}"
        file_path = os.path.join(sandbox_dir, safe_filename)

        try:
            logging.info(f"保存文件 [{idx+1}/{len(valid_files)}]: {file_path}")
            
            # 确保目录存在
            if not os.path.exists(sandbox_dir):
                os.makedirs(sandbox_dir, exist_ok=True)

            # 重置指针并保存
            file.stream.seek(0)
            atomic_save_filestorage(file, file_path)
            
            # 等待文件出现
            max_wait = 5
            start = time.time()
            while not os.path.exists(file_path):
                if time.time() - start > max_wait:
                    raise TimeoutError(f"文件保存超时: {file_path}")
                time.sleep(0.01)
            
            time.sleep(0.1)
            
            # 强制刷盘
            with open(file_path, 'a') as f:
                os.fsync(f.fileno())
            
            # 最终验证
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"文件保存后消失: {file_path}")
            
            if os.path.getsize(file_path) == 0:
                raise ValueError(f"文件为空: {file_path}")
            
            all_file_paths.append(file_path)
            logging.info(f"✅ 文件 {idx+1} 保存成功: {file_path} ({os.path.getsize(file_path)} bytes)")

        except Exception as e:
            logging.error(f"❌ 文件 {idx+1} 保存失败: {e}")
            logging.error(traceback.format_exc())
            return jsonify({
                'error': f'文件 {filename} 保存失败', 
                'details': str(e)
            }), 500


# ==================== 4. 获取请求信息 ====================
    try:
        request_info = get_request_info()
        request_info['file_count'] = len(valid_files)
        request_info['files'] = [os.path.basename(p) for p in all_file_paths]
    except Exception as e:
        logging.warning(f"⚠️ 获取请求信息失败: {e}")
        request_info = {"error": str(e)}

    try:
        with open(os.path.join(sandbox_dir, "request_info.json"), "w", encoding="utf-8") as f:
            json.dump(request_info, f, ensure_ascii=False, indent=4)
    except:
        pass
# ==================== 5. 处理所有文件 ====================
    all_images = []  # 存储所有页面
    all_lines = []   # 存储所有文本行
    all_processed_paths = []  # 存储所有处理后的图片路径
    all_ocr_pages = []  # 存储所有OCR结果
    try:
        # ✅ 遍历处理每个文件
        for file_idx, file_path in enumerate(all_file_paths):
            filename = os.path.basename(file_path)
            file_ext = os.path.splitext(filename)[1].lower()
            
            logging.info(f"处理文件 [{file_idx+1}/{len(all_file_paths)}]: {filename}")
            
            try:
                # 转换为图片
                if file_ext == '.pdf':
                    logging.info(f"Processing PDF: {filename}")
                    images = convert_from_path(file_path, dpi=135)
                else:
                    logging.info(f"Processing image: {filename}")
                    with Image.open(file_path) as img:
                        img_copy = img.copy()
                    images = [img_copy]

                # 确保所有图片都是RGB模式
                images = ensure_rgb(images)
                logging.info(f"文件 {file_idx+1} 转换完成, 共 {len(images)} 页")
                
                all_images.extend(images)

            except Exception as e:
                logging.error(f"文件 {file_idx+1} 处理失败: {e}")
                logging.error(traceback.format_exc())
                return jsonify({
                    'code': 500,
                    'error': f'文件 {filename} 处理失败',
                    'message': str(e),
                    'data': {}
                }), 500

# ==================== 6. 处理所有页面 ====================
        pre_rotate_dir = os.path.join(sandbox_dir, "pre_rotate_img")
        os.makedirs(pre_rotate_dir, exist_ok=True)
        save_dir = os.path.join(sandbox_dir, "paddle_rotate_img")
        os.makedirs(save_dir, exist_ok=True)
        ocr_result_dir = os.path.join(sandbox_dir, "ocr_result_dir")
        
        logging.info(f"开始处理 {len(all_images)} 个页面...")

        
        for idx, pil_img in enumerate(all_images):
            try:
                # PIL -> OpenCV BGR
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                rgb = np.array(pil_img)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                # 保存为PNG
                tag = f"{idx}_{uuid.uuid4().hex[:8]}"
                inp_path = os.path.join(pre_rotate_dir, f"inp_{tag}.png")
                out_path = os.path.join(pre_rotate_dir, f"out_{tag}.png")

                atomic_imwrite_png(inp_path, bgr, png_compression=3)
                wait_until_readable(inp_path, timeout=6.0)

                # 纠偏/旋转
                run_correct_document_with_retry(inp_path, out_path, retries=2)
                all_processed_paths.append(out_path)
 
            except Exception as e:
                logging.error(f"[页处理失败] idx={idx}, err={e}")
                logging.error(f"[诊断] pre_rotate_dir list: {safe_listdir(pre_rotate_dir)}")
                raise
            finally:
                try:
                    pil_img.close()
                except Exception:
                    pass
        result = ocr.predict(all_processed_paths)
        print("1")
        # 先把每页结果写到 ocr_result_dir
        for res in result:
            res.save_to_json(ocr_result_dir)
        ocr_result_file = os.path.join(sandbox_dir, "ocr_result.json")
        # 1) 先把“回正后图片”存到 save_dir，并覆盖 all_processed_paths
        # save_oriented_images_and_overwrite_paths(all_processed_paths, result, save_dir)
        print("2")
        #all_lines_per_image，matched_categories_per_image是一个列表
        all_lines_per_image = build_all_lines_per_image_from_dir(
            ocr_result_dir=ocr_result_dir,
            all_processed_paths=all_processed_paths,
            ocr_result_file=ocr_result_file
        )
        angles = load_ccw_angles_from_ocr_result(ocr_result_file)
        save_oriented_images_and_overwrite_paths_by_angles(all_processed_paths, angles, save_dir)
        matched_categories_per_image, country_per_image = classify_per_image(all_lines_per_image)


        # vig判定
        def field_in_lines(field_keywords, lines, lowercase=True):
            target_lines = [line.lower() for line in lines] if lowercase else lines
            return all(any(keyword in line for line in target_lines) for keyword in field_keywords)
        # 每张图一个 vig
        vig_per_image = [2] * len(all_processed_paths)

        for i, (cats, country, img_path) in enumerate(zip(matched_categories_per_image, country_per_image, all_processed_paths)):
            if "ultrasound" in cats and country.lower() in ("china"):
                vig_per_image[i] = 0
            if "ultrasound_tai" in cats and country.lower() in ("thailand", "english"):
                vig_per_image[i] = 0
                if country.lower() in ("thailand", "english"):
                    yolo_cls(img_path, yolo_result_dir)
        end_file_path = os.path.join(yolo_result_dir, 'end.txt')
        with open(end_file_path, 'w'):
            pass


        for i in range(len(all_processed_paths)):
            all_lines = all_lines_per_image[i]
            cats = matched_categories_per_image[i]
            country = country_per_image[i]
            lines_lower = [line.lower() for line in all_lines]
            if vig_per_image[i] == 0:
                continue                
           
            if "immuno_five" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "ultrasound" in cats and country.lower() in ("china"):
                vig_per_image[i] = 0

            elif "coag_function" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "renal_function" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
                not_exit = [
                ['葡萄糖苷酶']
                ]
                lines_lower = [line.lower() for line in all_lines]
                if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                    vig_per_image[i] = 2
            elif "blood_type" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "blood_routine" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "ct_dna" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "infectious_disease" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "torch" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "mycoplasma" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "hcg_pregnancy" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "anemia_four" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "liver_function" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "thyroid_function" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
                        
            elif "preconception_health" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

                        
            elif "urine_routine" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "nuclear_medicine" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "tb_tcell" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "rf_typing" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "blood_lipid" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "blood_glucose" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "homocysteine" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
        
            elif "tct" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
            
            elif "y_microdeletion" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "lupus" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "thalassemia" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "leukorrhea_routine_report" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "tumor_marker_report" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "neisseria_gonorrhoeae_culture" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
                
            elif "membrane_potential" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0

            elif "dna_fragmentation_index" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
                if "ultrasound" in cats:
                    matched_categories_per_image[i] = [c for c in cats if c != "dna_fragmentation_index"]

            elif "sex_hormone" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
                if boolean_requirevalidation == "1":           
                    if country.lower() in ('china',):
                        gender_keywords = [
                            [ '别'],  # 触发词
                            ['女', '男']  # 期望的后续内容
                        ]
                        age_keywords = [
                            ['龄', '出生日期'],  # 触发词
                            ['岁', '年']  # 期望的后续内容（数字会单独检查）
                        ]
                        not_exit = [
                            ['超敏抗缪勒氏管激素'],
                            ['抗精子抗体'],
                            ['抗促甲状腺激素受体抗体检测']
                        ]
                        
                        lines_lower = [line.lower() for line in all_lines]
                        full_text = ''.join(all_lines)
                        
                        # 检查排除项
                        if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                            vig_per_image[i] = 2
                        else:
                            # 检查性别信息
                            gender_found = False
                            for trigger in gender_keywords[0]:
                                if trigger in full_text:
                                    idx = full_text.index(trigger)
                                    # 检查触发词后5个字符内是否有期望的关键词
                                    following_text = full_text[idx:idx + len(trigger) + 5]
                                    if any(kw in following_text for kw in gender_keywords[1]):
                                        gender_found = True
                                        break
                            
                            if not gender_found:
                                vig_per_image[i] = 3
                            else:
                                # 检查年龄信息
                                age_found = False
                                for trigger in age_keywords[0]:
                                    if trigger in full_text:
                                        idx = full_text.index(trigger)
                                        # 检查触发词后5个字符内是否有数字或期望的关键词
                                        following_text = full_text[idx:idx + len(trigger) + 5]
                                        if any(char.isdigit() for char in following_text) or any(kw in following_text for kw in age_keywords[1]):
                                            age_found = True
                                            break
                                
                                if not age_found:
                                    vig_per_image[i] = 4
                                    
                    if country.lower() in ('thailand','english',):
                        gender_keywords = [
                            ['gender', 'sex'],
                            ['male', 'female', 'man', 'woman']
                        ]
                        age_keywords = [
                            ['age', 'dob', 'd.o.b.'],
                            ['year', 'yrs', 'yr']
                        ]
                        not_exit = [
                            ['summaryreport'],
                            ['summary report'], ['fertilization'],          
                            ['fertilization result'], ['fertilizationresult'],
                            ['用药方案']
                        ]
                        
                        lines_lower = [line.lower() for line in all_lines]
                        full_text_lower = ''.join(lines_lower)
                        
                        # 检查排除项
                        if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                            vig_per_image[i] = 2
                        else:
                            # 检查性别信息
                            gender_found = False
                            for trigger in gender_keywords[0]:
                                if trigger in full_text_lower:
                                    idx = full_text_lower.index(trigger)
                                    following_text = full_text_lower[idx:idx + len(trigger) + 5]
                                    if any(kw in following_text for kw in gender_keywords[1]):
                                        gender_found = True
                                        break
                            
                            if not gender_found:
                                vig_per_image[i] = 3
                            else:
                                # 检查年龄信息
                                age_found = False
                                for trigger in age_keywords[0]:
                                    if trigger in full_text_lower:
                                        idx = full_text_lower.index(trigger)
                                        following_text = full_text_lower[idx:idx + len(trigger) + 5]
                                        if any(char.isdigit() for char in following_text) or any(kw in following_text for kw in age_keywords[1]):
                                            age_found = True
                                            break
                                
                                if not age_found:
                                    vig_per_image[i] = 4
                            
            elif "amh" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0
                if boolean_requirevalidation == "1":           
                    if country.lower() in ('china',):
                        age_keywords = [
                            [ '龄', '出生日期'],  # 触发词
                            ['岁', '年']  # 期望的后续内容（数字会单独检查）
                        ]
                        not_exit = [
                            ['超敏抗缪勒氏管激素'],
                            ['抗精子抗体'],
                            ['抗促甲状腺激素受体抗体检测']
                        ]
                        
                        lines_lower = [line.lower() for line in all_lines]
                        full_text = ''.join(all_lines)
                        
                        # 检查排除项
                        if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                            vig_per_image[i] = 2
                        else:                   
                            # 检查年龄信息
                            age_found = False
                            for trigger in age_keywords[0]:
                                if trigger in full_text:
                                    idx = full_text.index(trigger)
                                    # 检查触发词后5个字符内是否有数字或期望的关键词
                                    following_text = full_text[idx:idx + len(trigger) + 5]
                                    if any(char.isdigit() for char in following_text) or any(kw in following_text for kw in age_keywords[1]):
                                        age_found = True
                                        break
                            
                            if not age_found:
                                vig_per_image[i] = 4
                                    
                    if country.lower() in ('thailand','english',):
                        age_keywords = [
                            ['age', 'dob', 'd.o.b.'],
                            ['year', 'yrs', 'yr']
                        ]
                        not_exit = [
                            ['summaryreport'],
                            ['summary report'], ['fertilization'],          
                            ['fertilization result'], ['fertilizationresult'],
                            ['用药方案']
                        ]
                        
                        lines_lower = [line.lower() for line in all_lines]
                        full_text_lower = ''.join(lines_lower)
                        
                        # 检查排除项
                        if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                            vig_per_image[i] = 2
                        else:
                            # 检查年龄信息
                            age_found = False
                            for trigger in age_keywords[0]:
                                if trigger in full_text_lower:
                                    idx = full_text_lower.index(trigger)
                                    following_text = full_text_lower[idx:idx + len(trigger) + 5]
                                    if any(char.isdigit() for char in following_text) or any(kw in following_text for kw in age_keywords[1]):
                                        age_found = True
                                        break
                            
                            if not age_found:
                                vig_per_image[i] = 4

            elif "sperm_status" in cats and country.lower() in ("thailand", "english", "china"):
                vig_per_image[i] = 0      
                if country.lower() in ('china',):
                    not_exit = [
                    []
                    ]
                    lines_lower = [line.lower() for line in all_lines]
                    if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                        vig_per_image[i] = 2

                if country.lower() in ('thailand',):
                    not_exit = [

                    ]
                    lines_lower = [line.lower() for line in all_lines]
                    if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                        vig_per_image[i] = 2

                if country.lower() in ('english',):
                    not_exit = [

                    ]
                    lines_lower = [line.lower() for line in all_lines]
                    if any(any(word in line for line in lines_lower) for group in not_exit for word in group):
                        vig_per_image[i] = 2

        pages1 = []
        for i, (img_path, all_lines, cats, country, vig) in enumerate(
            zip(all_processed_paths, all_lines_per_image, matched_categories_per_image, country_per_image, vig_per_image)
        ):
            pages1.append({
                "pageIndex": i,
                "imagePath": img_path,  # 如不想回传绝对路径，可用 os.path.basename(img_path)
                "country": country,
                "matchedCategories": [cat.lower() for cat in (cats or [])] or ["UNCLASSIFIED"],
                "vig": int(vig),
            })
        pages2 = []
        for i, (img_path, all_lines, cats, country, vig) in enumerate(
            zip(all_processed_paths, all_lines_per_image, matched_categories_per_image, country_per_image, vig_per_image)
        ):
            pages2.append({
                "pageIndex": i,
                "imagePath": img_path,  # 如不想回传绝对路径，可用 os.path.basename(img_path)
                "country": country,
                "matchedCategories": [cat.upper() for cat in (cats or [])] or ["UNCLASSIFIED"],
                "vig": int(vig),
            })
        # 记录本任务到沙箱下的 task_record.json
        task_record_file = os.path.join(sandbox_dir, 'task_record.json')
        # 获取当前时间
        now = datetime.now()
        # 格式化为 年-月-日 时:分
        time_str = now.strftime("%Y-%m-%d %H:%M")

        record_data = {
            "task_id": task_id,
            "file_count": len(valid_files),  # ✅ 记录文件数量
            "page_count": len(all_images),   # ✅ 记录总页数
            "create_time": time_str,
            "callback_url": callback_url,
            "pages": pages1,  # ✅ 关键：记录多页结果
        }
        # 一步到位写为 {task_id: record_data}
        os.makedirs(os.path.dirname(task_record_file), exist_ok=True)

        with open(task_record_file, 'w', encoding='utf-8') as f:
            json.dump({task_id: record_data}, f, indent=2, ensure_ascii=False)

        return jsonify({
            "code": 200,
            "message": "ok",
            "data": {
                "taskId": task_id,
                "fileCount": len(valid_files),
                "pageCount": len(all_images),
                "pages": pages2  # ✅ 多页返回
            }    
        })
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            'code': 500,
            'error': 'Processing failed',
            'message': str(e),
            'data': {}
        }), 500
        
    finally:
        # ✅ 清理所有 Image 对象
        if all_images:
            for img in all_images:
                try:
                    img.close()
                except:
                    pass
            del all_images
            logging.info("All images closed")

@app.route('/diagnosis/submit-date', methods=['POST'])
def submit_date():
    task_id = request.form.get('taskId')
    report_date = request.form.get('reportDate')

    if not task_id:
        return jsonify({"code": 400, "message": "缺少 taskId", "data": {}})

    sandbox_dir = os.path.join(output_base, "sandbox", task_id)
    task_record_file = os.path.join(sandbox_dir, 'task_record.json')

    # 1. 检查任务记录是否存在
    # 直接检查文件是否存在，不再进行无意义的轮询。
    # 如果 classify 接口成功返回，此文件必然存在。
    if not os.path.exists(task_record_file):
        logging.warning(f"Task record not found for ID: {task_id}")
        return jsonify({"code": 404, "message": "任务记录不存在", "data": {}})
    
    try:
        with open(task_record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read task record: {e}")
        return jsonify({"code": 500, "message": "任务记录读取失败", "data": {}})

    if task_id not in record:
        return jsonify({"code": 404, "message": "task_id 无效", "data": {}})

    # 2. 更新 report_date (只做这一件事，不等待 report_time)
    if report_date is not None:
        record[task_id]['report_date'] = report_date
        # 这里的 report_time 如果还没生成，就不更新进去了，交给回调线程去读最新的
        try:
            with open(task_record_file, 'w', encoding='utf-8') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        except Exception as e:
             logging.error(f"Failed to update report date: {e}")

    # 3. 启动后台线程
    callback_url = record[task_id].get('callback_url')
    
    if callback_url:
        # 将 sandbox_dir 传进去，方便线程内部读取文件
        thread = threading.Thread(
            target=wait_for_template_and_send, 
            args=(task_id, callback_url, sandbox_dir)
        )
        thread.daemon = True 
        thread.start()
        logging.info(f"Background thread started for TaskID: {task_id}")

    # 4. 立即返回响应，不阻塞
    return jsonify({
        "code": 200,
        "message": "报告日期已保存，后台处理中",
        "data": {
            "taskId": task_id,
            "reportDate": report_date
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, processes=True)