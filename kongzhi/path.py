import os

#项目根目录
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
#base_path = r"/root/md_ocr/"

# 要监控的 sandbox 目录
watch_paths =[os.path.join(base_path,"output_bf_test/sandbox/")]


# main_mdocr.py 的绝对路径
md_ocr_path = os.path.join(base_path, "md_ocr/main_mdocr.py")
# advice_generate.py的绝对路径（旧决策树）
adv_gen_path = os.path.join(base_path, "d_tree/advice_generate.py")
# advice_generate.py的绝对路径（新决策树）
d_tree_path = os.path.join(base_path, "new_d_tree/advice_generate.py")
# main_eval.py的绝对路径
eval_core_path = os.path.join(base_path, "eval_core/main_eval.py")
# md_llm_api.py的绝对路径
md_llm_api_path = os.path.join(base_path, "llm_polish/md_llm_api.py")
