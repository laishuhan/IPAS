import os
import argparse
import shutil
from .utils import process_single_image, preprocess, sort_by_region_number
from .ocr import OCR_digital_extraction
from .size import size_matching
from .yolo_detection import yolo_follicle_detection, yolo_number_detection
from .bayesian_detection import bayesian_detection

def yolo_cls(image_input, result_output):

    """
    主函数： 批量处理图片
    
    Helps:
        image_input: 检测图片路径
        result_output: 输出结果根路径
        {
            ./result_output/image_name/crop/: 分割后图片路径
            ./result_output/image_name/follicle/: 卵泡识别结果路径
            ./result_output/image_name/number/: 数字识别结果路径
            ./result_output/image_name/final/: 最终结果路径
            ./result_output/image_name/result.json: 输出信息文件路径
            }
    """

    # image_input = './test'
    # result_output = './test_output'

    image = preprocess(image_input)

    for i in range(len(image)):

        crop_file_path, image_name = process_single_image(image[i], result_output)
        
        results = {}
        ovary = {}
        dist = {}
        endom = {}
        plus = {}
        crop_save_path = os.path.join(crop_file_path, 'crop')
        sorted_filenames = sort_by_region_number(os.listdir(crop_save_path))

        for filename in sorted_filenames:
            file_path = os.path.join(crop_save_path, filename)
            
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                result, ovary_exist, dist_exist, endom_exist, plus_exist = OCR_digital_extraction(file_path, debug=False)
                base_filename = os.path.splitext(filename)[0]
                results[base_filename] = result
                ovary[base_filename] = ovary_exist
                dist[base_filename] = dist_exist
                endom[base_filename] = endom_exist
                plus[base_filename] = plus_exist

        follicle_result = os.path.join(crop_file_path, "follicle")
        number_result = os.path.join(crop_file_path, "number")
        json_output_path = os.path.join(crop_file_path, "result.json")

        bayesian_detection(crop_save_path, follicle_result)
        # yolo_follicle_detection(crop_save_path, follicle_result)
        yolo_number_detection(crop_save_path, number_result)
    
        size_matching(crop_file_path, results, dist, endom, ovary, plus, json_output_path)

        # report_info_path = os.path.join(crop_file_path, "report_info.json")
        # target_dir = os.path.abspath(os.path.join(report_info_path, "..", "..", ".."))
        # target_path = os.path.join(target_dir, "report_info.json")
        # #shutil.copy(report_info_path, target_path)
        # print("复制完成:", target_path)
