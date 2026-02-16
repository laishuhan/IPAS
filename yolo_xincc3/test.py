import os
import argparse
from utils import process_single_image, preprocess, sort_by_region_number
from ocr import OCR_digital_extraction
from size import size_matching
from yolo_detection import yolo_follicle_detection, yolo_number_detection

if __name__ == "__main__":

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

    image_input = './test_image_outdir'
    result_output = './test_output_outdir'

    image = preprocess(image_input)

    for i in range(len(image)):

        # if i != 0:
        #     continue

        crop_file_path, image_name = process_single_image(image[i], result_output)
        
        results = {}
        dist = {}
        endom = {}
        crop_save_path = os.path.join(crop_file_path, 'crop')
        sorted_filenames = sort_by_region_number(os.listdir(crop_save_path))

        for filename in sorted_filenames:
            file_path = os.path.join(crop_save_path, filename)
            
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                result, dist_exist, endom_exist = OCR_digital_extraction(file_path, debug=True)
                base_filename = os.path.splitext(filename)[0]
                results[base_filename] = result
                dist[base_filename] = dist_exist
                endom[base_filename] = endom_exist

        follicle_result = os.path.join(crop_file_path, "follicle")
        number_result = os.path.join(crop_file_path, "number")
        json_output_path = os.path.join(crop_file_path, "result.json")

        yolo_follicle_detection(crop_save_path, follicle_result)
        yolo_number_detection(crop_save_path, number_result)
    
        size_matching(crop_file_path, results, dist, endom, json_output_path)
