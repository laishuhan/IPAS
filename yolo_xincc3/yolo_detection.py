import os
os.environ["MKL_THREADING_LAYER"] = "GNU"
from ultralytics import YOLO


def yolo_follicle_detection(input, output):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, 'model/follicle/best.pt') 

    model = YOLO(model_file)

    results = model.predict(source=input,
                            conf=0.25,                      # 置信度阈值
                            iou=0.5,                        # IoU 阈值
                            imgsz=800,                      
                            device=0,                       
                            max_det=1000,                  
                            project=output,                  
                            name='predict',                 
                            augment=False,                  # 启用推理时增强
                            agnostic_nms=False,             # 启用类无关的NMS
                            classes=None,                   # 指定要检测的类别
                            retina_masks=False,             # 使用高分辨率分割掩码
                            embed=None,                     # 提取特征向量层
                            show=False,                     # 是否显示推理图像
                            save=True,                      # 保存推理结果
                            save_txt=True,                  # 保存检测结果到文本文件
                            save_conf=True,                 # 保存置信度到文本文件
                            save_crop=False,                # 保存裁剪的检测对象图像
                            show_labels=True,               # 显示检测的标签
                            show_conf=True,                 # 显示检测置信度
                            show_boxes=True,                # 显示检测框
                            line_width=2,                   # 设置边界框的线条宽度
                            verbose=False,                   # 显示输出信息
                        )
    
    return 0



def yolo_number_detection(input, output): 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, 'model/number/best.pt') 

    model = YOLO(model_file)
    
    results = model.predict(source=input,                   # 数据来源，可以是文件夹、图片路径、视频、URL，或设备ID（如摄像头）
                            conf=0.3,                       # 置信度阈值
                            iou=0.5,                        # IoU 阈值
                            imgsz=800,                      # 图像大小
                            half=False,                     # 使用半精度推理
                            device=1,                       # 使用设备，None 表示自动选择，比如'cpu','0'
                            max_det=1000,                   # 最大检测数量
                            project=output,                  
                            name='predict',
                            visualize=False,                # 可视化模型特征
                            augment=False,                  # 启用推理时增强
                            agnostic_nms=False,             # 启用类无关的NMS
                            classes=None,                   # 指定要检测的类别
                            retina_masks=False,             # 使用高分辨率分割掩码
                            embed=None,                     # 提取特征向量层
                            show=False,                     # 是否显示推理图像
                            save=True,                      # 保存推理结果
                            save_txt=True,                  # 保存检测结果到文本文件
                            save_conf=False,                # 保存置信度到文本文件
                            save_crop=False,                # 保存裁剪的检测对象图像
                            show_labels=True,               # 显示检测的标签
                            show_conf=True,                 # 显示检测置信度
                            show_boxes=True,                # 显示检测框
                            line_width=1,                   # 设置边界框的线条宽度，比如2，4
                            verbose=False, 
                        )
    
    return 0