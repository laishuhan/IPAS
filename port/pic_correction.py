import cv2
import numpy as np
def minAreaRect(image_path):

    # 读入图片3通道 [[[255,255,255],[255,255,255]],[[255,255,255],[255,255,255]]]
    image = cv2.imread(image_path)
    # 转为灰度单通道 [[255 255],[255 255]]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 黑白颠倒
    gray = cv2.bitwise_not(gray)
    # 二值化
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    ys, xs = np.where(thresh > 0)
    # 组成坐标[[306  37][306  38][307  38]],里面都是非零的像素
    coords = np.column_stack([xs,ys])
    # 获取最小矩形的信息 返回值(中心点，长宽，角度) 
    rect = cv2.minAreaRect(coords)
    angle = rect[-1] # 最后一个参数是角度
    print(rect,angle) # ((26.8, 23.0), (320.2, 393.9), 63.4)

    box = cv2.boxPoints(rect)
    box = np.int32(cv2.boxPoints(rect))
    print(box) # [[15 181][367  5][510 292][158 468]]

    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    return angle
def rotate_bound(image, angle):
    #获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # 提取旋转矩阵 sin cos 
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # 计算图像的新边界尺寸
    nW = int((h * sin) + (w * cos))
    nH = h
    # 调整旋转矩阵
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    return cv2.warpAffine(image, M, (nW, nH),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
def doc_correction(image_path, output_path):
    """
    对输入的图像进行旋转校正并保存结果。

    参数:
    - image_path: str, 输入图像的文件路径。
    - output_path: str, 旋转后图像的输出文件路径。

    返回:
    - None
    """
    # 获取最小面积矩形的角度
    angle = minAreaRect(image_path)
    
    # 读取图像并进行旋转
    image = rotate_bound(cv2.imread(image_path), 90 - angle)
    
    # 保存旋转后的图像
    cv2.imwrite(output_path, image)
    print(f"图像已保存为 {output_path}")

if __name__ == '__main__':
    # 运行代码
    doc_correction('test.jpg', 'output_test.jpg')