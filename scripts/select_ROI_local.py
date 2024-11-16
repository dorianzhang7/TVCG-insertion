import os.path

import cv2
import numpy as np
import joblib  # pip install joblib


# 声明鼠标点击函数，从图像中选入ROI的系列点
def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，选择点
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，取消最近一次选择的点
        pts.pop()
    if event == cv2.EVENT_MBUTTONDOWN:  # 中键绘制轮廓
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))

        # 选择多边形
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # 用于求 ROI
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 用于 显示在桌面的图像ss
        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)


        cv2.imshow("mask", mask2)
        # cv2.imwrite(path.replace('image', 'mask', 1), mask2)s

        # mask_path = r'D:\DeepLearning\model\fusion-diffusion3.0\examples\mask\example_53.png'
        mask_path = r'D:\DeepLearning\data\InsertSet\id_image\virtual_object\snowman\mask.jpg'
        cv2.imwrite(mask_path, mask2)

        # cv2.imwrite(path.replace('image', 'mask', 1).replace('img', 'mask', 1), mask2)
        # cv2.imshow("show_img", show_image)
        # ROI = cv2.bitwise_and(mask2, img)
        # cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        # 将pts中的最后一点画出来
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
    if len(pts) > 1:
        # 画线
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 为鼠标点击地方的坐标
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
    cv2.imshow('image', img2)


# 创建图像坐标参数
pts = []  # 用于存放点坐标
# path = r"D:\DeepLearning\model\fusion-diffusion3.0\examples\image\example_6.png"  # 读入图片路径
# path = r"D:\DeepLearning\data\Fusionset\test_bench\rabbit\image\0002_img.png"

# path = r"C:\Users\pc\Desktop\test_set\Dog\test\data\1876493923c4b4d0.jpg"
# path = r"D:\DeepLearning\model\fusion-diffusion3.0\examples\image\example_53.png"
# path = r"D:\DeepLearning\data\COCO\test2017\000000303353.jpg"
# path = r"D:\DeepLearning\data\Fusionset\test_bench\horse\image\0000_img.png"
path = r'D:\DeepLearning\data\InsertSet\id_image\virtual_object\snowman\test.jpg'

# 为了使ROI与实际的点的坐标一致，需要将图片resize成目标大小
# 若是在视频中画ROI，则需要匹配单帧中的实际大小
img_org = cv2.imread(path)
print('img_org.size:', img_org.shape)  # 打印原始帧图像的通道信息
img = cv2.resize(img_org, (512, 512))
print('img.size:', img.shape)
# img_path = r"D:\DeepLearning\data\Fusionset\test_bench\chair\image\0057_img.png"
# cv2.imwrite(img_path, img)
# 将图像窗口与鼠标回调函数绑定
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)
print("[INFO] 单击左键：选择点，单击右键：删除上一次选择的点，单击中键：确定ROI区域")
print("[INFO] 按‘S’确定选择区域并保存")
print("[INFO] 按 ESC 退出")

# 退出与保存
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        # saved_data = {"ROI": pts}
        # joblib.dump(value=saved_data, filename="config.pkl")  # .pkl 为二进制文件
        # print("[INFO] ROI坐标已保存到本地.")
        print("mask已保存到本地.")
        break
cv2.destroyAllWindows()


# 加载保存好的 .pkl 二进制文件中提取的roi坐标
# def Load_Model(filepath):
#     img = cv2.imread(path)
#     model = joblib.load(filepath)
#     print(type(model))
#     print(model)  # 打印坐标点的信息
#     return model

#
# Load_Model('config.pkl')
