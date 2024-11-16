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

        # 寻找mask区域的最小外接矩阵
        thresh = cv2.Canny(mask2, 128, 256)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_img = np.copy(mask2)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 绘制矩形
            cv2.rectangle(mask_img, (x, y + h), (x + w, y), (0, 255, 255))

        file_path = r"C:\Users\pc\Desktop\foxman_new\bbox\9.txt"
        f = open(file_path, 'w')
        print(x, y, x + w, y + h, file=f)

        cv2.imshow("mask", mask2)

        mask_path = r'C:\Users\pc\Desktop\foxman_new\mask\9.jpg'
        # mask_path = r'C:\Users\pc\Desktop\cat\mask\0049_mask.png'
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
path = r"C:\Users\pc\Desktop\foxman_new\9.jpg"
img_org = cv2.imread(path)
# print('img_org.size:', img_org.shape)  # 打印原始帧图像的通道信息
img = cv2.resize(img_org, (512, 512))

img_path = r"C:\Users\pc\Desktop\foxman_new\data\9.jpg"
cv2.imwrite(img_path, img)

print('img.size:', img.shape)
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
