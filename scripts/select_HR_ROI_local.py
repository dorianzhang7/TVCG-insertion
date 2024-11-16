import cv2
import numpy as np
import joblib  # pip install joblib


def findcontour(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图像灰度化
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 图像二值化

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找物体轮廓

    return contours, hierarchy

# image, contours, hierarchy = findcontour(original)


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
        cv2.namedWindow('mask', 0)
        cv2.imshow("mask", mask2)
        mask3 = mask2

        contours, hierarchy = findcontour(mask3)

        # contours = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        M = cv2.moments(contours[0])  # 计算轮廓的各阶矩,字典形式
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        # cv2.drawContours(img, contours, 0, 255, -1)  # 绘制轮廓，填充
        m_image = cv2.circle(img, (center_x, center_y), 7, 128, -1)  # 绘制中心点
        # cv2.namedWindow('contour', 0)
        # cv2.imshow("contour", img)

        point_1_x = center_x - 256
        point_1_y = center_y - 256

        point_2_x = center_x + 256
        point_2_y = center_y - 256

        point_3_x = center_x - 256
        point_3_y = center_y + 256

        point_4_x = center_x + 256
        point_4_y = center_y + 256

        log = []
        log.append(point_1_x)
        log.append(point_1_y)
        log.append(point_2_x)
        log.append(point_3_y)

        file_path = r"D:\DeepLearning\model\fusion-diffusion3.0\examples\image\example_101_ROI.txt"
        f = open(file_path, 'w')

        for points in log:
            print(points, file=f)
        # cv2.imwrite(os.path.join(reg_data_dir, image + '.jpg'), dataset_image)

        # 尝试改进为单一mask
        # -------------------------------------------------------------------------------------------------------------
        test = cv2.imread(r'D:\DeepLearning\model\fusion-diffusion3.0\examples\mask\mask_5.png')

        # 轴对称
        # test = np.flip(test, axis=1)
        # 旋转
        # test = np.rot90(test,k=3)


        mask2[point_1_y:point_3_y, point_1_x:point_2_x, :] = test

        cv2.imwrite(path.replace('image', 'mask', 1), mask2)
        # cv2.imshow("show_img", show_image)
        ROI = cv2.bitwise_and(mask2, img)
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
path = r"D:\DeepLearning\model\fusion-diffusion3.0\examples\image\example_101.png"  # 读入图片路径

# 为了使ROI与实际的点的坐标一致，需要将图片resize成目标大小
# 若是在视频中画ROI，则需要匹配单帧中的实际大小
img_org = cv2.imread(path)
print('img_org.size:', img_org.shape)  # 打印原始帧图像的通道信息

# img = cv2.resize(img_org, (512, 512))
# print('img.size:', img.shape)

img = img_org

# 将图像窗口与鼠标回调函数绑定
cv2.namedWindow('image',0)
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
        print("ROI坐标已保存到本地.")
        print("mask已保存到本地.")
        break
cv2.destroyAllWindows()


# # 加载保存好的 .pkl 二进制文件中提取的roi坐标
# def Load_Model(filepath):
#     img = cv2.imread(path)
#     model = joblib.load(filepath)
#     print(type(model))
#     print(model)  # 打印坐标点的信息
#     return model
#
#
# Load_Model('config.pkl')
