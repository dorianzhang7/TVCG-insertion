import os
import cv2
import pandas as pd
import argparse

import torch
from tqdm import tqdm


# os.mkdir('/home/zq/data2/zq/code/fusion-diffusion2.0/real_reg_test/bbox/test')
dir_list=os.listdir('/home/zq/data2/zq/code/fusion-diffusion2.0/real_reg_cat/data/')
dir_list.sort()
for dir_name in dir_list:
    print(dir_name)
    # if 'test' in dir_name:

    csv_file_path='/home/zq/data2/zq/code/fusion-diffusion2.0/real_reg_cat/test/labels/detections.csv'
    # csv_file_path = '/home/zq/data2/zq/code/fusion-diffusion2.0/real_reg_test/validation-annotations-bbox.csv'

    # elif 'test' in dir_name:
    #     csv_file_path='dataset/open-images/annotations/test-annotations-bbox.csv'
    # else:
    #     csv_file_path='dataset/open-images/annotations/oidv6-train-annotations-bbox.csv'

    download_dir = os.path.join('/home/zq/data2/zq/code/fusion-diffusion2.0/real_reg_cat/data/', dir_name)
    label_dir = os.path.join('/home/zq/data2/zq/code/fusion-diffusion2.0/real_reg_cat/bbox/', dir_name)
    # os.mkdir(label_dir)

    downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
    images_label_list = list(set(downloaded_images_list))
    df_val = pd.read_csv(csv_file_path)
    groups = df_val.groupby(df_val.ImageID)
    num = 0
    for image in tqdm(images_label_list):
        try:
            current_image_path = os.path.join(download_dir, image + '.jpg')
            dataset_image = cv2.imread(current_image_path)
            # print(image)
            boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
            boxes_new=[]

            for box in boxes:
                # 过滤掉图像中占比太大或太小的物体
                if not((box[1]-box[0])*(box[3]-box[2])>0.8 or (box[1]-box[0])*(box[3]-box[2])<0.02):
                    box[0] *= int(dataset_image.shape[1])
                    box[1] *= int(dataset_image.shape[1])
                    box[2] *= int(dataset_image.shape[0])
                    box[3] *= int(dataset_image.shape[0])
                    boxes_new.append([box[0],box[1],box[2],box[3]])

            if len(boxes_new) == 1:
                file_name = str(image.split('.')[0]) + '.txt'
                file_path = os.path.join(label_dir, file_name)
                # print(file_path)
                if os.path.isfile(file_path):
                    f = open(file_path, 'a')
                else:
                    f = open(file_path, 'w')

                for box in boxes_new:
                        # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
                    print(box[0], box[2], box[1], box[3], file=f)
                num += 1

        except Exception as e:
            pass

    print("num:", num)


# -------------------------------------------------------------------------------------------------------------
# dir_list=os.listdir('/home/zq/data2/zq/code/fusion-diffusion2.0/data/cat/data/')
# dir_list.sort()
#
# dic = torch.load('/home/zq/data2/zq/code/fusion-diffusion2.0/data/cat/results.pth')
#
# download_dir = '/home/zq/data2/zq/code/fusion-diffusion2.0/data/cat/data/'
# label_dir = '/home/zq/data2/zq/code/fusion-diffusion2.0/data/cat/bbox/'
#
# downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
# images_label_list = list(set(downloaded_images_list))
#
# num = 0
# for image in tqdm(images_label_list):
#     try:
#         current_image_path = os.path.join(download_dir, image + '.jpg')
#         dataset_image = cv2.imread(current_image_path)
#
#         boxes_temp = []
#         image_path = os.path.join(image + '.jpg')
#         for i in range(len(dic["file_name_list"])):
#             if image_path == dic["file_name_list"][i]:
#                 boxes_temp = dic["bbox_list"][i]
#                 break
#
#         boxes_temp = boxes_temp.cpu().numpy().squeeze()
#
#         # XMin, YMix, XMax, YMax (left top right bottom)
#         # boxes = []
#         # boxes.append([boxes_temp[0], boxes_temp[2], boxes_temp[1], boxes_temp[3]])
#
#         file_name = str(image.split('.')[0]) + '.txt'
#         file_path = os.path.join(label_dir, file_name)
#         # print(file_path)
#         if os.path.isfile(file_path):
#             f = open(file_path, 'a')
#         else:
#             f = open(file_path, 'w')
#
#         box = boxes_temp
#         print(box[0], box[1], box[2], box[3], file=f)
#         num += 1
#
#     except Exception as e:
#         pass
#
# print("num:", num)
