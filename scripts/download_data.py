import torch
import torchvision
import argparse
import cv2

import numpy as np
import sys
import pandas as pd
sys.path.append('./')
import random
import os
import json
import tqdm
import fiftyone.zoo as foz
import torch
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0,'
import fiftyone.zoo


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    # parser.add_argument('--model_path', type=str, default='./checkpoints/model_19.pth', help='model path')
    # parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    parser.add_argument('--obj_class', type=str)
    parser.add_argument('--ref_class', type=str)
    parser.add_argument('--down_initial_dir', type=str)
    parser.add_argument('--down_final_dir', type=str)
    parser.add_argument('--down', type=str, default='test', help='train or test or validation')
    parser.add_argument('--category', type=int, default=-1, help='coco id class')
    parser.add_argument('--nums', type=int, default=200, help='number of download')
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    initial_path = os.path.join(args.down_initial_dir, args.ref_class)
    if not os.path.exists(initial_path):
        os.mkdir(initial_path)

    if args.category == -1:
    # -----------------------------------------------------------------------------------------------------------------
        dataset = foz.load_zoo_dataset(
            "open-images-v6",
            split=args.down,  # 下载训练集
            label_types=["detections"],  # 下载目标检测标注文件
            classes=[args.ref_class],  # 下载数据集中的某几类别
            max_samples=1000,  # 下载图片数目
            only_matching=True,  # 只下载匹配到类别的图片
            dataset_dir=initial_path,  # 下载目录
        )

        csv_file_path = os.path.join(args.down_initial_dir, args.ref_class, args.down, 'labels/detections.csv')

        download_dir = os.path.join(args.down_initial_dir, args.ref_class, args.down, 'data')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
        images_label_list = list(set(downloaded_images_list))
        df_val = pd.read_csv(csv_file_path)
        groups = df_val.groupby(df_val.ImageID)

        num = 0
        for image in tqdm(images_label_list):
            if num < args.nums:
                try:
                    current_image_path = os.path.join(download_dir, image + '.jpg')
                    dataset_image = cv2.imread(current_image_path)
                    # print(image)
                    boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
                    boxes_new = []

                    for box in boxes:
                        # 过滤掉图像中占比太大或太小的物体
                        if not ((box[1] - box[0]) * (box[3] - box[2]) > 0.8 or (box[1] - box[0]) * (
                                box[3] - box[2]) < 0.1):
                            box[0] *= int(dataset_image.shape[1])
                            box[1] *= int(dataset_image.shape[1])
                            box[2] *= int(dataset_image.shape[0])
                            box[3] *= int(dataset_image.shape[0])
                            boxes_new.append([box[0], box[1], box[2], box[3]])

                    if len(boxes_new) == 1:
                        file_name = str(image.split('.')[0]) + '.txt'

                        if not os.path.exists(os.path.join(args.down_final_dir, args.obj_class, 'bbox')):
                            os.makedirs(os.path.join(args.down_final_dir, args.obj_class, 'bbox'))
                        if not os.path.exists(os.path.join(args.down_final_dir, args.obj_class, 'data')):
                            os.makedirs(os.path.join(args.down_final_dir, args.obj_class, 'data'))

                        file_path = os.path.join(args.down_final_dir, args.obj_class, 'bbox', file_name)
                        # print(file_path)
                        # if os.path.isfile(file_path):
                        #     f = open(file_path, 'a')
                        # else:
                        #     f = open(file_path, 'w')
                        f = open(file_path, 'w')

                        for box in boxes_new:
                            # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
                            print(box[0], box[2], box[1], box[3], file=f)
                        num += 1
                        cv2.imwrite(os.path.join(args.down_final_dir, args.obj_class, 'data', image + '.jpg'),  dataset_image)

                except Exception as e:
                    pass
            else:
                break
        print("num:", num)

    else:
        # ---------------------------------------------------------------------------------------------------------------------
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            # "open-images-v6",
            split=args.down,  # 下载训练集
            label_types=["detections"],  # 下载目标检测标注文件
            classes=[args.ref_class],  # 下载数据集中的某几类别
            max_samples=1000,  # 下载图片数目
            only_matching=True,  # 只下载匹配到类别的图片
            dataset_dir=initial_path,  # 初步下载目录
        )

        if not os.path.exists(os.path.join(args.down_initial_dir, args.ref_class, args.down)):
            os.makedirs(os.path.join(args.down_initial_dir, args.ref_class, args.down))

        download_dir = os.path.join(args.down_initial_dir, args.ref_class, args.down, 'data')
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
        images_label_list = list(set(downloaded_images_list))

        instance_json_path = 'instances_' + args.down + '2017.json'
        json_path = os.path.join(args.down_initial_dir, args.ref_class, 'raw', instance_json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            content = json.load(f)
            num = 0
            for image in tqdm(images_label_list):
                if num < 200:
                    try:
                        current_image_path = os.path.join(download_dir, image + '.jpg')
                        dataset_image = cv2.imread(current_image_path)

                        box_new = []

                        for index in content['annotations']:

                            if image.lstrip('0') == str(index['image_id']) and index['category_id'] == args.category and index['iscrowd'] == 0:

                                box = index['bbox']
                                # 过滤掉图像中占比太大或太小的物体
                                image_area = int(dataset_image.shape[0]) * int(dataset_image.shape[1])
                                if not ((box[2] * box[3]) > 0.8 * image_area or (box[2] * box[3]) < 0.1 * image_area):
                                    box_new.append(box[0])
                                    box_new.append(box[1])
                                    box_new.append(box[0] + box[2])
                                    box_new.append(box[1] + box[3])
                                break

                        if not box_new:
                            continue
                        else:
                            file_name = str(image.split('.')[0]) + '.txt'
                            file_path = os.path.join(args.down_final_dir, args.obj_class, 'bbox', file_name)
                            if not os.path.exists(os.path.join(args.down_final_dir, args.obj_class, 'bbox')):
                                os.makedirs(os.path.join(args.down_final_dir, args.obj_class, 'bbox'))
                            if not os.path.exists(os.path.join(args.down_final_dir, args.obj_class, 'data')):
                                os.makedirs(os.path.join(args.down_final_dir, args.obj_class, 'data'))
                            f = open(file_path, 'w')
                            # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
                            print(box_new[0], box_new[1], box_new[2], box_new[3], file=f)
                            num += 1
                            cv2.imwrite(os.path.join(args.down_final_dir, args.obj_class, 'data', image + '.jpg'),  dataset_image)

                    except Exception as e:
                        pass
                else:
                    break
            print("num:", num)



if __name__ == "__main__":
    main()

    # Visualize the dataset in the FiftyOne App
    # session = fiftyone.launch_app(dataset)