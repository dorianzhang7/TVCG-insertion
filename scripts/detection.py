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
import tqdm
# import fiftyone.zoo as foz
import torch
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1,'


id_names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat',
         '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse',
         '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie',
         '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove',
         '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon',
         '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut',
         '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse',
         '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book',
         '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}


def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Faster-rcnn Detection')

    # parser.add_argument('--model_path', type=str, default='./checkpoints/model_19.pth', help='model path')
    # parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--dataset', default='coco', help='model')
    parser.add_argument('--score', type=float, default=0.8, help='objectness score threshold')
    parser.add_argument('--dir', type=str)
    parser.add_argument('--image_dir', type= str, default=None)
    parser.add_argument('--bbox_dir', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--obj_class', type=str)
    parser.add_argument('--ref_class', type=str)
    parser.add_argument('--down_reg_dir', type=str)
    parser.add_argument('--down', type=str, default='test', help='train or test or validation')
    args = parser.parse_args()

    return args


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def main():
    args = get_args()

    image_dir = os.path.join(args.dir, args.obj_class, 'reg_data')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    bbox_dir = os.path.join(args.dir, args.obj_class, 'reg_bbox')
    if not os.path.exists(bbox_dir):
        os.makedirs(bbox_dir)
    save_dir = os.path.join(args.dir, args.obj_class, 'reg_save')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    input = []
    num_classes = 91
    names = id_names

    # Model creating
    print("Creating model")
    model = torchvision.models.detection.__dict__[args.model](num_classes=num_classes, pretrained=True)
    model = model.cuda()

    model.eval()

    dir = image_dir
    images_list = [f for f in os.listdir(dir)]
    # images_list = [f.split('.')[0] for f in os.listdir(dir) if f.endswith('.jpg')]
    for image in images_list:
        image_path = os.path.join(dir, image)
        src_img = cv2.imread(image_path)
        img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().cuda()
        input.append(img_tensor)
        out = model(input)
        del input[0]
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        x1, y1, x2, y2 = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]

        if not os.path.exists(bbox_dir):
            os.mkdir(bbox_dir)

        file_name = str(image.split('.')[0]) + '.txt'
        file_path = os.path.join(bbox_dir, file_name)
        # print(file_path)
        # if os.path.isfile(file_path):
        #     f = open(file_path, 'a')
        # else:
        #     f = open(file_path, 'w')
        f = open(file_path, 'w')

        # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
        print(x1.item(), y1.item(), x2.item(), y2.item(), file=f)

        name = names.get(str(labels[0].item()))
        cv2.rectangle(src_img, (int(x1), int(y1)), (int(x2), int(y2)), random_color(), thickness=2)
        # cv2.putText(src_img, text=name, org=(int(x1), int(y1) + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #             fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=(0, 0, 255))

        # cv2.imshow('result', src_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        image_save_path = os.path.join(save_dir, image)
        cv2.imwrite(image_save_path, src_img)

    # reg_path = os.path.join(args.down_reg_dir, args.ref_class)
    # # if os.path.exists(reg_path):
    # #     pass
    # # else:
    # dataset = foz.load_zoo_dataset(
    #     "coco-2017",
    #     # "open-images-v6",
    #     split=args.down,  # 下载训练集
    #     label_types=["detections"],  # 下载目标检测标注文件
    #     classes=[args.ref_class],  # 下载数据集中的某几类别
    #     max_samples=10,  # 下载图片数目
    #     only_matching=True,  # 只下载匹配到类别的图片
    #     dataset_dir=reg_path,  # 下载目录
    # )
    #
    # csv_file_path = os.path.join(args.down_reg_dir, args.ref_class, args.down, 'labels/detections.csv')
    #
    # download_dir = os.path.join(args.down_reg_dir, args.ref_class, args.down, 'data')
    # if not os.path.exists(download_dir):
    #     os.makedirs(download_dir)
    #
    # reg_data_dir = image_dir[:-4] + 'reg_data'
    # if not os.path.exists(reg_data_dir):
    #     os.mkdir(reg_data_dir)
    # reg_bbox_dir = image_dir[:-4] + 'reg_bbox'
    # # reg_bbox_dir = image_dir.replace('/data/', '/reg_bbox/', 2).replace('/reg_bbox/', '/data/', 1)
    # if not os.path.exists(reg_bbox_dir):
    #     os.mkdir(reg_bbox_dir)
    #
    # downloaded_images_list = [f.split('.')[0] for f in os.listdir(download_dir) if f.endswith('.jpg')]
    # images_label_list = list(set(downloaded_images_list))
    # df_val = pd.read_csv(csv_file_path)
    # groups = df_val.groupby(df_val.ImageID)
    #
    # num = 0
    # for image in tqdm(images_label_list):
    #     if num < 200:
    #         try:
    #             current_image_path = os.path.join(download_dir, image + '.jpg')
    #             dataset_image = cv2.imread(current_image_path)
    #             # print(image)
    #             boxes = groups.get_group(image.split('.')[0])[['XMin', 'XMax', 'YMin', 'YMax']].values.tolist()
    #             boxes_new = []
    #
    #             for box in boxes:
    #                 # 过滤掉图像中占比太大或太小的物体
    #                 if not ((box[1] - box[0]) * (box[3] - box[2]) > 0.8 or (box[1] - box[0]) * (
    #                         box[3] - box[2]) < 0.02):
    #                     box[0] *= int(dataset_image.shape[1])
    #                     box[1] *= int(dataset_image.shape[1])
    #                     box[2] *= int(dataset_image.shape[0])
    #                     box[3] *= int(dataset_image.shape[0])
    #                     boxes_new.append([box[0], box[1], box[2], box[3]])
    #
    #             if len(boxes_new) == 1:
    #                 file_name = str(image.split('.')[0]) + '.txt'
    #                 file_path = os.path.join(reg_bbox_dir, file_name)
    #                 # print(file_path)
    #                 # if os.path.isfile(file_path):
    #                 #     f = open(file_path, 'a')
    #                 # else:
    #                 #     f = open(file_path, 'w')
    #                 f = open(file_path, 'w')
    #
    #                 for box in boxes_new:
    #                     # each row in a file is name of the class_name, XMin, YMix, XMax, YMax (left top right bottom)
    #                     print(box[0], box[2], box[1], box[3], file=f)
    #                 num += 1
    #                 cv2.imwrite(os.path.join(reg_data_dir, image + '.jpg'),  dataset_image)
    #
    #         except Exception as e:
    #             pass
    #     else:
    #         break
    # print("num:", num)


if __name__ == "__main__":
    main()