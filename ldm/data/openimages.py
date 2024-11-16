from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image, ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn as nn
import copy
import math
from functools import partial
import albumentations as A
import bezier
from transformers import CLIPVisionModel


def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))


def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)


def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                            (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


class OpenImageDataset(data.Dataset):
    def __init__(self, state, aug=True, arbitrary_mask_percent=0, **args
                 ):
        self.aug = aug
        self.state = state
        self.args = args
        self.repeat = 0
        self.arbitrary_mask_percent = arbitrary_mask_percent
        self.kernel = np.ones((1, 1), np.uint8)
        self.size = self.args['image_size']

        bad_list = [
        ]

        self.bbox_real_path_list = []
        self.bbox_reg_path_list = []
        if state == "train":

            bbox_dir = os.path.join(args['dataset_dir'], 'reg_bbox')
            per_dir_file_list = os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_reg_path_list.append(os.path.join(bbox_dir, file_name))

            bbox_dir = os.path.join(args['dataset_dir'], 'bbox')
            per_dir_file_list = os.listdir(bbox_dir)
            for file_name in per_dir_file_list:
                if file_name not in bad_list:
                    self.bbox_real_path_list.append(os.path.join(bbox_dir, file_name))


        self.feature = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        # 冻结clip参数
        self.feature = self.feature.eval()
        for param in self.feature.parameters():
            param.requires_grad = False
        # 统一将先验图像设置为 224 * 224
        self.resize = A.Resize(height=224, width=224)

        # 获取输入id图像的特征 self.prior_feature
        # --------------------------------------------------------------------------------------------------------
        img_input = []
        symbol = 1
        for i in range(len(self.bbox_real_path_list)):
            if isinstance(self.bbox_real_path_list[i], tuple):
                bbox_path = self.bbox_real_path_list[i][0]
            else:
                bbox_path = self.bbox_real_path_list[i]
            file_name = os.path.splitext(os.path.basename(bbox_path))[0] + '.jpg'

            img_path = bbox_path.replace('bbox', 'data', 1).replace('txt', 'jpg', 1)

            bbox_list = []
            with open(bbox_path) as f:
                line = f.readline()
                while line:
                    line_split = line.strip('\n').split(" ")
                    bbox_temp = []
                    for j in range(4):
                        bbox_temp.append(int(float(line_split[j])))
                    bbox_list.append(bbox_temp)
                    line = f.readline()
            bbox = random.choice(bbox_list)

            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_f = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

            img_f = self.resize(image=img_f)
            img_input.append(img_f['image'])


        inputs = torch.Tensor(np.array(img_input))

        inputs = torch.transpose(inputs, 1, 3)

        feature_inputs = self.feature(inputs)
        self.prior_feature = feature_inputs.last_hidden_state  # num * 257 * 1024
        # ---------------------------------------------------------------------------------------------------------

        self.bbox_reg_path_list.sort()
        self.length2 = len(self.bbox_reg_path_list)
        # self.length2 = 0

        self.bbox_real_path_list.sort()
        self.length1 = len(self.bbox_real_path_list)

    def __len__(self):
        if self.length2 > 0:
            return 2 * self.length2
        elif self.repeat > 0:
            return self.length1 * self.repeat
        else:
            return self.length1

        # return self.length1 + self.length2

    def __getitem__(self, index):

        # 基于id原图像的数据增强 提取前景图像进行的变换 与另一个id图像的背景结合（组合方式更多，并且增加变换的随机性）
        if index >= self.length2 or self.length2 == 0:
            bbox_path = self.bbox_real_path_list[index % self.length1]
            file_name = os.path.splitext(os.path.basename(bbox_path))[0] + '.jpg'
            # dir_name = bbox_path.split('/')[-2]

            img_dir = os.path.join(self.args['dataset_dir'], 'data')  # /test

            img_path = os.path.join(img_dir, file_name)

            bbox_list = []
            with open(bbox_path) as f:
                line = f.readline()
                while line:
                    line_split = line.strip('\n').split(" ")
                    bbox_temp = []
                    for i in range(4):
                        bbox_temp.append(int(float(line_split[i])))
                    bbox_list.append(bbox_temp)
                    line = f.readline()

            # 得到输入图像和其bbox范围
            bbox = random.choice(bbox_list)
            img_p = Image.open(img_path).convert("RGB")

            # 随机在相同id的图像中选择一张前景图像作为参考图像
            per_dir_file_list = os.listdir(img_dir)

            if self.length1 > 1:
                img_dir_id = np.random.randint(0, self.length1-1)

            # 如果输入图像只有一张id类型的，这里不再进行背景的随机组合，用原图像即可
            else:
                img_dir_id = 0

            ref_img_path = os.path.join(img_dir, per_dir_file_list[img_dir_id])

            img_p_np = cv2.imread(ref_img_path)
            img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)

            bbox_name = os.path.splitext(os.path.basename(ref_img_path))[0] + '.txt'
            ref_bbox_path = os.path.join(self.args['dataset_dir'], 'bbox', bbox_name)
            ref_bbox_list = []
            with open(ref_bbox_path) as fr:
                line = fr.readline()
                while line:
                    line_split = line.strip('\n').split(" ")
                    bbox_temp = []
                    for i in range(4):
                        bbox_temp.append(int(float(line_split[i])))
                    ref_bbox_list.append(bbox_temp)
                    line = fr.readline()
            ref_bbox = random.choice(ref_bbox_list)

            # ref图像分割范围边缘延拓 防止分割算法误差造成前景物体缺陷
            ref_bbox_pad = copy.copy(ref_bbox)

            ref_image_tensor = img_p_np[ref_bbox_pad[1]:ref_bbox_pad[3], ref_bbox_pad[0]:ref_bbox_pad[2], :]

            r = np.random.random()
            random_trans = A.Compose([
                A.Resize(height=224, width=224),
                A.HorizontalFlip(p=0.7*r),
                A.Rotate(limit=20),
                A.Blur(p=0.4*r),
                A.ElasticTransform(p=0.4*r)
            ])

            # ************************************
            ref_image_tensor = random_trans(image=ref_image_tensor)

            ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor = get_tensor_clip()(ref_image_tensor)


        # # 基于ref的数据增强 只将本数据前景部分进行适当变换后重组 得到新的数据

        else:
            bbox_path = self.bbox_reg_path_list[index]

            file_name = os.path.splitext(os.path.basename(bbox_path))[0] + '.jpg'

            # dir_name = bbox_path.split('/')[-2]
            img_path = os.path.join(self.args['dataset_dir'], 'reg_data', file_name)

            bbox_list = []
            with open(bbox_path) as f:
                line = f.readline()
                while line:
                    line_split = line.strip('\n').split(" ")
                    bbox_temp = []
                    for i in range(4):
                        bbox_temp.append(int(float(line_split[i])))
                    bbox_list.append(bbox_temp)
                    line = f.readline()
            bbox = random.choice(bbox_list)
            img_p = Image.open(img_path).convert("RGB")

            ### Get reference image
            bbox_pad = copy.copy(bbox)
            bbox_pad[0] = bbox[0] - min(10, bbox[0] - 0)
            bbox_pad[1] = bbox[1] - min(10, bbox[1] - 0)
            bbox_pad[2] = bbox[2] + min(10, img_p.size[0] - bbox[2])
            bbox_pad[3] = bbox[3] + min(10, img_p.size[1] - bbox[3])
            img_p_np = cv2.imread(img_path)
            img_p_np = cv2.cvtColor(img_p_np, cv2.COLOR_BGR2RGB)
            ref_image_tensor = img_p_np[bbox_pad[1]:bbox_pad[3], bbox_pad[0]:bbox_pad[2], :]

            random_trans = A.Compose([
                A.Resize(height=224, width=224),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20),
                A.Blur(p=0.3),
                A.ElasticTransform(p=0.3)
            ])

            ref_image_tensor = random_trans(image=ref_image_tensor)
            ref_image_tensor = Image.fromarray(ref_image_tensor["image"])
            ref_image_tensor = get_tensor_clip()(ref_image_tensor)

        # -------------------------------------------------------------------------------------------------------------
        ### Generate mask
        image_tensor = get_tensor()(img_p)
        W, H = img_p.size

        extended_bbox = copy.copy(bbox)
        left_freespace = bbox[0] - 0
        right_freespace = W - bbox[2]
        up_freespace = bbox[1] - 0
        down_freespace = H - bbox[3]
        extended_bbox[0] = bbox[0] - random.randint(0, int(0.4 * left_freespace))
        extended_bbox[1] = bbox[1] - random.randint(0, int(0.4 * up_freespace))
        extended_bbox[2] = bbox[2] + random.randint(0, int(0.4 * right_freespace))
        extended_bbox[3] = bbox[3] + random.randint(0, int(0.4 * down_freespace))

        prob = random.uniform(0, 1)
        if prob < self.arbitrary_mask_percent:
            mask_img = Image.new('RGB', (W, H), (255, 255, 255))
            bbox_mask = copy.copy(bbox)
            extended_bbox_mask = copy.copy(extended_bbox)
            top_nodes = np.asfortranarray([
                [bbox_mask[0], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[2]],
                [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
            ])
            down_nodes = np.asfortranarray([
                [bbox_mask[2], (bbox_mask[0] + bbox_mask[2]) / 2, bbox_mask[0]],
                [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
            ])
            left_nodes = np.asfortranarray([
                [bbox_mask[0], extended_bbox_mask[0], bbox_mask[0]],
                [bbox_mask[3], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[1]],
            ])
            right_nodes = np.asfortranarray([
                [bbox_mask[2], extended_bbox_mask[2], bbox_mask[2]],
                [bbox_mask[1], (bbox_mask[1] + bbox_mask[3]) / 2, bbox_mask[3]],
            ])
            top_curve = bezier.Curve(top_nodes, degree=2)
            right_curve = bezier.Curve(right_nodes, degree=2)
            down_curve = bezier.Curve(down_nodes, degree=2)
            left_curve = bezier.Curve(left_nodes, degree=2)
            curve_list = [top_curve, right_curve, down_curve, left_curve]
            pt_list = []
            random_width = 5
            for curve in curve_list:
                x_list = []
                y_list = []
                for i in range(1, 19):
                    if (curve.evaluate(i * 0.05)[0][0]) not in x_list and (
                            curve.evaluate(i * 0.05)[1][0] not in y_list):
                        pt_list.append((curve.evaluate(i * 0.05)[0][0] + random.randint(-random_width, random_width),
                                        curve.evaluate(i * 0.05)[1][0] + random.randint(-random_width, random_width)))
                        x_list.append(curve.evaluate(i * 0.05)[0][0])
                        y_list.append(curve.evaluate(i * 0.05)[1][0])
            mask_img_draw = ImageDraw.Draw(mask_img)
            mask_img_draw.polygon(pt_list, fill=(0, 0, 0))
            mask_tensor = get_tensor(normalize=False, toTensor=True)(mask_img)[0].unsqueeze(0)
        else:
            mask_img = np.zeros((H, W))
            mask_img[extended_bbox[1]:extended_bbox[3], extended_bbox[0]:extended_bbox[2]] = 1
            mask_img = Image.fromarray(mask_img)
            mask_tensor = 1 - get_tensor(normalize=False, toTensor=True)(mask_img)

        ### Crop square image
        if W > H:
            left_most = extended_bbox[2] - H
            if left_most < 0:
                left_most = 0
            right_most = extended_bbox[0] + H
            if right_most > W:
                right_most = W
            right_most = right_most - H
            if right_most <= left_most:
                image_tensor_cropped = image_tensor
                mask_tensor_cropped = mask_tensor
            else:
                left_pos = random.randint(left_most, right_most)
                free_space = min(extended_bbox[1] - 0, extended_bbox[0] - left_pos, left_pos + H - extended_bbox[2],
                                 H - extended_bbox[3])
                random_free_space = random.randint(0, int(0.6 * free_space))
                image_tensor_cropped = image_tensor[:, 0 + random_free_space:H - random_free_space,
                                       left_pos + random_free_space:left_pos + H - random_free_space]
                mask_tensor_cropped = mask_tensor[:, 0 + random_free_space:H - random_free_space,
                                      left_pos + random_free_space:left_pos + H - random_free_space]

        elif W < H:
            upper_most = extended_bbox[3] - W
            if upper_most < 0:
                upper_most = 0
            lower_most = extended_bbox[1] + W
            if lower_most > H:
                lower_most = H
            lower_most = lower_most - W
            if lower_most <= upper_most:
                image_tensor_cropped = image_tensor
                mask_tensor_cropped = mask_tensor
            else:
                upper_pos = random.randint(upper_most, lower_most)
                free_space = min(extended_bbox[1] - upper_pos, extended_bbox[0] - 0, W - extended_bbox[2],
                                 upper_pos + W - extended_bbox[3])
                random_free_space = random.randint(0, int(0.6 * free_space))
                image_tensor_cropped = image_tensor[:, upper_pos + random_free_space:upper_pos + W - random_free_space,
                                       random_free_space:W - random_free_space]
                mask_tensor_cropped = mask_tensor[:, upper_pos + random_free_space:upper_pos + W - random_free_space,
                                      random_free_space:W - random_free_space]
        else:
            image_tensor_cropped = image_tensor
            mask_tensor_cropped = mask_tensor

        image_tensor_resize = T.Resize([self.args['image_size'], self.args['image_size']])(image_tensor_cropped)

        mask_tensor_resize = T.Resize([self.args['image_size'], self.args['image_size']])(mask_tensor_cropped)
        inpaint_tensor_resize = image_tensor_resize * mask_tensor_resize

        return {"GT": image_tensor_resize,
                "inpaint_image": inpaint_tensor_resize,
                "inpaint_mask": mask_tensor_resize,
                "ref_imgs": ref_image_tensor,
                "id_feature": self.prior_feature}
