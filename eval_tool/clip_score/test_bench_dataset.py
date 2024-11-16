from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import imp

import os
from io import BytesIO
import json
import logging
import base64
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw
import torch.utils.data as data
import json
import time
import cv2
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
import albumentations as A
import clip
import bezier



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


class COCOImageDataset(data.Dataset):
    def __init__(self,test_bench_dir,result_dir):

        self.test_bench_dir=test_bench_dir
        self.result_dir=result_dir
        self.id_list=np.load('test_bench/id_list.npy')
        self.id_list=self.id_list.tolist()
        print("length of test bench",len(self.id_list))
        self.length=len(self.id_list)

       

    
    def __getitem__(self, index):
        result_path=os.path.join(os.path.join(self.result_dir,str(self.id_list[index]).zfill(12)+'.png'))
        result_p = Image.open(result_path).convert("RGB")
        result_tensor = get_tensor_clip()(result_p)

        ### Get reference
        ref_img_path=os.path.join(os.path.join(self.test_bench_dir,'Ref_3500',str(self.id_list[index]).zfill(12)+'_ref.png'))
        ref_img=Image.open(ref_img_path).resize((224,224)).convert("RGB")
        ref_image_tensor=get_tensor_clip()(ref_img)

   
        ### bbox mask
        mask_path=os.path.join(os.path.join(self.test_bench_dir,'Mask_bbox_3500',str(self.id_list[index]).zfill(12)+'_mask.png'))
        mask_img=cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        idx0 = np.nonzero(mask_img.ravel()==255)[0]
        idxs = [idx0.min(), idx0.max()]
        out = np.column_stack(np.unravel_index(idxs,mask_img.shape))
        crop_tensor=result_tensor[:,out[0][0]:out[1][0],out[0][1]:out[1][1]]
        crop_tensor=T.Resize([224,224])(crop_tensor)

    
        return crop_tensor,ref_image_tensor



    def __len__(self):
        return self.length


class TestImageDataset(data.Dataset):
    def __init__(self, result_dir, mask_dir):
        self.result_dir = result_dir
        self.mask_dir = mask_dir

        self.result_id_name = []
        self.mask_id_name = []
        id_result_list = os.listdir(result_dir)

        for file_name in id_result_list:
            self.result_id_name.append(os.path.join(result_dir, file_name))
            self.mask_id_name.append(os.path.join(mask_dir, file_name[:4] + '_mask.png'))

        self.length = self.result_id_name.__len__()
        # self.length = 100


    def __getitem__(self, index):
        result_path = self.result_id_name[index]

        # result_path = os.path.join(os.path.join(self.result_dir, str(index).zfill(4) + '.png'))
        result_p = Image.open(result_path).convert("RGB")
        result_tensor = get_tensor_clip()(result_p)

        ### bbox mask
        mask_path = self.mask_id_name[index]

        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 将mask之外的区域去除
        result_tensor = result_tensor * (mask_img / 255)

        # 寻找mask区域的最小外接矩阵
        thresh = cv2.Canny(mask_img, 128, 256)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = np.copy(mask_img)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # 绘制矩形
            cv2.rectangle(img, (x, y + h), (x + w, y), (0, 255, 255))

        crop_tensor = result_tensor[:, y:y + h, x:x + w]  # 根据内接矩形的顶点切割图片
        [b, h, w] = crop_tensor.size()

        if h == w:
            return T.Resize([224, 224])(crop_tensor)

        length = abs(h - w)
        if h < w:
            pad1 = torch.zeros(3, int(np.round(length/2, 0)),  w, dtype=torch.float64)
            pad2 = torch.zeros(3, np.round(int(length/2), 0),  w, dtype=torch.float64)
            crop_tensor = torch.cat((pad1, crop_tensor), 1)
            crop_tensor = torch.cat((crop_tensor, pad2), 1)

        if h > w:
            pad1 = torch.zeros(3, h, int(np.round(length/2, 0)), dtype=torch.float64)
            pad2 = torch.zeros(3, h, np.round(int(length / 2), 0), dtype=torch.float64)
            crop_tensor = torch.cat((pad1, crop_tensor), 2)
            crop_tensor = torch.cat((crop_tensor, pad2), 2)

        return T.Resize([224, 224])(crop_tensor)

    def __len__(self):
        return self.length


