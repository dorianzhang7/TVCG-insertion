import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import os
from tqdm import tqdm
import cv2
import clip
from test_bench_dataset import COCOImageDataset
from test_bench_dataset import TestImageDataset
from einops import rearrange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel,CLIPModel
import torchvision

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--result_dir', type=str, default=None,)
parser.add_argument('--pbe_dir', type=str, default=None,)
parser.add_argument('--mask_dir', type=str, default=None,)
parser.add_argument('--device', type=str, default=None, help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--reference_path', type=str, default=None)
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--output_name', type=str, default=None)

opt = parser.parse_args()
args={}

# # test_dataset=COCOImageDataset(test_bench_dir='test_bench',result_dir=opt.result_dir)
# test_dataset=TestImageDataset(result_dir=opt.result_dir, mask_dir=opt.mask_dir)
# test_dataloader= torch.utils.data.DataLoader(test_dataset,
#                                     batch_size=1,
#                                     num_workers=1,
#                                     pin_memory=True,
#                                     shuffle=False,#sampler=train_sampler,
#                                     drop_last=True)
#
#
# clip_model,preprocess = clip.load("ViT-B/32", device="cuda")
#
# # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
#
# # clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda').eval()
# sum=0
# count=0
# ref_p = Image.open(opt.reference_path).convert("RGB").resize((224, 224))
# ref_tensor = get_tensor_clip()(ref_p)
# ref_image_tensor = ref_tensor.unsqueeze(0)
#
# excel = np.array([])
#
# for crop_tensor in tqdm(test_dataloader):
#     crop_tensor=crop_tensor.to('cuda')
#     ref_image_tensor=ref_image_tensor.to('cuda')
#     result_feat = clip_model.encode_image(crop_tensor)
#     ref_feat = clip_model.encode_image(ref_image_tensor)
#
#     # result_feat = dinov2_vitg14(crop_tensor.float())
#     # ref_feat = dinov2_vitg14(ref_image_tensor.float())
#
#     result_feat=result_feat.to('cpu')
#     ref_feat=ref_feat.to('cpu')
#
#     # similarity = abs(clip_model.encode_image(crop_tensor) - clip_model.encode_image(ref_image_tensor)).squeeze().sum()
#
#     result_feat = result_feat / result_feat.norm(dim=-1, keepdim=True)
#     ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
#     # @ 表示矩阵乘法
#     similarity = (100.0 * result_feat @ ref_feat.T)
#
#     excel = np.append(excel, similarity.item())
#
#     print(count, similarity.item())
#     sum = sum + similarity.item()
#     count = count + 1
#
# excel = np.append(excel, sum/count)
# print(sum/count)
# df = pd.DataFrame(excel)
# if not os.path.exists(opt.output_path):
#     os.makedirs(opt.output_path)
# # df.to_excel(os.path.join(opt.output_path, opt.output_name + '.xlsx'), index=False)



# test_dataset_our = TestImageDataset(result_dir=opt.result_dir, mask_dir=opt.mask_dir)
# test_dataloader_our = torch.utils.data.DataLoader(test_dataset_our,
#                                     batch_size=1,
#                                     num_workers=1,
#                                     pin_memory=True,
#                                     shuffle=False,#sampler=train_sampler,
#                                     drop_last=True)
#
# test_dataset_pbe = TestImageDataset(result_dir=opt.pbe_dir, mask_dir=opt.mask_dir)
# test_dataloader_pbe = torch.utils.data.DataLoader(test_dataset_pbe,
#                                     batch_size=1,
#                                     num_workers=1,
#                                     pin_memory=True,
#                                     shuffle=False,#sampler=train_sampler,
#                                     drop_last=True)
#
# clip_model,preprocess = clip.load("ViT-B/32", device="cuda")
#
# # clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to('cuda').eval()
# sum=0
# count=0
# ref_p = Image.open(opt.reference_path).convert("RGB").resize((224, 224))
# ref_tensor = get_tensor_clip()(ref_p)
# ref_image_tensor = ref_tensor.unsqueeze(0)
#
# for crop_tensor_our, crop_tensor_pbe in zip(tqdm(test_dataloader_our), tqdm(test_dataloader_pbe)):
#     crop_tensor_our = crop_tensor_our.to('cuda')
#     crop_tensor_pbe = crop_tensor_pbe.to('cuda')
#     ref_image_tensor = ref_image_tensor.to('cuda')
#     result_feat_our = clip_model.encode_image(crop_tensor_our)
#     result_feat_pbe = clip_model.encode_image(crop_tensor_pbe)
#     ref_feat = clip_model.encode_image(ref_image_tensor)
#
#     result_feat_our = result_feat_our.to('cpu')
#     result_feat_pbe = result_feat_pbe.to('cpu')
#     ref_feat=ref_feat.to('cpu')
#
#     result_feat_our = result_feat_our / result_feat_our.norm(dim=-1, keepdim=True)
#     result_feat_pbe = result_feat_pbe / result_feat_pbe.norm(dim=-1, keepdim=True)
#     ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
#     # @ 表示矩阵乘法
#     similarity_our = (100.0 * result_feat_our @ ref_feat.T)
#     similarity_pbe = (100.0 * result_feat_pbe @ ref_feat.T)
#     print(count, similarity_our.item() - similarity_pbe.item())
#     sum=sum+(similarity_our.item() - similarity_pbe.item())
#     count=count+1
# print(sum/count)



import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch
import os
from tqdm import tqdm
import cv2
import clip
from test_bench_dataset import TestImageDataset
from einops import rearrange
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel,CLIPModel
import torchvision

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument('--device', type=str, default=None, help='Device to use. Like cuda, cuda:0 or cpu')

opt = parser.parse_args()
args={}

clip_model,preprocess = clip.load("ViT-B/32")
clip_model = clip_model.cuda(1)

dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
dinov2_vitg14 = dinov2_vitg14.cuda(1)

# root_dir = "/workspace/exp"
# dir_list = os.listdir(root_dir)

# root_mask_dir = "/workspace/data/InsertSet/test_bench"
#
# root_reference_dir = "/workspace/data/InsertSet/id_image_our"
#

for id_excel_save in os.listdir("/workspace/exp_final/ife_final"):
# id_excel_save = 'castle'

    excel_dir = '/workspace/evaluate/clip_dinov2_' + id_excel_save
    # #
    # id_excel_save_list = os.listdir(root_reference_dir)
    # for id_excel_save in id_excel_save_list:
    #
    #     for method in dir_list:
    #         sum_all = 0
    #         count_all = 0
    #         method_dir = os.path.join(root_dir, method)
    #         # for id in os.listdir(method_dir):
    #
    #         id = id_excel_save
    #         img_dir = os.path.join(method_dir, id, 'results')
    #         mask_dir = os.path.join(root_mask_dir, id, 'mask')
    #         reference_dir = os.path.join(root_reference_dir, id, 'test1.jpg')

    method = "ife"

    img_dir = "/workspace/exp_final/ife_final/" + id_excel_save + "/results/"
    mask_dir = "/workspace/data/InsertSet/test_bench_final/" + id_excel_save + "/mask/"
    reference_dir = "/workspace/data/InsertSet/id_image_our/" + id_excel_save + "/test1.jpg"

    test_dataset = TestImageDataset(result_dir=img_dir, mask_dir=mask_dir)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, pin_memory=True, shuffle=False, drop_last=True)

    ref_p = Image.open(reference_dir).convert("RGB").resize((224, 224))
    ref_tensor = get_tensor_clip()(ref_p)
    ref_image_tensor = ref_tensor.unsqueeze(0)

    i = 0
    sum_clip = 0
    count_clip = 0
    sum_dinov2 = 0
    count_dinov2 = 0

    excel_save = []
    excel_id = np.array(['id_img'])
    excel_clip = np.array(['clip_' + method])
    excel_dinov2 = np.array(['dinov2_' + method])

    for crop_tensor in tqdm(test_dataloader):

        excel_id = np.append(excel_id, int(test_dataloader.dataset.result_id_name[i][-7:-4]))

        crop_tensor = crop_tensor.to('cuda:1')
        ref_image_tensor = ref_image_tensor.to('cuda:1')

        # clip
        # -------------------------------------------------------------------------------------------
        result_feat_clip = clip_model.encode_image(crop_tensor)
        ref_feat_clip = clip_model.encode_image(ref_image_tensor)

        result_feat_clip = result_feat_clip.to('cpu')
        ref_feat_clip = ref_feat_clip.to('cpu')

        result_feat_clip = result_feat_clip / result_feat_clip.norm(dim=-1, keepdim=True)
        ref_feat_clip = ref_feat_clip / ref_feat_clip.norm(dim=-1, keepdim=True)
        # @ 表示矩阵乘法
        similarity_clip = (100.0 * result_feat_clip @ ref_feat_clip.T)

        excel_clip = np.append(excel_clip, similarity_clip.item())

        # print(count, similarity.item())
        sum_clip = sum_clip + similarity_clip.item()
        count_clip = count_clip + 1

        # dinov2
        # -------------------------------------------------------------------------------------------
        result_feat_dinov2 = dinov2_vitg14(crop_tensor.float())
        ref_feat_dinov2 = dinov2_vitg14(ref_image_tensor.float())

        result_feat_dinov2 = result_feat_dinov2.to('cpu')
        ref_feat_dinov2 = ref_feat_dinov2.to('cpu')

        result_feat_dinov2 = result_feat_dinov2 / result_feat_dinov2.norm(dim=-1, keepdim=True)
        ref_feat_dinov2 = ref_feat_dinov2 / ref_feat_dinov2.norm(dim=-1, keepdim=True)
        # @ 表示矩阵乘法
        similarity_dinov2 = (100.0 * result_feat_dinov2 @ ref_feat_dinov2.T)

        excel_dinov2 = np.append(excel_dinov2, similarity_dinov2.item())

        # print(count, similarity.item())
        sum_dinov2 = sum_dinov2 + similarity_dinov2.item()
        count_dinov2 = count_dinov2 + 1

        i = i + 1

    excel_id = np.append(excel_id, 'average')
    excel_save.append(excel_id)

    print('clip_' + method +":")
    print(sum_clip/count_clip)
    print('\n')

    excel_clip = np.append(excel_clip, sum_clip / count_clip)
    excel_save.append(excel_clip)

    print('dinov2_' + method + ":")
    print(sum_dinov2/count_dinov2)
    print('\n')

    excel_dinov2 = np.append(excel_dinov2, sum_dinov2 / count_dinov2)
    excel_save.append(excel_dinov2)
    #
    excel_save = np.array(excel_save)
    df = pd.DataFrame(excel_save.transpose())
    if not os.path.exists(os.path.join(excel_dir)):
        os.makedirs(os.path.join(excel_dir))
    df.to_excel(os.path.join(excel_dir, id_excel_save + '.xlsx'), index=False)




