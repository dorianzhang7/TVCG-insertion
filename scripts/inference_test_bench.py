import sys
sys.path.append("/workspace/code/fusion-diffusion3.0/")		# 保证在终端运行时，可以被检索到目录
import argparse, os, sys, glob
# import sys
# # 获取根目录
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # 将根目录添加到path中
# sys.path.append(BASE_DIR)

import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
# from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import albumentations as A
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from imwatermark import WatermarkEncoder
# from ldm.data.test_bench_dataset import COCOImageDataset
from ldm.data.test_bench_dataset import TestImageDataset
import clip
from torchvision.transforms import Resize
from transformers import CLIPVisionModel
# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

wm = "Paint-by-Example"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photograph of an astronaut riding a horse",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--id_class",
        type=str,
        default="",
        help="",
    )
    parser.add_argument(
        "--delta_ckpt",
        type=str,
        default=None,
        help="path to delta checkpoint of fine-tuned diffusion block",
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(  # 插入对象的few-shot图像和bbox路径
        "--id_dir",
        type=str,
        default=''
    )
    parser.add_argument(
        "--test_bench_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--list_dir",
        type=str,
        default=None
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    torch.cuda.set_device(opt.gpu)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    if opt.delta_ckpt is not None:
        delta_st = torch.load(opt.delta_ckpt, map_location=device)
        delta_st = delta_st['state_dict']
        model.load_state_dict(delta_st, strict=False)

    model = model.to(device)
    model.eval()

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    result_path = os.path.join(outpath, "results")
    grid_path=os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
 

    # test_dataset=COCOImageDataset(test_bench_dir='/home/zq/data2/zq/code/fusion-diffusion3.0/results/test_bench/')

    test_dataset = TestImageDataset(test_bench_dir=opt.test_bench_dir, list_dir=opt.list_dir)
    test_dataloader= torch.utils.data.DataLoader(test_dataset, 
                                        batch_size=batch_size, 
                                        num_workers=1,
                                        pin_memory=True,
                                        shuffle=False,#sampler=train_sampler, 
                                        drop_last=True)


    # 获取输入id图像的特征 self.prior_feature
    # -----------------------------------------------------------------------------------------------------------------
    bbox_real_path_list = []
    bbox_dir = os.path.join(opt.id_dir, opt.id_class, "bbox")
    # bbox_dir = os.path.join("/home/zq/data2/code/fusion-diffusion3.0/data/", opt.id_class, "bbox")
    per_dir_file_list = os.listdir(bbox_dir)
    for file_name in per_dir_file_list:
        bbox_real_path_list.append(os.path.join(bbox_dir, file_name))

    clip_feature = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
    # 冻结clip参数
    clip_feature = clip_feature.eval()
    for param in clip_feature.parameters():
        param.requires_grad = False
    resize = A.Resize(height=224, width=224)

    img_input = []
    symbol = 1
    for i in range(len(bbox_real_path_list)):
        if isinstance(bbox_real_path_list[i], tuple):
            bbox_path = bbox_real_path_list[i][0]
        else:
            bbox_path = bbox_real_path_list[i]

        img_path = bbox_path.replace('bbox', 'data', 1).replace('txt', 'jpg', 1)

        id_class_bbox_list = []
        with open(bbox_path) as f:
            line = f.readline()
            while line:
                line_split = line.strip('\n').split(" ")
                bbox_temp = []
                for j in range(4):
                    bbox_temp.append(int(float(line_split[j])))
                id_class_bbox_list.append(bbox_temp)
                line = f.readline()
        id_bbox = id_class_bbox_list[0]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_f = img[id_bbox[1]:id_bbox[3], id_bbox[0]:id_bbox[2], :]

        img_f = resize(image=img_f)
        img_input.append(img_f['image'])

        # # -----------------------------------------
        # # 针对单一光源进行图像增强和特征提取
        # rotate_num = i / len(bbox_real_path_list)
        #
        # img_f_1 = resize(image=img_f)
        # if symbol == -1:
        #     horizontalfilp = A.HorizontalFlip(p=1)
        #     img_f_1 = horizontalfilp(image=img_f_1['image'])
        # rotate = A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=50 * rotate_num, p=1)
        # img_f_1 = rotate(image=img_f_1['image'])
        # img_input.append(img_f_1['image'])
        #
        # symbol *= -1
        # # -----------------------------------------

    inputs = torch.Tensor(np.array(img_input))

    inputs = torch.transpose(inputs, 1, 3)

    feature_inputs = clip_feature(inputs)
    prior_feature = feature_inputs.last_hidden_state  # num * 257 * 1024
    #------------------------------------------------------------------------------------------------------------------


    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext

    opt.reference_path = opt.id_dir + opt.id_class + "/test.jpg"

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for test_batch, test_model_kwargs,segment_id_batch in test_dataloader:
                    test_model_kwargs={n:test_model_kwargs[n].to(device,non_blocking=True) for n in test_model_kwargs }

                    ref_p = Image.open(opt.reference_path).convert("RGB").resize((224, 224))
                    ref_tensor = get_tensor_clip()(ref_p)
                    ref_tensor = ref_tensor.unsqueeze(0)

                    # ref_tensor = torch.randn_like(ref_tensor)

                    ref_tensor = ref_tensor.to(device)

                    c_dict = {'c_image': ref_tensor, 'bbox_list': prior_feature.to(device)}

                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector.repeat(test_batch.shape[0],1,1)

                    c = model.get_learned_conditioning(c_dict)
                    if c.shape[-1]==1024:
                        c = model.proj_out(c)
                    if len(c.shape)==2:
                        c = c.unsqueeze(1)
                    inpaint_image=test_model_kwargs['inpaint_image']
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-1],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    x_checked_image_torch = x_samples_ddim

                    # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    #
                    # x_checked_image=x_samples_ddim
                    # x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
                    def un_norm_clip(x):
                        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
                        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
                        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
                        return x

                    if not opt.skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):
                            

                            all_img=[]
                            all_img.append(un_norm(test_batch[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            ref_img=ref_tensor.squeeze(1)
                            ref_img=Resize([512,512])(ref_img)
                            all_img.append(un_norm_clip(ref_img[i]).cpu())
                            all_img.append(x_sample.cpu())
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img.save(os.path.join(grid_path, 'grid-'+segment_id_batch[i]+'.png'))

                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(result_path, segment_id_batch[i]+".png"))
                            
                            mask_save=255.*rearrange(un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                            mask_save= cv2.cvtColor(mask_save,cv2.COLOR_GRAY2RGB)
                            mask_save = Image.fromarray(mask_save.astype(np.uint8))
                            mask_save.save(os.path.join(sample_path, segment_id_batch[i]+"_mask.png"))
                            GT_img=255.*rearrange(all_img[0], 'c h w -> h w c').numpy()
                            GT_img = Image.fromarray(GT_img.astype(np.uint8))
                            GT_img.save(os.path.join(sample_path, segment_id_batch[i]+"_GT.png"))
                            inpaint_img=255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
                            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
                            inpaint_img.save(os.path.join(sample_path, segment_id_batch[i]+"_inpaint.png"))
                            ref_img=255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(sample_path, segment_id_batch[i]+"_ref.png"))
                            base_count += 1



                    if not opt.skip_grid:
                        all_samples.append(x_checked_image_torch)


    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
