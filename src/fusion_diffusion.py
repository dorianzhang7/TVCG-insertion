
import torch
from einops import rearrange, repeat
from torch import nn, einsum

import torchvision
from torch.optim.lr_scheduler import LambdaLR
from ldm.models.diffusion.ddpm import LatentDiffusion as LatentDiffusion
from ldm.util import default
from ldm.modules.attention import BasicTransformerBlock as BasicTransformerBlock
from ldm.modules.attention import CrossAttention as CrossAttention
from ldm.util import log_txt_as_img, exists, ismap, isimage, mean_flat, count_params, instantiate_from_config
from torchvision.utils import make_grid
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
import numpy as np

class FusionDiffusion(LatentDiffusion):
    def __init__(self,
                 freeze_model='crossattn-kv',
                 cond_stage_trainable=False,
                 # add_token=False,
                 *args, **kwargs):

        self.freeze_model = freeze_model
        # self.add_token = add_token
        self.cond_stage_trainable = cond_stage_trainable
        super().__init__(cond_stage_trainable=cond_stage_trainable, *args, **kwargs)


        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                # elif not ('attn2.to_k' in x[0]):
                elif not ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0]):
                    x[1].requires_grad = False
                else:
                    # pass
                    x[1].requires_grad = True

        elif self.freeze_model == 'crossattn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' not in x[0]:
                    x[1].requires_grad = False
                elif not 'attn2' in x[0]:
                    x[1].requires_grad = False
                else:
                    x[1].requires_grad = True

        def change_checkpoint(model):
            for layer in model.children():
                if type(layer) == BasicTransformerBlock:
                    layer.checkpoint = False
                else:
                    change_checkpoint(layer)

        change_checkpoint(self.model.diffusion_model)

        def new_forward(self, x, context=None, mask=None):
            h = self.heads
            crossattn = False
            if context is not None:
                crossattn = True

            # 背景（x_0, inpaint, mask）
            q = self.to_q(x)  # q (1*4096*320)
            context = default(context, x)  # context (1*5*768)

            # 对象（few_shot, id）
            k = self.to_k(context)  # k (1*5*320)
            v = self.to_v(context)  # v (1*5*320)

            if crossattn:
                modifier = torch.ones_like(k)
                modifier[:, :1, :] = modifier[:, :1, :]*0.
                k = modifier*k + (1-modifier)*k.detach()
                v = modifier*v + (1-modifier)*v.detach()

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            attn = sim.softmax(dim=-1)

            out = einsum('b i j, b j d -> b i d', attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)

        def change_forward(model):
            for layer in model.children():
                if type(layer) == CrossAttention:
                    bound_method = new_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)
                else:
                    change_forward(layer)

        change_forward(self.model.diffusion_model)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.freeze_model == 'crossattn-kv':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    # if 'attn2.to_k' in x[0]:
                    if 'attn2.to_k' in x[0] or 'attn2.to_v' in x[0]:
                        params += [x[1]]
                        print(x[0])
        elif self.freeze_model == 'crossattn':
            for x in self.model.diffusion_model.named_parameters():
                if 'transformer_blocks' in x[0]:
                    if 'attn2' in x[0]:
                        params += [x[1]]
                        print(x[0])
        else:
            params = list(self.model.parameters())

        print(f"{self.__class__.__name__}: Also optimizing conditioner params!")

        params = params + list(self.cond_stage_model.final_ln.parameters()) + list(self.cond_stage_model.mapper.parameters()) +\
                 list(self.proj_out.parameters()) + list(self.cond_stage_model.prior_encoder.prior_block.parameters())

        # params = params + list(self.proj_out.parameters())

        # params = params + list(self.proj_out.parameters()) + list(self.cond_stage_model.final_ln.parameters()) + list(self.cond_stage_model.mapper.parameters())

        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        params.append(self.learnable_vector)

        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt




