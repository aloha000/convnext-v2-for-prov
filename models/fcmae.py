# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn

from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)

from timm.models.layers import trunc_normal_
from .convnextv2_sparse import SparseConvNeXtV2
from .convnextv2 import Block

class FCMAE(nn.Module):
    """ Fully Convolutional Masked Autoencoder with ConvNeXtV2 backbone
    """
    def __init__(
                self,
                img_size=[128,64],
                in_chans=3,
                depths=[3, 3, 9, 3],
                dims=[96, 192, 384, 768],
                decoder_depth=1,
                decoder_embed_dim=512,
                patch_size=32,
                mask_ratio=0.6,
                norm_pix_loss=False):
        super().__init__()

        # configs
        self.img_size = img_size
        self.depths = depths
        self.imds = dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (self.img_size[0] // patch_size) * (self.img_size[1] // patch_size)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = SparseConvNeXtV2(
            in_chans=in_chans, depths=depths, dims=dims, D=2)
        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size ** 2 * in_chans,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight)
            nn.init.constant_(m.linear.bias, 0)
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *self.in_chans)
        """
        p = self.patch_size
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_chans))
        return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_size
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]
        
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    # def gen_random_mask(self, x, mask_ratio):
    #     N = x.shape[0]
    #     L = (x.shape[2] // self.patch_size) * (x.shape[3] // self.patch_size)
    #     len_keep = int(L * (1 - mask_ratio))

    #     noise = torch.randn(N, L, device=x.device)

    #     # sort noise for each sample
    #     ids_shuffle = torch.argsort(noise, dim=1)
    #     ids_restore = torch.argsort(ids_shuffle, dim=1)

    #     # generate the binary mask: 0 is keep 1 is remove
    #     mask = torch.ones([N, L], device=x.device)
    #     mask[:, :len_keep] = 0
    #     # unshuffle to get the binary mask
    #     mask = torch.gather(mask, dim=1, index=ids_restore)
    #     return mask


    def gen_random_mask(self, x, mask_ratio):
        """
        Args:
            x: Input tensor of shape (N, C, H, W)
            mask_ratio: ratio of masked patches
        Returns:
            mask: Pixel-level mask of shape (N, H*W), where 0 is keep and 1 is remove.
        """
        N, C, H, W = x.shape
        p = self.patch_size
        
        # 计算 patch 的网格数量 (h_patches, w_patches)
        h_patches = H // p
        w_patches = W // p
        L = h_patches * w_patches
        
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        print(mask.shape)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # --- 修改开始: 将 Patch Level 还原为 Pixel Level ---

        # 1. 恢复成 2D Patch 网格: (N, L) -> (N, h_patches, w_patches)
        mask = mask.view(N, h_patches, w_patches)
        print(mask.shape)
        
        # 2. 放大回像素尺寸: 
        # 方法 A: 使用 repeat_interleave (推荐，逻辑清晰，无插值误差)
        # (N, h_p, w_p) -> (N, h_p, p, w_p, p) -> (N, H, W)
        mask = mask.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)
        print(mask.shape)
        
        # 3. 展平成 b*n (N, H*W) 格式
        mask = mask.flatten(1) 
        print(mask.shape)
        
        # --- 修改结束 ---

        return mask


    def upsample_mask(self, mask, scale):
        print('fcmae upsample mask #############')
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    # def upsample_mask(self, mask, scale,img_size):
    #     h = img_size[0]
    #     w = img_size[1]
    #     assert len(mask.shape) == 2
    #     print('#######')
    #     print(mask.shape)
        
    #     # p = int(mask.shape[1] ** .5)
    #     H_patch, W_patch = solve_hw(mask.shape[1], h, w)
    #     print('*****')
    #     print(H_patch,W_patch)
    #     print('*****')
    #     return mask.reshape(-1, H_patch, W_patch).\
    #                 repeat_interleave(scale, axis=1).\
    #                 repeat_interleave(scale, axis=2)
    
    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        print('#######fcmae forwad encoder')
        print(mask.shape)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        print(f'x before decoder {x.shape}')
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        print('forward loss')
        print('imgs:',imgs.shape)
        print("pred:", pred.shape)
        print("target:", target.shape)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def convnextv2_atto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnextv2_pico(**kwargs):
    model = FCMAE(
        depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = FCMAE(
        depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = FCMAE(
        depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = FCMAE(
        depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def solve_hw(L, h, w):
    y = math.sqrt(L * w / h)
    x = L / y

    # 强制转成整数
    x = int(round(x))
    y = int(round(y))

    # 校验
    assert x * y == L, f"Invalid factorization: {x}*{y}!={L}"
    assert abs(x / y - h / w) < 1e-6, "Aspect ratio mismatch"

    return x, y