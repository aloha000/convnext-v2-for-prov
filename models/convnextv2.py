# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN




class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=1),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features_keep(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x # global average pooling, (N, C, H, W) -> (N, C, H/4, W/4)

    def forward_features_mean(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward_features_center(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x[:,:,3,3]) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class ConvNeXtV2ForPowerEstimate(nn.Module):
    def __init__(self, convnextv2, mlp,cnn, time_encoder, feature_type="mean"):
        super().__init__()
        self.convnextv2 = convnextv2
        self.mlp = mlp
        self.power_map_cnn = cnn
        self.time_encoder = time_encoder
        for param in self.convnextv2.parameters():
            param.requires_grad = False
            
        # if feature_type == "mean":
        #     self.nwp_encoder = convnextv2.forward_features_mean
        # elif feature_type == "center":
        #     self.nwp_encoder = convnextv2.forward_features_center
        # elif feature_type == "keep":
        #     self.nwp_encoder = convnextv2.forward_features_keep
        self.nwp_encoder = convnextv2.forward_features_keep
    
    def forward(self, x, x_time=None, farm=None):
        # deep feature process
        with torch.no_grad():
            #x = self.convnextv2.forward_features_mean(x)
            x = self.nwp_encoder(x)
        x_time = self.time_encoder(x_time)
        x_time = x_time.unsqueeze(-1).unsqueeze(-1)
        x_time = x_time.expand(x_time.shape[0], x_time.shape[1], x.shape[-2], x.shape[-1])

        x = torch.cat([x, x_time], dim=1)
                
        # concate the x(B, C1, H, W), x_time (C2) per channel
        # the ouptut should be (B, C1+C2, H, W)
        
        # USE CNN TO PROCESS THE MAP FEATURE
        
        # PRDICET THE POWER MAP
        
        
        # PREDICT THE PROVINCE POWER
        
        pred = self.power_map_cnn(x)
        
        return pred

class ConvNeXtV2ForAllFarmPowerEstimate(ConvNeXtV2ForPowerEstimate):
    def __init__(self, convnextv2, mlp, time_encoder, farm_id_encoder, feature_type="mean"):
        super().__init__(convnextv2, mlp, time_encoder, feature_type)
        self.farm_id_encoder = farm_id_encoder
        
    
    def forward(self, x, x_c=None, x_time=None, farm=None):
        # deep feature process
        with torch.no_grad():
            #x = self.convnextv2.forward_features_mean(x)
            x = self.nwp_encoder(x)
        x_time = self.time_encoder(x_time)
        x_farm = self.farm_id_encoder(farm)
                
        x = torch.cat([x, x_c, x_time, x_farm], dim=-1)
        pred = self.mlp(x)
        
        return pred

class ConvNeXtV2N2NPowerEstimate(ConvNeXtV2ForPowerEstimate):
    def __init__(self, convnextv2, transformer, mlp_pre, mlp_post, time_encoder, feature_type="mean"):
        super().__init__(convnextv2, mlp_pre, time_encoder, feature_type)
        self.transformer_encoder = transformer
        self.mlp_post = mlp_post
    
    def forward(self, x, x_c=None, x_time=None, farm=None):
        # Conv2D only accepts 3D or 4D input, so convert x:[B, G, C, H, W] -> [B*G, C, H, W]
        b, g, c, h, w = x.shape
        x = x.view(b*g, c, h, w)
        with torch.no_grad():
            x = self.nwp_encoder(x) #[B*G, output_channels]
        x = x.view(b, g, -1) #[B, G, output_channels]
        # x_c:[B, G, fcmae_input_dim]
        x_time = self.time_encoder(x_time) # x_time:[B, G, time_dim]
        x = torch.cat([x, x_c, x_time], dim=-1)
        x = self.mlp(x)
        x = self.transformer_encoder(x)
        pred = self.mlp_post(x)
        return pred
    
class ConvNeXtV2N2NForAllFarmPowerEstimate(ConvNeXtV2ForPowerEstimate):
    def __init__(self, d_transformer, convnextv2, transformer, mlp_pre, mlp_post, time_encoder, farm_id_encoder, farm_mlp, feature_type="mean"):
        super().__init__(convnextv2, mlp_pre, time_encoder, feature_type)
        self.d_transformer = d_transformer
        self.transformer_encoder = transformer
        self.mlp_post = mlp_post
        self.farm_id_encoder = farm_id_encoder
        self.farm_mlp = farm_mlp
        self.pre_norm = LayerNorm(self.d_transformer, eps=1e-6)
    
    def forward(self, x, x_c=None, x_time=None, farm=None):
        #x:[B, G, C, H, W] -> [B*G, C, H, W]
        b, g, c, h, w = x.shape
        x = x.view(b*g, c, h, w)
        with torch.no_grad():
            x = self.nwp_encoder(x) #[B*G, output_channels]
        x = x.view(b, g, -1) #[B, G, output_channels]
        # x_c:[B, G, fcmae_input_dim]
        x_time = self.time_encoder(x_time) # x_time:[B, G, time_dim]
        x = torch.cat([x, x_c, x_time], dim=-1)
        x = self.mlp(x)
        
        # add special token for farm embedding
        x_farm = self.farm_id_encoder(farm) # x_farm: [B, farm_dim]
        x_farm = x_farm.unsqueeze(1) # x_farm:[B, 1, farm_dim]
        x_farm_token = self.farm_mlp(x_farm) # x_farm_token: [B, 1, d_tfm]
        x_input = torch.cat([x_farm_token, x], dim=1) # [B, 1+group_len, d_tfm]
        
        # 1114: add an extra layer norm before transformer encoder
        x_input = self.pre_norm(x_input)
        x = self.transformer_encoder(x_input)
        pred = self.mlp_post(x) # [B, 1+group_len, bins]
        pred = pred[:, 1:, :] # [B, group_len, bins]
        return pred

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model
