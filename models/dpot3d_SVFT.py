# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
import math
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from models.SVFTLayer import SVFTLayer
import warnings   


import math
import logging

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


# Convert AFNO3DReplaceAFNO3DSVFT (Fromdpot3d_Lora.pyExtract)
class AFNO3DSVFT(nn.Module):
    def __init__(
        self,
        width=32,
        num_blocks=8,
        channel_first=False,
        sparsity_threshold=0.01,
        modes=32,
        temporal_modes=8,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        act='gelu',
        svft_r=32,
        svft_alpha=1.0,
        off_diag=1,
        pattern='banded'
    ):
        """
        这是带SVFTInject的AFNO3DVersion，ForParameter高效Fine-tune
        """
        super(AFNO3DSVFT, self).__init__()
        assert width % num_blocks == 0
        
        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.act = ACTIVATION[act]
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        # OriginalLearnableParameter
        self.w1 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor)
        )
        self.b1 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor)
        )
        self.w2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size)
        )
        self.b2 = nn.Parameter(
            self.scale * torch.rand(2, self.num_blocks, self.block_size)
        )

        # SVFTWeightModule
        self.svft_w1_real = nn.ModuleList()
        self.svft_w1_imag = nn.ModuleList()
        self.svft_w2_real = nn.ModuleList()
        self.svft_w2_imag = nn.ModuleList()

        for _ in range(num_blocks):
            self.svft_w1_real.append(
                SVFTLayer(
                    in_features=self.block_size,
                    out_features=self.block_size * self.hidden_size_factor,
                    r=svft_r,
                    alpha=svft_alpha,
                    bias=False,
                    off_diag=off_diag,
                    pattern=pattern
                )
            )
            self.svft_w1_imag.append(
                SVFTLayer(
                    in_features=self.block_size,
                    out_features=self.block_size * self.hidden_size_factor,
                    r=svft_r,
                    alpha=svft_alpha,
                    bias=False,
                    off_diag=off_diag,
                    pattern=pattern
                )
            )
            self.svft_w2_real.append(
                SVFTLayer(
                    in_features=self.block_size * self.hidden_size_factor,
                    out_features=self.block_size,
                    r=svft_r,
                    alpha=svft_alpha,
                    bias=False,
                    off_diag=off_diag,
                    pattern=pattern
                )
            )
            self.svft_w2_imag.append(
                SVFTLayer(
                    in_features=self.block_size * self.hidden_size_factor,
                    out_features=self.block_size,
                    r=svft_r,
                    alpha=svft_alpha,
                    bias=False,
                    off_diag=off_diag,
                    pattern=pattern
                )
            )

    def forward(self, x, spatial_size=None):
        B, C, H, W, L = x.shape  # Note: InAFNO3D, InputChannel [B, C, H, W, L]
        
        # CheckInputChannelANDMatch
        if C != self.hidden_size:
            # warnings.warn(f"InputChannel {C} AND {self.hidden_size} Match, Adjust")
            # IfMatch, CreateSize TemporaryTensor
            if C < self.hidden_size:
                temp = torch.zeros(B, self.hidden_size, H, W, L, device=x.device)
                temp[:, :C] = x
                x = temp
            else:
                x = x[:, :self.hidden_size]
            C = self.hidden_size
        
        # SaveInputShapeInformation, ForCheck
        input_shape = (B, C, H, W, L)
        
        # 3D FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        ft_shape = x_ft.shape  # [B, C, H, W, L_ft]
        
        # BlockProcessFrequency domainData
        # ConvertChannelBlock, BlockSize block_size
        x_ft = x_ft.reshape(B, self.num_blocks, self.block_size, *ft_shape[2:])
        
        # Process
        real_part = x_ft.real
        imag_part = x_ft.imag
        
        # Process: SeparateImplementationFrequency domainConvolution
        try:
            # ForBlockProcess
            real_output = torch.zeros_like(real_part)
            imag_output = torch.zeros_like(imag_part)
            
            # ForBlockProcess
            for i in range(self.num_blocks):
                # Flatten
                real_block = real_part[:, i].reshape(B, self.block_size, -1)
                imag_block = imag_part[:, i].reshape(B, self.block_size, -1)
                
                # ApplyWeight
                real_w1 = self.w1[0, i]  # [block_size, block_size*factor]
                real_b1 = self.b1[0, i]  # [block_size*factor]
                
                # ApplyWeight
                imag_w1 = self.w1[1, i]  # [block_size, block_size*factor]
                imag_b1 = self.b1[1, i]  # [block_size*factor]
                
                # Transform
                real_transformed = torch.matmul(real_block, real_w1) + real_b1.unsqueeze(0).unsqueeze(-1)
                imag_transformed = torch.matmul(imag_block, imag_w1) + imag_b1.unsqueeze(0).unsqueeze(-1)
                
                # ApplyActivation function
                real_transformed = self.act(real_transformed)
                imag_transformed = self.act(imag_transformed)
                
                # Transform
                real_w2 = self.w2[0, i]  # [block_size*factor, block_size]
                real_b2 = self.b2[0, i]  # [block_size]
                imag_w2 = self.w2[1, i]  # [block_size*factor, block_size]
                imag_b2 = self.b2[1, i]  # [block_size]
                
                real_final = torch.matmul(real_transformed, real_w2) + real_b2.unsqueeze(0).unsqueeze(-1)
                imag_final = torch.matmul(imag_transformed, imag_w2) + imag_b2.unsqueeze(0).unsqueeze(-1)
                
                # ReshapeOriginalShape
                real_output[:, i] = real_final.reshape(B, self.block_size, *ft_shape[2:])
                imag_output[:, i] = imag_final.reshape(B, self.block_size, *ft_shape[2:])
            
            # 
            x_ft = torch.complex(
                real_output.reshape(B, C, *ft_shape[2:]),
                imag_output.reshape(B, C, *ft_shape[2:])
            )
        except Exception as e:
            # warnings.warn(f"AFNO3DFrequency domainProcessFailure: {e}, UseProcess")
            # : MaintainOriginalFrequency domainData
            x_ft = x_ft.reshape(B, C, *ft_shape[2:])
        
        # FFT
        x = torch.fft.irfftn(x_ft, s=(H, W, L), dim=(-3, -2, -1))
        
        # EnsureOutputShapePositive
        if x.shape != input_shape:
            # warnings.warn(f"IFFTOutputShape {x.shape} ANDInputShape {input_shape} Match, Adjust")
            # CreatePositiveSize TensorCopyData
            output = torch.zeros(input_shape, device=x.device)
            # Copy
            min_b = min(output.shape[0], x.shape[0])
            min_c = min(output.shape[1], x.shape[1])
            min_h = min(output.shape[2], x.shape[2])
            min_w = min(output.shape[3], x.shape[3])
            min_l = min(output.shape[4], x.shape[4])
            output[:min_b, :min_c, :min_h, :min_w, :min_l] = x[:min_b, :min_c, :min_h, :min_w, :min_l]
            x = output
        
        return x





_logger = logging.getLogger(__name__)




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x





class TimeAggregator(nn.Module):
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp'):
        super(TimeAggregator, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) *torch.randn(n_timesteps, out_channels, out_channels),requires_grad=True)   # initialization could be tuned
            self.gamma = nn.Parameter(2**torch.linspace(-10,10, out_channels).unsqueeze(0),requires_grad=True)  # 1, C


    ##  B, X, Y, T, C
    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device) # T, 1
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)

        return x




class PatchEmbedSVFT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, out_dim=128, 
                 act='gelu', use_svft=True, svft_r=32, svft_alpha=1.0, off_diag=2, pattern='banded'):
        super().__init__()
        # EnsureParameterFormatConsistent
        img_size = (img_size, img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        
        # ComputeOutputShape
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (img_size[2] // patch_size[2])
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        
        # SaveDimensionInformation
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_svft = use_svft
        
        # Activation function
        self.act_layer = ACTIVATION[act]
        
        # Convolution layer
        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv2 = nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1)
        
        # StandardSequence
        self.proj = nn.Sequential(
            self.conv1,
            self.act_layer,
            self.conv2
        )
        
        # SVFTLayer
        if use_svft:
            # Convolution layerCreateSVFT
            conv1_in_features = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
            
            self.svft_conv1 = SVFTLayer(
                in_features=conv1_in_features,
                out_features=embed_dim,
                r=svft_r,
                alpha=svft_alpha,
                bias=(self.conv1.bias is not None),
                off_diag=off_diag,
                pattern=pattern
            )
            
            self.svft_conv2 = SVFTLayer(
                in_features=embed_dim,
                out_features=out_dim,
                r=svft_r,
                alpha=svft_alpha,
                bias=(self.conv2.bias is not None),
                off_diag=off_diag,
                pattern=pattern
            )

    def forward(self, x):
        """
        x: [B, C, H, W, L]
            - B: batch size
            - C: Channel数(IncludeOriginalInputChannel + Concatenate的坐标Channel)
            - H, W, L: 3D 尺寸
        """
        B, C, H, W, L = x.shape
        
        # CheckInputRequirement
        if H != self.img_size[0] or W != self.img_size[1] or L != self.img_size[2]:
            # warnings.warn(f"Input ({H}*{W}*{L}) AND ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]}) Match, Adjust")
            # AdjustInput
            x = F.interpolate(x, size=self.img_size, mode='trilinear', align_corners=False)
            H, W, L = self.img_size
        
        # ComputeOutput
        Kx, Ky, Kz = self.patch_size
        out_H, out_W, out_L = H // Kx, W // Ky, L // Kz
        
        # IfUseSVFT, UseConvolution
        if not self.use_svft:
            x = self.proj(x)  # conv1 + activation + conv2
            return x  # RemovePositionEmbedding, ReturnConvolution
        
        try:
            # Convolution: TopatchifyProcess
            # 1. SeparateInput patches
            x_patches = x.unfold(2, Kx, Kx).unfold(3, Ky, Ky).unfold(4, Kz, Kz)
            x_patches = x_patches.contiguous().view(B, C, out_H, out_W, out_L, Kx*Ky*Kz)
            x_patches = x_patches.permute(0, 2, 3, 4, 1, 5).contiguous()
            x_patches = x_patches.view(B, out_H*out_W*out_L, C*Kx*Ky*Kz)
            
            # 2. ConvolutionWeight
            weight_flat = self.conv1.weight.view(self.conv1.out_channels, -1)
            bias = self.conv1.bias if hasattr(self.conv1, 'bias') and self.conv1.bias is not None else None
            
            # 3. ApplySVFT
            svft_out = self.svft_conv1(x_patches, weight_flat, bias)
            
            # 4. ReshapeConvolutionOutputFormat
            svft_x1 = svft_out.view(B, out_H, out_W, out_L, self.conv1.out_channels)
            svft_x1 = svft_x1.permute(0, 4, 1, 2, 3).contiguous()
            
            # ApplyActivation function (RemovePositionEmbedding)
            svft_x1 = self.act_layer(svft_x1)
            
            # 6. Convolution (1x1x1)
            # Flatten [B, embed_dim, -1]
            x2_flat = svft_x1.view(B, self.conv1.out_channels, -1)
            # [B, pixels, channels]
            x2_flat = x2_flat.permute(0, 2, 1).contiguous()
            
            # 7. ConvolutionWeight
            weight2_flat = self.conv2.weight.view(self.conv2.out_channels, -1)
            bias2 = self.conv2.bias if hasattr(self.conv2, 'bias') and self.conv2.bias is not None else None
            
            # 8. ApplySVFT
            svft_out2 = self.svft_conv2(x2_flat, weight2_flat, bias2)
            
            # 9. ReshapeConvolutionOutputFormat
            svft_x2 = svft_out2.view(B, out_H, out_W, out_L, self.conv2.out_channels)
            svft_x2 = svft_x2.permute(0, 4, 1, 2, 3).contiguous()
            
            return svft_x2  # Return [B, out_dim, out_H, out_W, out_L]
            
        except Exception as e:
            # warnings.warn(f"SVFTProcessFailure: {e}, ToStandardConvolution")
            x = self.proj(x)  # UseStandardConvolution
            return x  # RemovePositionEmbedding, ReturnConvolution

    
    def enable_svft(self, enabled=True):
        """EnableORDisableSVFTLayer"""
        self.use_svft = enabled


class BlockSVFT(nn.Module):
    def __init__(self, mixing_type='afno', double_skip=True, width=32, n_blocks=4, mlp_ratio=1., 
                 channel_first=True, modes=32, drop=0., drop_path=0., act='gelu', h=14, w=8, 
                 svft_r=32, svft_alpha=1.0, off_diag=1, pattern='banded'):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO3DSVFT(
                width=width,
                num_blocks=n_blocks,
                sparsity_threshold=0.01,
                channel_first=channel_first,
                modes=modes,
                hard_thresholding_fraction=1,
                hidden_size_factor=1,
                act=act,
                svft_r=svft_r,
                svft_alpha=svft_alpha,
                off_diag=off_diag,
                pattern=pattern
            )

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)

        x = x + residual

        return x

class DPOTNet3D_SVFT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mixing_type='afno', in_channels=1, 
                 out_channels=3, in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768, 
                 out_layer_dim=32, depth=12, modes=32, mlp_ratio=1., n_cls=1, normalize=False, 
                 act='gelu', time_agg='exp_mlp', svft_r=32, svft_alpha=1.0, off_diag=2, pattern='banded'):
        super().__init__()
        
        # SaveParameter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps
        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        
        # Use PatchEmbedSVFT
        self.patch_embed = PatchEmbedSVFT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels + 4,  # 4 Channel
            embed_dim=embed_dim,        
            out_dim=embed_dim,
            act=act,
            use_svft=True,
            svft_r=svft_r,
            svft_alpha=svft_alpha,
            off_diag=off_diag,
            pattern=pattern
        )
        
        # Updatelatent_size
        self.latent_size = self.patch_embed.out_size
        
        # CreatePositiveDimension PositionEmbedding
        self.pos_embed = nn.Parameter(torch.zeros(
            1, embed_dim, 
            self.latent_size[0], 
            self.latent_size[1], 
            self.latent_size[2]
        ))
        
        # InitializePositionEmbedding
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls

        h = img_size // patch_size
        w = h // 2 + 1

        # UseSVFTVersion Block
        self.blocks = nn.ModuleList([
            BlockSVFT(
                mixing_type=mixing_type,
                modes=modes,
                width=embed_dim,
                mlp_ratio=mlp_ratio,
                channel_first=True,
                n_blocks=n_blocks,
                double_skip=False,
                h=h, w=w,
                act=act,
                svft_r=svft_r,
                svft_alpha=svft_alpha,
                off_diag=off_diag,
                pattern=pattern
            )
            for i in range(depth)
        ])

        # Maintain
        if self.normalize:
            self.scale_feats_mu = nn.Linear(2 * in_channels, embed_dim)
            self.scale_feats_sigma = nn.Linear(2 * in_channels, embed_dim)
            
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, embed_dim),
            self.act,
            nn.Linear(embed_dim, n_cls)
        )

        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)

        self.out_layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels=embed_dim, out_channels=out_layer_dim, 
                             kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=out_layer_dim, 
                     kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, 
                     out_channels=self.out_channels * self.out_timesteps,
                     kernel_size=1, stride=1)
        )

        self.mixing_type = mixing_type

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, x):
        batchsize, size_x, size_y = x.shape[0], x.shape[1], x.shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(x.device)
        return grid

    def get_grid_4d(self, x):
        batchsize, size_x, size_y, size_z, size_t = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).to(x.device).repeat([batchsize, 1, size_y, size_z, size_t, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).to(x.device).repeat([batchsize, size_x, 1, size_z, size_t, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).to(x.device).repeat([batchsize, size_x, size_y, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, 1, size_t, 1).to(x.device).repeat([batchsize, size_x, size_y, size_z, 1, 1])
        grid = torch.cat((gridx, gridy, gridz, gridt), dim=-1)
        return grid

    def forward(self, x):
        B, _, _, _, T, _ = x.shape
        if self.normalize:
            mu, sigma = x.mean(dim=(1,2,3,4), keepdim=True), x.std(dim=(1,2,3,4), keepdim=True) + 1e-6
            x = (x - mu) / sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 4, 1, 2, 3)
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 4, 1, 2, 3)

        # ProcessInput
        grid = self.get_grid_4d(x)
        x = torch.cat((x, grid), dim=-1).contiguous()
        x = rearrange(x, 'b x y z t c -> (b t) c x y z')
        
        # Applypatch embedding - ReturnPositionEmbedding
        x = self.patch_embed(x)  # Output [B*T, embed_dim, H', W', D']
        
        # InAddPositionEmbedding
        if x.shape[2:] != self.pos_embed.shape[2:] or x.shape[1] != self.pos_embed.shape[1]:
            # AdjustPositionEmbeddingToMatchFeatureDimension
            pos_embed_resized = F.interpolate(
                self.pos_embed, 
                size=x.shape[2:], 
                mode='trilinear', 
                align_corners=False
            )
            x = x + pos_embed_resized
        else:
            x = x + self.pos_embed
        
        # ContinueProcess
        x = rearrange(x, '(b t) c x y z -> b x y z t c', b=B, t=T)
        x = self.time_agg_layer(x)
        x = rearrange(x, 'b x y z c -> b c x y z')
        
        if self.normalize:
            x = scale_sigma * x + scale_mu

        for blk in self.blocks:
            x = blk(x)

        x = self.out_layer(x).permute(0, 2, 3, 4, 1)
        x = x.reshape(*x.shape[:4], self.out_timesteps, self.out_channels).contiguous()

        if self.normalize:
            x = x * sigma + mu

        return x

    def extra_repr(self) -> str:
        named_modules = set()
        for p in self.named_modules():
            named_modules.update([p[0]])
        named_modules = list(named_modules)

        string_repr = ''
        for p in self.named_parameters():
            name = p[0].split('.')[0]
            if name not in named_modules:
                string_repr = string_repr + '(' + name + '): ' \
                              + 'tensor(' + str(tuple(p[1].shape)) + ', requires_grad=' + str(
                    p[1].requires_grad) + ')\n'

        return string_repr



def resize_pos_embed(posemb, posemb_new):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if True:
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    gs_new = int(math.sqrt(ntok_new))
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


if __name__ == "__main__":
    x = torch.rand(2, 64, 64, 64, 10, 3)

    from utils.utilities import load_3d_components_from_2d
    load_path = "/root/autodl-tmp/checkpoints/model_S.pth"
    
    # UseSVFT
    net = DPOTNet3D_SVFT(
        img_size=64, 
        patch_size=8, 
        in_channels=3, 
        out_channels=3, 
        in_timesteps=10, 
        embed_dim=1024,
        n_blocks=8, 
        depth=6,
        svft_r=16,        # SVFTRank
        svft_alpha=32.0,  # SVFTScaling factor
        off_diag=2,       # For
        pattern='banded'  # SparseMode
    )
    
    import argparse
    import torch.serialization

    state_dict = torch.load(load_path, map_location='cpu')['model']
    load_3d_components_from_2d(net, state_dict, ['blocks','time_agg'])

    y = net(x)
    print(y.shape)