# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
import math
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F



import math
import logging

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}


class AFNO3DPrompt(nn.Module):
    """AFNO3D layer with prompt vectors"""
    def __init__(self, width=32, num_blocks=8, channel_first=False, sparsity_threshold=0.01, 
                 modes=32, temporal_modes=8, hard_thresholding_fraction=1, hidden_size_factor=1, 
                 act='gelu', prompt_dim=32, prompt_pos='both'):
        super(AFNO3DPrompt, self).__init__()
        assert width % num_blocks == 0, f"hidden_size {width} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.act = ACTIVATION[act]
        self.prompt_pos = prompt_pos  # 'pre', 'post', 'both'

        # Base parameters
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)
        self.w1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.rand(2, self.num_blocks, self.block_size))

        # Add trainable prompt vectors
        # Prompt before FFT
        if self.prompt_pos in ['pre', 'both']:
            self.pre_prompt = nn.Parameter(torch.zeros(1, prompt_dim, self.hidden_size, 1, 1, 1))
            nn.init.normal_(self.pre_prompt, std=0.02)
        
        # Prompt after FFT
        if self.prompt_pos in ['post', 'both']:
            self.post_prompt = nn.Parameter(torch.zeros(1, prompt_dim, self.hidden_size, 1, 1, 1))
            nn.init.normal_(self.post_prompt, std=0.02)

    def forward(self, x, spatial_size=None):
        if self.channel_first:
            x = rearrange(x, 'b c x y z -> b x y z c')
        
        B, H, W, L, C = x.shape
        
        # Add prompt before FFT
        if self.prompt_pos in ['pre', 'both']:
            prompt = self.pre_prompt.expand(B, -1, -1, H, W, L)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            prompt = rearrange(prompt, 'b c x y z -> b x y z c')
            x = x + prompt
        
        x_orig = x
        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        kept_modes = self.modes

        o1_real[:, :kept_modes, :kept_modes, :self.temporal_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes, :self.temporal_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes, :self.temporal_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :kept_modes, :kept_modes, :self.temporal_modes] = F.gelu(
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes, :self.temporal_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :kept_modes, :kept_modes, :self.temporal_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :kept_modes, :kept_modes, :self.temporal_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes, :self.temporal_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes, :self.temporal_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :kept_modes, :kept_modes, :self.temporal_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :kept_modes, :kept_modes, :self.temporal_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :kept_modes, :kept_modes, :self.temporal_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(H, W, L), dim=(1, 2, 3), norm="ortho")

        x = x + x_orig
        
        # Add prompt after FFT
        if self.prompt_pos in ['post', 'both']:
            prompt = self.post_prompt.expand(B, -1, -1, H, W, L)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            prompt = rearrange(prompt, 'b c x y z -> b x y z c')
            x = x + prompt
        
        if self.channel_first:
            x = rearrange(x, 'b x y z c -> b c x y z')
        
        return x


class PatchEmbedPrompt(nn.Module):
    """PatchEmbed layer with prompt vectors"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, 
                 act='gelu', prompt_dim=32, prompt_pos='both'):
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]
        self.prompt_pos = prompt_pos

        # Base projection layer
        self.conv1 = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv2 = nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1)
        
        self.proj = nn.Sequential(
            self.conv1,
            self.act,
            self.conv2
        )
        
        # Add trainable prompt vectors
        if self.prompt_pos in ['pre', 'both']:
            self.pre_prompt = nn.Parameter(torch.zeros(1, prompt_dim, in_chans, 1, 1, 1))
            nn.init.normal_(self.pre_prompt, std=0.02)
        
        if self.prompt_pos in ['mid', 'both']:
            self.mid_prompt = nn.Parameter(torch.zeros(1, prompt_dim, embed_dim, 1, 1, 1))
            nn.init.normal_(self.mid_prompt, std=0.02)
        
        if self.prompt_pos in ['post', 'both']:
            self.post_prompt = nn.Parameter(torch.zeros(1, prompt_dim, out_dim, 1, 1, 1))
            nn.init.normal_(self.post_prompt, std=0.02)

    def forward(self, x):
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2], \
            f"Input image size ({H}*{W}*{L}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        # Add prompt to input
        if self.prompt_pos in ['pre', 'both']:
            prompt = self.pre_prompt.expand(B, -1, C, H, W, L)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            x = x + prompt
        
        # First convolution
        x = self.conv1(x)
        
        # Add prompt in the middle
        if self.prompt_pos in ['mid', 'both']:
            out_h, out_w, out_l = H // self.patch_size[0], W // self.patch_size[1], L // self.patch_size[2]
            embed_dim = x.shape[1]  # get actual embed_dim
            prompt = self.mid_prompt.expand(B, -1, embed_dim, out_h, out_w, out_l)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            x = x + prompt
        
        x = self.act(x)
        x = self.conv2(x)
        
        # Add prompt at output
        if self.prompt_pos in ['post', 'both']:
            out_h, out_w, out_l = H // self.patch_size[0], W // self.patch_size[1], L // self.patch_size[2]
            out_dim = x.shape[1]  # get actual out_dim
            prompt = self.post_prompt.expand(B, -1, out_dim, out_h, out_w, out_l)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            x = x + prompt
        
        return x


class BlockPrompt(nn.Module):
    """Block layer with prompt vectors"""
    def __init__(self, mixing_type='afno', double_skip=True, width=32, n_blocks=4, mlp_ratio=1.,
                 channel_first=True, modes=32, drop=0., drop_path=0., act='gelu', h=14, w=8,
                 prompt_dim=32, prompt_pos='both'):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]
        self.prompt_pos = prompt_pos

        if mixing_type == "afno":
            self.filter = AFNO3DPrompt(
                width=width, num_blocks=n_blocks, sparsity_threshold=0.01, 
                channel_first=channel_first, modes=modes, prompt_pos=prompt_pos
            )

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )

        # Add trainable prompt vectors
        if self.prompt_pos in ['pre', 'both']:
            self.pre_prompt = nn.Parameter(torch.zeros(1, prompt_dim, width, 1, 1, 1))
            nn.init.normal_(self.pre_prompt, std=0.02)
        
        if self.prompt_pos in ['post', 'both']:
            self.post_prompt = nn.Parameter(torch.zeros(1, prompt_dim, width, 1, 1, 1))
            nn.init.normal_(self.post_prompt, std=0.02)

        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        
        # Add prompt before processing
        if self.prompt_pos in ['pre', 'both']:
            B, C, H, W, L = x.shape
            prompt = self.pre_prompt.expand(B, -1, C, H, W, L)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            x = x + prompt
        
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        
        # Add prompt after processing
        if self.prompt_pos in ['post', 'both']:
            B, C, H, W, L = x.shape
            prompt = self.post_prompt.expand(B, -1, C, H, W, L)
            prompt = prompt.sum(dim=1)  # merge prompt_dim dimension
            x = x + prompt

        x = x + residual

        return x


class TimeAggregatorPrompt(nn.Module):
    """TimeAggregator with prompt vectors"""
    def __init__(self, n_channels, n_timesteps, out_channels, type='mlp', prompt_dim=32):
        super(TimeAggregatorPrompt, self).__init__()
        self.n_channels = n_channels
        self.n_timesteps = n_timesteps
        self.out_channels = out_channels
        self.type = type
        
        if self.type == 'mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) * torch.randn(n_timesteps, out_channels, out_channels), requires_grad=True)
        elif self.type == 'exp_mlp':
            self.w = nn.Parameter(1/(n_timesteps * out_channels**0.5) * torch.randn(n_timesteps, out_channels, out_channels), requires_grad=True)
            self.gamma = nn.Parameter(2**torch.linspace(-10, 10, out_channels).unsqueeze(0), requires_grad=True)
        
        # Add trainable prompt vectors
        self.prompt = nn.Parameter(torch.zeros(1, 1, 1, 1, out_channels))
        nn.init.normal_(self.prompt, std=0.02)

    def forward(self, x):
        if self.type == 'mlp':
            x = torch.einsum('tij, ...ti->...j', self.w, x)
        elif self.type == 'exp_mlp':
            t = torch.linspace(0, 1, x.shape[-2]).unsqueeze(-1).to(x.device)
            t_embed = torch.cos(t @ self.gamma)
            x = torch.einsum('tij,...ti->...j', self.w, x * t_embed)
        
        # Add prompt vectors
        x = x + self.prompt
        
        return x


class DPOTNet3D(nn.Module):
    """DPOTNet3D with Prompt Tuning"""
    def __init__(self, img_size=224, patch_size=16, mixing_type='afno', in_channels=1, out_channels=3,
                 in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768, out_layer_dim=32, depth=12,
                 modes=32, mlp_ratio=1., n_cls=1, normalize=False, act='gelu', time_agg='exp_mlp',
                 prompt_dim=32, prompt_pos='both'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_timesteps = in_timesteps
        self.out_timesteps = out_timesteps

        self.n_blocks = n_blocks
        self.modes = modes
        self.num_features = self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.act = ACTIVATION[act]
        self.prompt_pos = prompt_pos
        
        # Use PatchEmbed with prompts
        self.patch_embed = PatchEmbedPrompt(
            img_size=img_size, patch_size=patch_size, 
            in_chans=in_channels + 4, embed_dim=out_channels * patch_size + 4, 
            out_dim=embed_dim, act=act, prompt_dim=prompt_dim, prompt_pos=prompt_pos
        )

        self.latent_size = self.patch_embed.out_size

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1], self.patch_embed.out_size[2]))
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls

        h = img_size // patch_size
        w = h // 2 + 1

        # Use Blocks with prompts
        self.blocks = nn.ModuleList([
            BlockPrompt(
                mixing_type=mixing_type, modes=modes,
                width=embed_dim, mlp_ratio=mlp_ratio, 
                channel_first=True, n_blocks=n_blocks,
                double_skip=False, h=h, w=w, act=act,
                prompt_dim=prompt_dim, prompt_pos=prompt_pos
            ) for i in range(depth)
        ])

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

        # Use TimeAggregator with prompts
        self.time_agg_layer = TimeAggregatorPrompt(in_channels, in_timesteps, embed_dim, time_agg, prompt_dim=prompt_dim)

        # Output layer
        self.out_layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps, kernel_size=1, stride=1)
        )
        
        # Prompt vectors for output layer
        self.output_prompt = nn.Parameter(torch.zeros(1, self.out_channels * self.out_timesteps, 1, 1, 1))
        nn.init.normal_(self.output_prompt, std=0.02)

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

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
            mu, sigma = x.mean(dim=(1,2,3,4),keepdim=True), x.std(dim=(1,2,3,4),keepdim=True) + 1e-6
            x = (x - mu) / sigma
            scale_mu = self.scale_feats_mu(torch.cat([mu, sigma],dim=-1)).squeeze(-2).permute(0,4,1,2,3)
            scale_sigma = self.scale_feats_sigma(torch.cat([mu, sigma], dim=-1)).squeeze(-2).permute(0, 4, 1, 2, 3)

        grid = self.get_grid_4d(x)
        x = torch.cat((x, grid), dim=-1).contiguous()
        x = rearrange(x, 'b x y z t c -> (b t) c x y z')
        x = self.patch_embed(x)

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y z -> b x y z t c', b=B, t=T)

        x = self.time_agg_layer(x)

        x = rearrange(x, 'b x y z c -> b c x y z')

        if self.normalize:
            x = scale_sigma * x + scale_mu

        for blk in self.blocks:
            x = blk(x)

        x = self.out_layer(x)
        
        # Add prompt at output
        B, C, H, W, L = x.shape
        output_prompt = self.output_prompt.expand(B, -1, H, W, L)
        x = x + output_prompt
        
        x = x.permute(0, 2, 3, 4, 1)
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





if __name__ == "__main__":
    # x = torch.rand(4, 20, 20, 100)
    # net = AFNO2D(in_timesteps=3, out_timesteps=1, n_channels=2, width=100, num_blocks=5)
    x = torch.rand(2, 64, 64, 64, 10, 3)

    from utils.utilities import load_3d_components_from_2d
    load_path = "/NEW_EDS/JJ_Group/zhw/DPOT/model_S.pth"
    net = DPOTNet3D(img_size=64, patch_size=8, in_channels=3, out_channels=3, in_timesteps=10, embed_dim=1024,n_blocks=8, depth=6)
    import argparse
    import torch.serialization

    torch.serialization.add_safe_globals([argparse.Namespace])
    state_dict = torch.load(load_path, map_location='cpu')['model']

    load_3d_components_from_2d(net, state_dict, ['blocks','time_agg'])

    y = net(x)
    print(y.shape)
