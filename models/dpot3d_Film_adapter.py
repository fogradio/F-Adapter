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

# Define Adapter base module
class Adapter(nn.Module):
    """FiLM-style Adapter using channel scaling and shift"""
    def __init__(self, in_features, bottleneck_dim, activation='gelu', scale=1.0, init_option='zero'):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        self.scale = scale
        
        # Down projection: reduce dimension
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        
        # Activation function
        self.act = ACTIVATION[activation]
        
        # FiLM output layer: generate gamma and beta (scale and shift)
        self.film_fc = nn.Linear(bottleneck_dim, in_features * 2)
        
        # Initialization strategy
        self._init_weights(init_option)
    
    def _init_weights(self, init_option):
        """Initialize weights"""
        if init_option == 'zero':
            # Initialize FiLM output to zero so early training does not affect the base model
            nn.init.zeros_(self.film_fc.weight)
            # gamma initialized to 0, beta to 0
            nn.init.zeros_(self.film_fc.bias)
        elif init_option == 'normal':
            # Use normal distribution initialization
            nn.init.normal_(self.down_proj.weight, std=0.01)
            nn.init.normal_(self.film_fc.weight, std=0.01)
        # Down projection uses standard initialization
        if init_option != 'normal':
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            if self.down_proj.bias is not None:
                nn.init.zeros_(self.down_proj.bias)
    
    def forward(self, x):
        """Forward: input -> down projection -> activation -> gamma/beta -> FiLM modulation -> return"""
        # Save input for FiLM modulation
        identity = x
        
        # Down projection
        x = self.down_proj(x)
        
        # Activation
        x = self.act(x)
        
        # Generate gamma and beta (scale and shift)
        film_params = self.film_fc(x)
        gamma, beta = film_params.chunk(2, dim=-1)
        
        # Scale gamma
        gamma = gamma * self.scale
        beta = beta * self.scale
        
        # FiLM modulation: x = identity * (1 + gamma) + beta
        x = identity * (1 + gamma) + beta
        
        return x

# Frequency-domain Adapter for AFNO3D
class AFNO3DAdapter(nn.Module):
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
        adapter_dim=32,  # Adapter bottleneck dimension
        adapter_scale=1.0
    ):
        super(AFNO3DAdapter, self).__init__()
        assert width % num_blocks == 0, (
            f"hidden_size {width} should be divisible by num_blocks {num_blocks}"
        )

        self.hidden_size = width
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.channel_first = channel_first
        self.modes = modes
        self.temporal_modes = temporal_modes
        self.hidden_size_factor = hidden_size_factor
        self.act = ACTIVATION[act]
        # Original scale
        self.scale = 1 / (self.block_size * self.block_size * self.hidden_size_factor)

        # Original learnable parameters
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

        # Add Adapter modules
        # Add input/output adapters for each block
        self.adapters_in = nn.ModuleList([
            Adapter(self.block_size, adapter_dim, activation=act, scale=adapter_scale)
            for _ in range(num_blocks)
        ])
        
        self.adapters_mid = nn.ModuleList([
            Adapter(self.block_size * self.hidden_size_factor, adapter_dim, activation=act, scale=adapter_scale)
            for _ in range(num_blocks)
        ])
        
        self.adapters_out = nn.ModuleList([
            Adapter(self.block_size, adapter_dim, activation=act, scale=adapter_scale)
            for _ in range(num_blocks)
        ])

    def forward(self, x, spatial_size=None):
        if self.channel_first:
            x = rearrange(x, 'b c x y z -> b x y z c')
        B, H, W, L, C = x.shape
        x_orig = x

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        dtype_float = x.real.dtype
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        # Create result container
        o1_real = torch.zeros(
            [B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device, dtype=dtype_float
        )
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(x.shape, device=x.device, dtype=dtype_float)
        o2_imag = torch.zeros_like(o2_real)

        kept_modes = self.modes

        # Process each block and apply adapter
        for block_idx in range(self.num_blocks):
            # Get data for current block
            x_sub_real = x[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx].real
            x_sub_imag = x[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx].imag
            
            # Flatten data to apply adapter
            flat_shape = x_sub_real.shape
            flat_x_real = x_sub_real.reshape(-1, self.block_size)
            flat_x_imag = x_sub_imag.reshape(-1, self.block_size)
            
            # Apply input adapter
            adapted_x_real = self.adapters_in[block_idx](flat_x_real).reshape(flat_shape)
            adapted_x_imag = self.adapters_in[block_idx](flat_x_imag).reshape(flat_shape)
            
            # Compute original (o1 real part)
            o1_real_block = (
                torch.einsum('...i,io->...o', adapted_x_real, self.w1[0, block_idx]) - 
                torch.einsum('...i,io->...o', adapted_x_imag, self.w1[1, block_idx]) + 
                self.b1[0, block_idx]
            )
            
            # Compute original (o1 imaginary part)
            o1_imag_block = (
                torch.einsum('...i,io->...o', adapted_x_imag, self.w1[0, block_idx]) + 
                torch.einsum('...i,io->...o', adapted_x_real, self.w1[1, block_idx]) + 
                self.b1[1, block_idx]
            )
            
            # Apply GELU
            o1_real_block = F.gelu(o1_real_block)
            o1_imag_block = F.gelu(o1_imag_block)
            
            # Apply middle adapter
            flat_o1_real = o1_real_block.reshape(-1, self.block_size * self.hidden_size_factor)
            flat_o1_imag = o1_imag_block.reshape(-1, self.block_size * self.hidden_size_factor)
            
            adapted_o1_real = self.adapters_mid[block_idx](flat_o1_real).reshape(o1_real_block.shape)
            adapted_o1_imag = self.adapters_mid[block_idx](flat_o1_imag).reshape(o1_imag_block.shape)
            
            # Compute o2 real and imaginary parts
            o2_real_block = (
                torch.einsum('...i,io->...o', adapted_o1_real, self.w2[0, block_idx]) - 
                torch.einsum('...i,io->...o', adapted_o1_imag, self.w2[1, block_idx]) + 
                self.b2[0, block_idx]
            )
            
            o2_imag_block = (
                torch.einsum('...i,io->...o', adapted_o1_imag, self.w2[0, block_idx]) + 
                torch.einsum('...i,io->...o', adapted_o1_real, self.w2[1, block_idx]) + 
                self.b2[1, block_idx]
            )
            
            # Apply output adapter
            flat_o2_real = o2_real_block.reshape(-1, self.block_size)
            flat_o2_imag = o2_imag_block.reshape(-1, self.block_size)
            
            adapted_o2_real = self.adapters_out[block_idx](flat_o2_real).reshape(o2_real_block.shape)
            adapted_o2_imag = self.adapters_out[block_idx](flat_o2_imag).reshape(o2_imag_block.shape)
            
            # Store results into output container
            o2_real[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx] = adapted_o2_real
            o2_imag[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx] = adapted_o2_imag

        # Combine real/imag back and IFFT
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(H, W, L), dim=(1, 2, 3), norm="ortho")

        # Skip connection
        x = x + x_orig
        
        if self.channel_first:
            x = rearrange(x, 'b x y z c -> b c x y z')
            
        return x

# Adapter version of PatchEmbed
class PatchEmbedAdapter(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu',
                 adapter_dim=32, adapter_scale=1.0):
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
        
        # Original projection layer
        self.conv1 = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv2 = nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1)
        
        # Standard sequence when no adapter
        self.proj = nn.Sequential(
            self.conv1,
            self.act,
            self.conv2
        )
        
        # Add channel adapters (for feature channels)
        self.channel_adapter1 = nn.Sequential(
            nn.Conv3d(embed_dim, adapter_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(adapter_dim, embed_dim, kernel_size=1, stride=1)
        )
        
        self.channel_adapter2 = nn.Sequential(
            nn.Conv3d(out_dim, adapter_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(adapter_dim, out_dim, kernel_size=1, stride=1)
        )
        
        # Adapter scale factor
        self.adapter_scale = adapter_scale
        
        # Initialization
        self._init_adapter_weights()
    
    def _init_adapter_weights(self):
        """Init adapter weights near zero to limit early impact"""
        for m in self.channel_adapter1.modules():
            if isinstance(m, nn.Conv3d):
                if m == self.channel_adapter1[-1]:  # Up projection init to zero
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:  # Down projection uses standard initialization
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        for m in self.channel_adapter2.modules():
            if isinstance(m, nn.Conv3d):
                if m == self.channel_adapter2[-1]:  # Up projection init to zero
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:  # Down projection uses standard initialization
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2], \
            f"Input image size ({H}*{W}*{L}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        # First convolution
        x1 = self.conv1(x)
        
        # Apply first adapter
        adapter_out1 = self.channel_adapter1(x1) * self.adapter_scale
        x1 = x1 + adapter_out1
        
        # Activation function
        x1 = self.act(x1)
        
        # Second convolution
        x2 = self.conv2(x1)
        
        # Apply second adapter
        adapter_out2 = self.channel_adapter2(x2) * self.adapter_scale
        x = x2 + adapter_out2
        
        return x

# Modify Block to use AFNO3DAdapter
class BlockAdapter(nn.Module):
    def __init__(self, mixing_type='afno', double_skip=True, width=32, n_blocks=4, mlp_ratio=1., 
                 channel_first=True, modes=32, drop=0., drop_path=0., act='gelu', h=14, w=8,
                 adapter_dim=32, adapter_scale=1.0):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO3DAdapter(
                width=width, 
                num_blocks=n_blocks, 
                sparsity_threshold=0.01, 
                channel_first=channel_first, 
                modes=modes,
                hard_thresholding_fraction=1, 
                hidden_size_factor=1, 
                act=act,
                adapter_dim=adapter_dim,
                adapter_scale=adapter_scale
            )

        self.norm2 = torch.nn.GroupNorm(8, width)

        mlp_hidden_dim = int(width * mlp_ratio)
        
        # Original MLP layer
        self.mlp_orig = nn.Sequential(
            nn.Conv3d(in_channels=width, out_channels=mlp_hidden_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=mlp_hidden_dim, out_channels=width, kernel_size=1, stride=1),
        )
        
        # Adapter for MLP layer
        self.mlp_adapter = nn.Sequential(
            nn.Conv3d(in_channels=width, out_channels=adapter_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=adapter_dim, out_channels=width, kernel_size=1, stride=1),
        )
        
        # Adapter scale factor
        self.adapter_scale = adapter_scale
        
        # Init adapter near zero
        self._init_adapter_weights()
        
        self.double_skip = double_skip

    def _init_adapter_weights(self):
        """Initialize adapter weights"""
        for m in self.mlp_adapter.modules():
            if isinstance(m, nn.Conv3d):
                if m == self.mlp_adapter[-1]:  # Up projection init to zero
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:  # Down projection uses standard initialization
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        
        # Apply original MLP
        mlp_out = self.mlp_orig(x)
        
        # Apply MLP adapter
        adapter_out = self.mlp_adapter(x) * self.adapter_scale
        
        # Add residual plus adapter output
        x = mlp_out + adapter_out + residual

        return x

# Full DPOTNet3D model with all adapter components
class DPOTNet3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mixing_type='afno', in_channels=1, out_channels=3, 
                 in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768, out_layer_dim=32, depth=12, 
                 modes=32, mlp_ratio=1., n_cls=1, normalize=False, act='gelu', time_agg='exp_mlp',
                 adapter_dim=32, adapter_scale=1.0):
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
        
        # Use PatchEmbed with adapter
        self.patch_embed = PatchEmbedAdapter(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_channels + 4, 
            embed_dim=out_channels * patch_size + 4, 
            out_dim=embed_dim,
            act=act,
            adapter_dim=adapter_dim,
            adapter_scale=adapter_scale
        )

        self.latent_size = self.patch_embed.out_size

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1], self.patch_embed.out_size[2]))
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls

        h = img_size // patch_size
        w = h // 2 + 1

        # Use Block with adapter
        self.blocks = nn.ModuleList([
            BlockAdapter(
                mixing_type=mixing_type, 
                modes=modes,
                width=embed_dim, 
                mlp_ratio=mlp_ratio, 
                channel_first=True, 
                n_blocks=n_blocks, 
                double_skip=False, 
                h=h, 
                w=w, 
                act=act,
                adapter_dim=adapter_dim,
                adapter_scale=adapter_scale
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

        self.time_agg_layer = TimeAggregator(in_channels, in_timesteps, embed_dim, time_agg)

        # Output layer
        self.out_layer_orig = nn.Sequential(
            nn.ConvTranspose3d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps, kernel_size=1, stride=1)
        )
        
        # Adapter for output layer
        self.out_layer_adapter = nn.Sequential(
            nn.ConvTranspose3d(in_channels=embed_dim, out_channels=adapter_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(in_channels=adapter_dim, out_channels=self.out_channels * self.out_timesteps, kernel_size=1, stride=1)
        )
        
        # Adapter scale factor
        self.adapter_scale = adapter_scale
        
        # Initialize output-layer adapter near zero
        self._init_adapter_weights()

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)

        self.mixing_type = mixing_type
    
    def _init_adapter_weights(self):
        """Initialize output-layer adapter weights"""
        for m in self.out_layer_adapter.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if m == self.out_layer_adapter[-1]:  # Up projection init to zero
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)
                else:  # Down projection uses standard initialization
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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

        # Original output layer
        out_orig = self.out_layer_orig(x)
        
        # AdapterOutput layer
        out_adapter = self.out_layer_adapter(x) * self.adapter_scale
        
        # Merge outputs
        x = out_orig + out_adapter
        
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

# Keep original TimeAggregator and helper functions
# (Remaining code should match the original file)

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

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu'):
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * ( img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.out_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.out_dim = out_dim
        self.act = ACTIVATION[act]

        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2], f"Input image size ({H}*{W}*{L}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x)
        return x

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
    # x = torch.rand(4, 20, 20, 100)
    # net = AFNO2D(in_timesteps=3, out_timesteps=1, n_channels=2, width=100, num_blocks=5)
    x = torch.rand(2, 64, 64, 64, 10, 3)

    from utils.utilities import load_3d_components_from_2d
    load_path = "/root/autodl-tmp/checkpoints/model_S.pth"
    net = DPOTNet3D(img_size=64, patch_size=8, in_channels=3, out_channels=3, in_timesteps=10, embed_dim=1024,n_blocks=8, depth=6)
    import argparse
    import torch.serialization

    #torch.serialization.add_safe_globals([argparse.Namespace])
    state_dict = torch.load(load_path, map_location='cpu')['model']

    load_3d_components_from_2d(net, state_dict, ['blocks','time_agg'])

    y = net(x)
    print(y.shape)
