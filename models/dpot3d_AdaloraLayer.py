# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
import math
import numpy as np
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import logging

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.AdaloraLayer import AdaLoraLayer

_logger = logging.getLogger(__name__)

ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU(),'silu':nn.SiLU()}

class RankAllocator:
    """
    Importance scoring and rank allocator for dynamically adjusting AdaLora ranks
    """
    def __init__(self, model, target_r=8, init_r=12, total_step=1000,
                 beta1=0.85, beta2=0.85, tinit=200, tfinal=800, deltaT=10):
        self.model = model
        self.target_r = target_r  # Target average rank
        self.init_r = init_r      # Initial rank
        self.total_step = total_step  # Total training steps
        self.beta1 = beta1  # Importance smoothing coefficient
        self.beta2 = beta2  # Uncertainty smoothing coefficient
        self.tinit = tinit  # Initial warmup steps
        self.tfinal = tfinal  # Final fine-tuning steps
        self.deltaT = deltaT  # Rank allocation time interval

        self.ipt = {}  # Importance scores
        self.exp_avg_ipt = {}  # Smoothed importance
        self.exp_avg_unc = {}  # Uncertainty assessment

        # Get all AdaLora layers
        self.adalora_layers = {}
        for name, module in model.named_modules():
            # Collect all AdaLoraLayers in AFNO3DAdaLora modules
            if isinstance(module, AFNO3DAdaLora):
                for i, layer in enumerate(module.lora_w1_real):
                    self.adalora_layers[f"{name}.lora_w1_real.{i}"] = layer
                for i, layer in enumerate(module.lora_w1_imag):
                    self.adalora_layers[f"{name}.lora_w1_imag.{i}"] = layer
                for i, layer in enumerate(module.lora_w2_real):
                    self.adalora_layers[f"{name}.lora_w2_real.{i}"] = layer
                for i, layer in enumerate(module.lora_w2_imag):
                    self.adalora_layers[f"{name}.lora_w2_imag.{i}"] = layer

            # Collect AdaLoraLayers in PatchEmbedAdaLora
            elif isinstance(module, PatchEmbedAdaLora):
                if hasattr(module, 'lora_down_conv1'):
                    self.adalora_layers[f"{name}.lora_down_conv1"] = module
                if hasattr(module, 'lora_down_conv2'):
                    self.adalora_layers[f"{name}.lora_down_conv2"] = module

        # Calculate initial total budget and target total budget
        self.init_budget = 0
        for name, layer in self.adalora_layers.items():
            if hasattr(layer, 'r'):
                self.init_budget += layer.r
            # For PatchEmbedAdaLora, may need separate handling

        self.target_budget = self.target_r * len(self.adalora_layers)

    def update_importance_score(self):
        """Update importance scores and uncertainty for each parameter"""
        for name, param in self.model.named_parameters():
            if 'lora_A' in name or 'lora_E' in name or 'lora_B' in name:
                if param.grad is not None:
                    if name not in self.ipt:
                        self.ipt[name] = torch.zeros_like(param)
                        self.exp_avg_ipt[name] = torch.zeros_like(param)
                        self.exp_avg_unc[name] = torch.zeros_like(param)
                    
                    with torch.no_grad():
                        # Compute current importance = |param * grad|
                        self.ipt[name] = (param * param.grad).abs().detach()
                        # Smooth importance estimates
                        self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + (1 - self.beta1) * self.ipt[name]
                        # Uncertainty estimation
                        self.exp_avg_unc[name] = self.beta2 * self.exp_avg_unc[name] + (1 - self.beta2) * (self.ipt[name] - self.exp_avg_ipt[name]).abs()
    
    def budget_schedule(self, step):
        """Compute budget for current step"""
        if step <= self.tinit:
            # Keep initial budget during warmup
            return self.init_budget, False
        elif step > self.total_step - self.tfinal:
            # Use target budget in final stage
            return self.target_budget, True
        else:
            # Linearly reduce budget in the middle stage
            ratio = 1 - (step - self.tinit) / (self.total_step - self.tfinal - self.tinit)
            budget = int((self.init_budget - self.target_budget) * (ratio**3) + self.target_budget)
            mask_ind = (step % self.deltaT == 0)
            return budget, mask_ind
    
    def _calculate_scores(self):
        """Compute combined score for each (A,E,B) tuple"""
        scores_dict = {}
        all_scores = []
        
        # Compute tuple score per layer
        for layer_name, layer in self.adalora_layers.items():
            if hasattr(layer, 'lora_A'):
                # Standard AdaLoraLayer
                a_score = self.exp_avg_ipt.get(f"{layer_name}.lora_A", None)
                e_score = self.exp_avg_ipt.get(f"{layer_name}.lora_E", None)
                b_score = self.exp_avg_ipt.get(f"{layer_name}.lora_B", None)
                
                if a_score is not None and e_score is not None and b_score is not None:
                    # Compute combined score
                    a_score = a_score * self.exp_avg_unc[f"{layer_name}.lora_A"]
                    e_score = e_score * self.exp_avg_unc[f"{layer_name}.lora_E"]
                    b_score = b_score * self.exp_avg_unc[f"{layer_name}.lora_B"]
                    
                    # Compute score per rank dimension
                    a_score = torch.mean(a_score, dim=1, keepdim=True)
                    b_score = torch.mean(b_score, dim=0, keepdim=False).view(-1, 1)
                    
                    # Combine E score with A and B
                    combined_score = e_score.view(-1) + a_score.view(-1) + b_score.view(-1)
                    scores_dict[layer_name] = combined_score
                    all_scores.append(combined_score)
            
            # PatchEmbed AdaLora may need special handling
        
        return scores_dict, torch.cat(all_scores) if all_scores else None
    
    def allocate_budget(self, budget):
        """Allocate budget to layers based on importance scores"""
        scores_dict, all_scores = self._calculate_scores()
        if all_scores is None:
            return {}
        
        # Find threshold: keep top-budget most important parameters
        total_params = all_scores.size(0)
        if total_params <= budget:
            # Budget sufficient; no pruning needed
            return {name: torch.ones_like(score, dtype=torch.bool) for name, score in scores_dict.items()}
        
        # Use the (total_params - budget)-th smallest value as threshold
        threshold = torch.kthvalue(all_scores, k=total_params-budget)[0].item()
        
        # Create mask for each layer
        rank_pattern = {}
        for name, score in scores_dict.items():
            mask = (score > threshold)
            rank_pattern[name] = mask
        
        return rank_pattern
    
    def update_and_allocate(self, step, force_update=False):
        """Update importance scores and allocate budget"""
        # Update importance scores
        self.update_importance_score()
        
        # Compute budget according to schedule
        budget, should_mask = self.budget_schedule(step)
        
        # If masks need updating or forced update
        if should_mask or force_update:
            # Allocate budget based on importance scores
            rank_pattern = self.allocate_budget(budget)
            
            # Apply masks to each layer
            for name, mask in rank_pattern.items():
                if name in self.adalora_layers:
                    layer = self.adalora_layers[name]
                    if hasattr(layer, 'mask_using_rank_pattern'):
                        layer.mask_using_rank_pattern(mask)
                        # Update effective rank
                        layer.update_rank(mask)
            
            return True, budget
        
        return False, budget


class AFNO3DAdaLora(nn.Module):
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
        lora_r=32,
        lora_alpha=1.0
    ):
        """
        AFNO3D version with AdaLoRA injection
        """
        super(AFNO3DAdaLora, self).__init__()
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

        # AdaLoRA layers
        self.lora_w1_real = nn.ModuleList()
        self.lora_w1_imag = nn.ModuleList()
        self.lora_w2_real = nn.ModuleList()
        self.lora_w2_imag = nn.ModuleList()

        for _ in range(num_blocks):
            self.lora_w1_real.append(
                AdaLoraLayer(
                    in_features=self.block_size,
                    out_features=self.block_size * self.hidden_size_factor,
                    r=lora_r,
                    alpha=lora_alpha,
                    bias=False
                )
            )
            self.lora_w1_imag.append(
                AdaLoraLayer(
                    in_features=self.block_size,
                    out_features=self.block_size * self.hidden_size_factor,
                    r=lora_r,
                    alpha=lora_alpha,
                    bias=False
                )
            )
            self.lora_w2_real.append(
                AdaLoraLayer(
                    in_features=self.block_size * self.hidden_size_factor,
                    out_features=self.block_size,
                    r=lora_r,
                    alpha=lora_alpha,
                    bias=False
                )
            )
            self.lora_w2_imag.append(
                AdaLoraLayer(
                    in_features=self.block_size * self.hidden_size_factor,
                    out_features=self.block_size,
                    r=lora_r,
                    alpha=lora_alpha,
                    bias=False
                )
            )

    def forward(self, x, spatial_size=None):
        if self.channel_first:
            x = rearrange(x, 'b c x y z -> b x y z c')

        B, H, W, L, C = x.shape
        x_orig = x

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        dtype_float = x.real.dtype
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        o1_real = torch.zeros(
            [B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor],
            device=x.device, dtype=dtype_float
        )
        o1_imag = torch.zeros_like(o1_real)

        o2_real = torch.zeros(
            [B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size],
            device=x.device, dtype=dtype_float
        )
        o2_imag = torch.zeros_like(o2_real)

        kept_modes = self.modes

        x_sub_real = x[:, :kept_modes, :kept_modes, :self.temporal_modes].real
        x_sub_imag = x[:, :kept_modes, :kept_modes, :self.temporal_modes].imag

        for block_idx in range(self.num_blocks):
            w1_real_block = self.w1[0, block_idx]
            b1_real_block = self.b1[0, block_idx]
            w1_imag_block = self.w1[1, block_idx]
            b1_imag_block = self.b1[1, block_idx]

            in_real = x_sub_real[..., block_idx, :]
            in_imag = x_sub_imag[..., block_idx, :]

            flat_in_real = in_real.reshape(-1, self.block_size)
            flat_in_imag = in_imag.reshape(-1, self.block_size)

            # Use AdaLoRA layer
            out_real_part1 = self.lora_w1_real[block_idx](flat_in_real, w1_real_block, b1_real_block)
            out_real_part2 = self.lora_w1_imag[block_idx](flat_in_imag, w1_imag_block, b1_imag_block)
            flat_o1_real = out_real_part1 - out_real_part2

            out_imag_part1 = self.lora_w1_real[block_idx](flat_in_imag, w1_real_block, b1_real_block)
            out_imag_part2 = self.lora_w1_imag[block_idx](flat_in_real, w1_imag_block, b1_imag_block)
            flat_o1_imag = out_imag_part1 + out_imag_part2

            o1_real_block = flat_o1_real.view(*in_real.shape[:-1], self.block_size * self.hidden_size_factor)
            o1_imag_block = flat_o1_imag.view(*in_imag.shape[:-1], self.block_size * self.hidden_size_factor)

            o1_real[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx, :] = F.gelu(o1_real_block)
            o1_imag[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx, :] = F.gelu(o1_imag_block)

        for block_idx in range(self.num_blocks):
            w2_real_block = self.w2[0, block_idx]
            b2_real_block = self.b2[0, block_idx]
            w2_imag_block = self.w2[1, block_idx]
            b2_imag_block = self.b2[1, block_idx]

            in_o1_real = o1_real[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx, :]
            in_o1_imag = o1_imag[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx, :]

            flat_in_o1_real = in_o1_real.reshape(-1, self.block_size * self.hidden_size_factor)
            flat_in_o1_imag = in_o1_imag.reshape(-1, self.block_size * self.hidden_size_factor)

            # Use AdaLoRA layer
            out_real_part1 = self.lora_w2_real[block_idx](flat_in_o1_real, w2_real_block, b2_real_block)
            out_real_part2 = self.lora_w2_imag[block_idx](flat_in_o1_imag, w2_imag_block, b2_imag_block)
            flat_o2_real = out_real_part1 - out_real_part2

            out_imag_part1 = self.lora_w2_real[block_idx](flat_in_o1_imag, w2_real_block, b2_real_block)
            out_imag_part2 = self.lora_w2_imag[block_idx](flat_in_o1_real, w2_imag_block, b2_imag_block)
            flat_o2_imag = out_imag_part1 + out_imag_part2

            o2_real_block = flat_o2_real.view(*in_o1_real.shape[:-1], self.block_size)
            o2_imag_block = flat_o2_imag.view(*in_o1_imag.shape[:-1], self.block_size)

            o2_real[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx, :] = o2_real_block
            o2_imag[:, :kept_modes, :kept_modes, :self.temporal_modes, block_idx, :] = o2_imag_block

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(H, W, L), dim=(1, 2, 3), norm="ortho")

        x = x + x_orig

        if self.channel_first:
            x = rearrange(x, 'b x y z c -> b c x y z')

        return x


class PatchEmbedAdaLora(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, out_dim=128, act='gelu', 
                 use_lora=True, lora_r=32, lora_alpha=1.0):
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
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        
        # Original projection layers
        self.conv1 = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.conv2 = nn.Conv3d(embed_dim, out_dim, kernel_size=1, stride=1)
        
        # Normal forward sequence
        self.proj = nn.Sequential(
            self.conv1,
            self.act,
            self.conv2
        )
        
        # Add AdaLoRA parameters for convolutions
        if use_lora:
            # AdaLoRA for the first convolution
            self.lora_down_conv1 = nn.Conv3d(in_chans, lora_r, kernel_size=patch_size, stride=patch_size, bias=False)
            self.lora_up_conv1 = nn.Conv3d(lora_r, embed_dim, kernel_size=1, stride=1, bias=False)
            self.lora_E_conv1 = nn.Parameter(torch.zeros(lora_r, 1))  # singular values
            self.lora_scale_conv1 = lora_alpha / lora_r
            self.ranknum_conv1 = nn.Parameter(torch.ones(1) * lora_r, requires_grad=False)
            
            # AdaLoRA for the second convolution
            self.lora_down_conv2 = nn.Conv3d(embed_dim, lora_r, kernel_size=1, stride=1, bias=False)
            self.lora_up_conv2 = nn.Conv3d(lora_r, out_dim, kernel_size=1, stride=1, bias=False)
            self.lora_E_conv2 = nn.Parameter(torch.zeros(lora_r, 1))  # singular values
            self.lora_scale_conv2 = lora_alpha / lora_r
            self.ranknum_conv2 = nn.Parameter(torch.ones(1) * lora_r, requires_grad=False)
            
            # Initialization
            nn.init.kaiming_uniform_(self.lora_down_conv1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up_conv1.weight)
            nn.init.zeros_(self.lora_E_conv1)
            
            nn.init.kaiming_uniform_(self.lora_down_conv2.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up_conv2.weight)
            nn.init.zeros_(self.lora_E_conv2)
    
    def update_rank(self, rank_pattern, conv_idx=None):
        """Update effective rank for the layer"""
        if isinstance(rank_pattern, list) or isinstance(rank_pattern, torch.Tensor):
            # If index list or tensor, count non-zeros as new rank
            if isinstance(rank_pattern, list):
                rank = sum(rank_pattern)
            else:  # tensor
                rank = rank_pattern.sum().item()
            
            # Update ranknum
            with torch.no_grad():
                if conv_idx == 1 or conv_idx is None:
                    self.ranknum_conv1.fill_(float(rank))
                if conv_idx == 2 or conv_idx is None:
                    self.ranknum_conv2.fill_(float(rank))
    
    def mask_using_rank_pattern(self, rank_pattern, conv_idx=None):
        """Mask unimportant parameters with importance pattern"""
        if isinstance(rank_pattern, torch.Tensor):
            # Convert to boolean mask
            mask = rank_pattern.bool()
        elif isinstance(rank_pattern, list):
            # Create boolean mask
            mask = torch.zeros(self.lora_r, dtype=torch.bool, device=self.lora_E_conv1.device)
            for idx in rank_pattern:
                mask[idx] = True
        else:
            raise ValueError("rank_pattern must be tensor or list")
        
        # Mask unimportant parameters
        with torch.no_grad():
            if conv_idx == 1 or conv_idx is None:
                # Apply mask to first conv AdaLora parameters
                mask_expanded = mask.view(-1, 1, 1, 1, 1).expand_as(self.lora_down_conv1.weight)
                self.lora_down_conv1.weight.masked_fill_(~mask_expanded, 0.0)
                
                # Mask singular values
                mask_e = mask.view(-1, 1).expand_as(self.lora_E_conv1)
                self.lora_E_conv1.masked_fill_(~mask_e, 0.0)
                
                # Apply column mask to up projection
                for i in range(self.lora_up_conv1.weight.size(0)):
                    for j in range(mask.size(0)):
                        if not mask[j]:
                            self.lora_up_conv1.weight[i, j] = 0.0
            
            if conv_idx == 2 or conv_idx is None:
                # Apply mask to second conv AdaLora parameters
                mask_expanded = mask.view(-1, 1, 1, 1, 1).expand_as(self.lora_down_conv2.weight)
                self.lora_down_conv2.weight.masked_fill_(~mask_expanded, 0.0)
                
                # Mask singular values
                mask_e = mask.view(-1, 1).expand_as(self.lora_E_conv2)
                self.lora_E_conv2.masked_fill_(~mask_e, 0.0)
                
                # Apply column mask to up projection
                for i in range(self.lora_up_conv2.weight.size(0)):
                    for j in range(mask.size(0)):
                        if not mask[j]:
                            self.lora_up_conv2.weight[i, j] = 0.0

    def forward(self, x):
        B, C, H, W, L = x.shape
        assert H == self.img_size[0] and W == self.img_size[1] and L == self.img_size[2], \
            f"Input image size ({H}*{W}*{L}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        
        if not self.use_lora:
            # If not using AdaLora, use original projection layers
            x = self.proj(x)
        else:
            # First convolution
            x1 = self.conv1(x)
            
            # AdaLora delta for first convolution
            if hasattr(self, 'lora_down_conv1'):
                down_hidden = self.lora_down_conv1(x)
                # Apply singular values
                down_hidden = down_hidden * self.lora_E_conv1.view(1, -1, 1, 1, 1)
                lora_x1 = self.lora_up_conv1(down_hidden) * self.lora_scale_conv1 / (self.ranknum_conv1 + 1e-5)
                x1 = x1 + lora_x1
            
            # Apply activation function
            x1_activated = self.act(x1)
            
            # Second convolution
            x2 = self.conv2(x1_activated)
            
            # AdaLora delta for second convolution
            if hasattr(self, 'lora_down_conv2'):
                down_hidden = self.lora_down_conv2(x1_activated)
                # Apply singular values
                down_hidden = down_hidden * self.lora_E_conv2.view(1, -1, 1, 1, 1)
                lora_x2 = self.lora_up_conv2(down_hidden) * self.lora_scale_conv2 / (self.ranknum_conv2 + 1e-5)
                x = x2 + lora_x2
            else:
                x = x2
        
        return x
    
    def enable_lora(self, enabled=True):
        """Enable or disable LoRA layers"""
        self.use_lora = enabled


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


class Block(nn.Module):
    def __init__(self, mixing_type='afno', double_skip=True, width=32, n_blocks=4, mlp_ratio=1., 
                 channel_first=True, modes=32, drop=0., drop_path=0., act='gelu', h=14, w=8, 
                 lora_r=32, lora_alpha=1.0):
        super().__init__()
        self.norm1 = torch.nn.GroupNorm(8, width)
        self.width = width
        self.modes = modes
        self.act = ACTIVATION[act]

        if mixing_type == "afno":
            self.filter = AFNO3DAdaLora(
                width=width, 
                num_blocks=n_blocks, 
                sparsity_threshold=0.01, 
                channel_first=channel_first, 
                modes=modes,
                hard_thresholding_fraction=1, 
                hidden_size_factor=1, 
                act=act,
                lora_r=lora_r,
                lora_alpha=lora_alpha
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


class DPOTNet3D_AdaLora(nn.Module):
    def __init__(self, img_size=224, patch_size=16, mixing_type='afno', in_channels=1, out_channels=3, 
                 in_timesteps=1, out_timesteps=1, n_blocks=4, embed_dim=768, out_layer_dim=32, 
                 depth=12, modes=32, mlp_ratio=1., n_cls=1, normalize=False, act='gelu', 
                 time_agg='exp_mlp', lora_r=32, lora_alpha=1.0, 
                 target_r=8, total_step=1000, tinit=200, tfinal=800):
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
        
        # AdaLora configuration
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.target_r = target_r
        self.total_step = total_step
        self.tinit = tinit
        self.tfinal = tfinal
        
        # Use PatchEmbed with AdaLora
        self.patch_embed = PatchEmbedAdaLora(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_channels + 4, 
            embed_dim=out_channels * patch_size + 4, 
            out_dim=embed_dim,
            act=act,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha
        )

        self.latent_size = self.patch_embed.out_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.patch_embed.out_size[0], self.patch_embed.out_size[1], self.patch_embed.out_size[2]))
        self.normalize = normalize
        self.time_agg = time_agg
        self.n_cls = n_cls

        h = img_size // patch_size
        w = h // 2 + 1

        # Create Block layers with AdaLora
        self.blocks = nn.ModuleList([
            Block(mixing_type=mixing_type, modes=modes,
                  width=embed_dim, mlp_ratio=mlp_ratio, channel_first=True, n_blocks=n_blocks, 
                  double_skip=False, h=h, w=w, act=act,
                  lora_r=lora_r, lora_alpha=lora_alpha)
            for i in range(depth)])

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
        self.out_layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels=embed_dim, out_channels=out_layer_dim, kernel_size=patch_size, stride=patch_size),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=out_layer_dim, kernel_size=1, stride=1),
            self.act,
            nn.Conv3d(in_channels=out_layer_dim, out_channels=self.out_channels * self.out_timesteps, kernel_size=1, stride=1)
        )

        torch.nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.mixing_type = mixing_type
        
        # RankAllocator manages AdaLora rank allocation
        self.rank_allocator = None

    def initialize_rank_allocator(self, total_step=None):
        """Initialize RankAllocator"""
        if total_step is not None:
            self.total_step = total_step
            
        self.rank_allocator = RankAllocator(
            model=self,
            target_r=self.target_r,
            init_r=self.lora_r,
            total_step=self.total_step,
            tinit=self.tinit,
            tfinal=self.tfinal
        )
        return self.rank_allocator
    
    def update_adalora_ranks(self, step, force_update=False):
        """Update AdaLora rank allocation"""
        if self.rank_allocator is None:
            self.initialize_rank_allocator()
        
        return self.rank_allocator.update_and_allocate(step, force_update)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.trunc_normal_(m.weight, std=.002)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

        grid = self.get_grid_4d(x)
        x = torch.cat((x, grid), dim=-1).contiguous()  # B, X, Y, Z, T, C+4
        x = rearrange(x, 'b x y z t c -> (b t) c x y z')
        x = self.patch_embed(x)  # Use PatchEmbed with AdaLora

        x = x + self.pos_embed

        x = rearrange(x, '(b t) c x y z -> b x y z t c', b=B, t=T)

        x = self.time_agg_layer(x)

        x = rearrange(x, 'b x y z c -> b c x y z')

        if self.normalize:
            x = scale_sigma * x + scale_mu

        # Pass through each Block layer
        for blk in self.blocks:
            x = blk(x)

        # Output layer
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
    """
    Resize positional embeddings for loading pretrained weights
    """
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
    """
    Filter checkpoint state dict for loading pretrained weights
    """
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For older models trained before conv-based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # Resize positional embeddings to match model size
            v = resize_pos_embed(v, model.pos_embed)
        out_dict[k] = v
    return out_dict


if __name__ == "__main__":
    # Test code
    x = torch.rand(2, 64, 64, 64, 10, 3)

    from utils.utilities import load_3d_components_from_2d
    load_path = "/root/autodl-tmp/checkpoints/model_S.pth"
    
    # Create AdaLoRA version of DPOT model
    net = DPOTNet3D_AdaLora(
        img_size=64, 
        patch_size=8, 
        in_channels=3, 
        out_channels=3, 
        in_timesteps=10, 
        embed_dim=1024,
        n_blocks=8, 
        depth=6,
        lora_r=32,
        lora_alpha=1.0,
        target_r=8,
        total_step=10000
    )
    
    # Initialize RankAllocator
    net.initialize_rank_allocator()
    
    # Load pretrained weights
    state_dict = torch.load(load_path, map_location='cpu')['model']
    load_3d_components_from_2d(net, state_dict, ['blocks','time_agg'])
    
    # Forward pass test
    y = net(x)
    print(y.shape)
