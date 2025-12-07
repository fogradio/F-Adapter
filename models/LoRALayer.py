import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1., bias=False, 
                 conv=False, transpose=False, kernel_size=1, stride=1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.conv = conv
        self.transpose = transpose
        
        if conv:
            if transpose:
                self.lora_down = nn.Conv3d(in_features, r, kernel_size=1, stride=1, bias=False)
                self.lora_up = nn.ConvTranspose3d(r, out_features, kernel_size=kernel_size, 
                                                stride=stride, bias=False)
            else:
                self.lora_down = nn.Conv3d(in_features, r, kernel_size=1, stride=1, bias=False)
                self.lora_up = nn.Conv3d(r, out_features, kernel_size=kernel_size, 
                                       stride=stride, bias=False)
        else:
            self.lora_down = nn.Linear(in_features, r, bias=False)
            self.lora_up = nn.Linear(r, out_features, bias=False)
            
        # Initialize
        if not transpose:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        self.scale = alpha / r
        
    def forward(self, x, original_weight=None, original_bias=None):
        if original_weight is not None:
            # UseOriginal weight
            if self.conv:
                if self.transpose:
                    out = F.conv_transpose3d(x, original_weight, original_bias, 
                                           stride=self.lora_up.stride)
                else:
                    out = F.conv3d(x, original_weight, original_bias)
            else:
                out = F.linear(x, original_weight, original_bias)
        else:
            out = 0
            
        # LoRA path
        lora = self.lora_up(self.lora_down(x)) * self.scale
        
        return out + lora
