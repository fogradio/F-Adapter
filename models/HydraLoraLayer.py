import torch
import torch.nn as nn
import math

class HydraLoraLayer(nn.Module):
    """
    HydraLoRA Layer: Based on the Method & Implementation of the HydraLoRA (MMOELoRA) paper.
    It shares a low-dimensional projection matrix A, while maintaining multiple high-dimensional projection matrices B as multi-head experts, enabling more efficient parameter sharing in practical implementations.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 32, 
                 alpha: float = 1.0, 
                 bias: bool = False,
                 expert_num: int = 4, 
                 init_option: str = 'zero'):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.expert_num = expert_num
        
        # ProjectionMatrixA
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        
        # BMatrix (Hydra )
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros((out_features, r))) 
            for _ in range(expert_num)
        ])
        
        # scaling
        self.scaling = self.alpha / self.r
        
        # Optional bias parameter
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)
        
        # Initialize parameters
        self._init_weights(init_option)
    
    def _init_weights(self, init_option):
        """InitializeWeight，Supporting different initialization strategies"""
        if init_option == 'zero':
            # AInitializePositive, BInitializeZero
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            for b in self.lora_B:
                nn.init.zeros_(b)
        elif init_option == 'normal':
            # All parameters use normal distribution
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            for b in self.lora_B:
                nn.init.normal_(b, mean=0.0, std=0.02)
        else:
            # Default initialization method
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            for b in self.lora_B:
                nn.init.zeros_(b)
    
    def forward(self, x, original_weight, original_bias=None, expert_weights=None):
        """
        Forward pass:
        x: Input tensor [batch, ..., in_features]
        original_weight: Original weight [out_features, in_features]
        original_bias: Original bias [out_features]
        expert_weights: MoE weight [batch_size, expert_num] OR [expert_num]
        """
        # Original linear transformation
        result = torch.matmul(x, original_weight.T)
        
        # IfWeight, AverageUse
        if expert_weights is None:
            expert_weights = torch.ones(self.expert_num, device=x.device) / self.expert_num
        
        # Checkexpert_weightsShape，EnsureCanBroadcast
        if expert_weights.dim() == 1:
            # If1D, SampleUse Weight
            expert_weights = expert_weights.view(1, -1)  # [1, expert_num]
        
        # ComputeLoRA: x @ A^T @ (∑ w_i * B_i)^T * scaling
        # Compute x @ A^T => [batch, ..., r]
        lora_a_output = torch.matmul(x, self.lora_A.T)
        
        # InitializeZero
        delta = torch.zeros_like(result)
        
        # Traverse,
        for i, lora_B_expert in enumerate(self.lora_B):
            # GetCurrentWeight w_i
            if expert_weights.size(1) > i:
                weight = expert_weights[..., i]
                while weight.dim() < delta.dim():
                    weight = weight.unsqueeze(-1)
                
                # ComputeCurrent : (x @ A^T) @ B_i^T * scaling * w_i
                expert_delta = torch.matmul(lora_a_output, lora_B_expert.T) * self.scaling * weight
                delta += expert_delta
        
        # AddLoRAToOriginalOutput
        result += delta
        
        # Add bias
        if original_bias is not None:
            result += original_bias
        if self.lora_bias is not None:
            result += self.lora_bias
        
        return result
    
    def get_expert_combinations(self, combination_weights):
        """
        Get the valid B-matrix after expert combination, for analysis or visualization
        combination_weights: Mixture-of-experts Weight [expert_num]
        Return: the combined BMatrix
        """
        if combination_weights.size(0) != self.expert_num:
            raise ValueError(f"Moe WeightNumber {combination_weights.size(0)} AND Exact Expert Number {self.expert_num} dDesn't Match")
        
        # InitializeBMatrix
        combined_B = torch.zeros_like(self.lora_B[0])
        
        # BMatrix
        for i, lora_B_expert in enumerate(self.lora_B):
            combined_B += combination_weights[i] * lora_B_expert
        
        return combined_B