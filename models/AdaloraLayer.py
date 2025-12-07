import torch
import torch.nn as nn
import math

class AdaLoraLayer(nn.Module):
    """
    AdaLora Layer: Based on paper https://openreview.net/pdf?id=lq62uWRJjiY
    Adaptive low-rank parameter-efficient fine-tuning method combining SVD decomposition, maintaining similar interface as LoRALayer
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 32, 
                 alpha: float = 1.0, 
                 bias: bool = False,
                 init_option: str = 'zero'):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        
        # AdaLora: Three parameters - A for right singular vectors, E for singular values, B for left singular vectors
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_E = nn.Parameter(torch.zeros((r, 1)))  # Singular values
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        # Current rank (used for scaling and also for masking unimportant parameters)
        self.ranknum = nn.Parameter(torch.ones(1) * r, requires_grad=False)

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
        """Initialize weights, supporting different initialization strategies"""
        if init_option == 'zero':
            # Initialize E to zero, A/B with normal distribution
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        elif init_option == 'normal':
            # All parameters use normal distribution
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_E, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            # Default initialization method
            nn.init.zeros_(self.lora_E)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, x, original_weight, original_bias=None):
        """
        Forward pass:
        x: Input tensor [batch, ..., in_features]
        original_weight: Original weight [out_features, in_features]
        original_bias: Original bias [out_features]
        """
        # Original linear transformation
        result = torch.matmul(x, original_weight.T)

        # AdaLora delta: x @ (A * E)^T @ B^T
        # Need to consider the effect of ranknum during computation
        delta = torch.matmul(
            torch.matmul(x, (self.lora_A * self.lora_E).T),
            self.lora_B.T
        ) * self.scaling / (self.ranknum + 1e-5)  # Prevent division by zero

        result += delta

        # Add bias
        if original_bias is not None:
            result += original_bias
        if self.lora_bias is not None:
            result += self.lora_bias

        return result

    def update_rank(self, rank_pattern):
        """
        Update the effective rank of the layer, used for weight pruning
        rank_pattern: Boolean mask or index list indicating which dimensions to retain
        """
        if isinstance(rank_pattern, list) or isinstance(rank_pattern, torch.Tensor):
            # If it's an index list or tensor, calculate the number of non-zero elements as the new rank
            if isinstance(rank_pattern, list):
                rank = sum(rank_pattern)
            else:  # tensor
                rank = rank_pattern.sum().item()

            # Update ranknum
            with torch.no_grad():
                self.ranknum.fill_(float(rank))

    def mask_using_rank_pattern(self, rank_pattern):
        """
        Mask unimportant parameters using importance pattern
        rank_pattern: Boolean mask or index list
        """
        if isinstance(rank_pattern, torch.Tensor):
            # Convert to boolean mask
            mask = rank_pattern.bool()
        elif isinstance(rank_pattern, list):
            # Create boolean mask
            mask = torch.zeros(self.r, dtype=torch.bool, device=self.lora_A.device)
            for idx in rank_pattern:
                mask[idx] = True
        else:
            raise ValueError("rank_pattern must be tensor or list type")

        # Mask unimportant parameters
        with torch.no_grad():
            # Create expanded masks
            mask_a = mask.unsqueeze(-1).expand_as(self.lora_A)
            mask_e = mask.unsqueeze(-1).expand_as(self.lora_E)
            mask_b = mask.unsqueeze(0).expand_as(self.lora_B)

            # Apply masks
            self.lora_A.masked_fill_(~mask_a, 0.0)
            self.lora_E.masked_fill_(~mask_e, 0.0)
            self.lora_B.masked_fill_(~mask_b, 0.0)