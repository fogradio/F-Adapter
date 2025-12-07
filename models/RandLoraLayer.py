import torch
import torch.nn as nn
import math

class RandLoraLayer(nn.Module):
    """
    RandLora Layer: Based on RandLoRA paper implementation
    Achieves parameter-efficient fine-tuning through random projection and low-rank decomposition, maintaining similar interface as other LoRA variants
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 32,  # LoRA rank
                 alpha: float = 1.0,  # LoRA scaling
                 bias: bool = False,
                 proj_type: str = 'gaussian',  # Random projection type: 'gaussian' or 'rademacher'
                 init_option: str = 'zero',
                 proj_factor: int = 4  # Random projection dimension expansion factor
                ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.proj_type = proj_type
        self.proj_dim = r * proj_factor  # Random projection space dimension

        # RandLoRA parameters
        # Q: Random projection matrix, fixed and not trained
        self.register_buffer('rand_proj_Q', self._init_random_matrix())
        
        # P, R: Trainable low-rank matrices
        self.lora_P = nn.Parameter(torch.zeros((out_features, r)))  # Left matrix
        self.lora_R = nn.Parameter(torch.zeros((r, self.proj_dim)))  # Right matrix
        
        # scaling
        self.scaling = self.alpha / self.r
        
        # Optional bias parameter
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)
        
        # Initialize parameters
        self._init_weights(init_option)
    
    def _init_random_matrix(self):
        """Initialize random projection matrix Q"""
        if self.proj_type == 'gaussian':
            # Gaussian random projection
            Q = torch.randn(self.in_features, self.proj_dim) / math.sqrt(self.proj_dim)
        elif self.proj_type == 'rademacher':
            # Rademacher random projection (+1/-1)
            Q = torch.randint(0, 2, (self.in_features, self.proj_dim)) * 2 - 1
            Q = Q / math.sqrt(self.proj_dim)
        else:
            raise ValueError(f"Unsupported projection type: {self.proj_type}")
        return Q
    
    def _init_weights(self, init_option):
        """Initialize weights"""
        if init_option == 'zero':
            # P initialized with normal distribution, R initialized to zero
            nn.init.normal_(self.lora_P, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_R)
        elif init_option == 'normal':
            # All parameters use normal distribution
            nn.init.normal_(self.lora_P, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_R, mean=0.0, std=0.02)
        else:
            # Default initialization
            nn.init.kaiming_uniform_(self.lora_P, a=math.sqrt(5))
            nn.init.zeros_(self.lora_R)
    
    def forward(self, x, original_weight, original_bias=None):
        """
        Forward pass: Comprehensively handles dimension differences between convolutional and linear layers
        x: Input tensor [batch, ..., in_features_actual]
        original_weight: Original weight [out_features, in_features_expected]
        """
        # Get input and weight dimensions
        batch_size = x.shape[0]
        actual_in_dim = x.shape[-1]
        expected_in_dim = self.in_features
        out_dim = self.out_features
        
        # Handle shape mismatch issues
        if actual_in_dim != expected_in_dim:
            print(f"Dimension adjustment: input={actual_in_dim}, expected={expected_in_dim}, weight={original_weight.shape}")

            # 1. Create new projection matrix based on actual input dimension
            device = x.device
            if self.proj_type == 'gaussian':
                new_Q = torch.randn(actual_in_dim, self.proj_dim, device=device) / math.sqrt(self.proj_dim)
            else:  # rademacher
                new_Q = torch.randint(0, 2, (actual_in_dim, self.proj_dim), device=device) * 2 - 1
                new_Q = new_Q / math.sqrt(self.proj_dim)
            
            # 2. Compute original output - handle potential weight mismatch
            if original_weight.shape[-1] >= actual_in_dim:
                # Weight is larger, extract matching portion
                adapted_weight = original_weight[:, :actual_in_dim]
                result = torch.matmul(x, adapted_weight.T)
            else:
                # Weight is smaller, use zero padding
                print(f"Warning: weight dimension ({original_weight.shape[-1]}) is smaller than input dimension ({actual_in_dim})")
                padded_weight = torch.zeros(out_dim, actual_in_dim, device=x.device)
                padded_weight[:, :original_weight.shape[-1]] = original_weight
                result = torch.matmul(x, padded_weight.T)
            
            # 3. Compute RandLoRA delta using new projection matrix
            proj_x = torch.matmul(x, new_Q)
            
            # Ensure output shape is correct
            delta = torch.matmul(
                torch.matmul(proj_x, self.lora_R.T),
                self.lora_P.T
            ) * self.scaling
            
            # Ensure delta and result have the same shape
            if delta.shape != result.shape:
                print(f"Warning: delta shape ({delta.shape}) does not match original output ({result.shape})")
                # Rearrange dimensions to match
                if delta.dim() > result.dim():
                    # Reduce dimensions
                    delta = delta.reshape(*result.shape)
                elif delta.dim() < result.dim():
                    # Increase dimensions
                    delta = delta.reshape(*result.shape)
        else:
            # Standard processing - dimensions match
            result = torch.matmul(x, original_weight.T)
            proj_x = torch.matmul(x, self.rand_proj_Q)
            delta = torch.matmul(
                torch.matmul(proj_x, self.lora_R.T),
                self.lora_P.T
            ) * self.scaling
        
        # Add RandLoRA delta
        result += delta
        
        # Add bias
        if original_bias is not None:
            result += original_bias
        if self.lora_bias is not None:
            result += self.lora_bias
        
        return result
    
    def get_lora_parameters(self):
        """Get all trainable LoRA parameters"""
        params = [self.lora_P, self.lora_R]
        if self.lora_bias is not None:
            params.append(self.lora_bias)
        return params
    
    def reset_lora_parameters(self, init_option='zero'):
        """Reset LoRA parameters"""
        with torch.no_grad():
            self._init_weights(init_option)
    
    def effective_lora_matrix(self):
        """
        Compute effective LoRA weight matrix
        Returns: [out_features, in_features] matrix
        """
        return self.scaling * torch.matmul(
            self.lora_P,  # [out_features, r]
            torch.matmul(self.lora_R, self.rand_proj_Q.T)  # [r, in_features]
        )