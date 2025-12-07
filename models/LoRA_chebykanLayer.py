import torch
import torch.nn as nn
from models.ChebyKANLayer import ChebyKANLayer

class LoRALayer(nn.Module):
    """
    Take the low-rank incremental representation of a linear layer as an example.
    Use a ChebyKANLayer as the non-linear transform layer instead of a linear one.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 32, 
                 alpha: float = 1.0, 
                 bias: bool = False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        
        # LoRA: UseChebyKANLayerNOTTransform
        self.lora_A = ChebyKANLayer(input_dim=in_features, output_dim=r, degree=8)
        self.lora_B = ChebyKANLayer(input_dim=r, output_dim=out_features, degree=8)

        # scaling
        self.scaling = self.alpha / self.r

        # Original bias CanChoice,
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)

        # Initialization strategy - ApplyChebyKAN
        # ZeroInitializeB , EnsureTrainingStartLoRA
        with torch.no_grad():
            nn.init.zeros_(self.lora_B.cheby_coeffs)

    def forward(self, x, original_weight, original_bias=None):
        # x Shape: [batch, ..., in_features]
        # original_weight: [out_features, in_features]
        
        # ComputeOriginalLayer Output
        result = torch.matmul(x, original_weight.T)
        
        # ComputeLoRA
        # NoteUseChebyKANLayer forwardMethod, WhileConvertMatrix
        lora_a_output = self.lora_A(x)  # [batch, r]
        lora_b_output = self.lora_B(lora_a_output)  # [batch, out_features]
        
        # ApplyScaleAddTo
        result += lora_b_output * self.scaling
        
        # Add bias
        if original_bias is not None:
            result += original_bias
        if self.lora_bias is not None:
            result += self.lora_bias
            
        return result
# , NaN？