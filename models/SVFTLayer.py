import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

class SVFTLayer(nn.Module):
    """
    SVFT Layer: Parameter-efficient fine-tuning method based on SVD decomposition and sparse updates
    Completely refactored to be compatible with 3D convolution operations while preserving SVFT core ideas
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 r: int = 32,  # Rank for low-rank approximation
                 alpha: float = 1.0,  # Scaling factor
                 bias: bool = False,
                 off_diag: int = 2,  # Diagonal bandwidth
                 pattern: str = 'banded',  # Sparse pattern
                 init_option: str = 'zero'):
        super().__init__()
        self.r = min(r, min(in_features, out_features))  # Ensure rank does not exceed minimum dimension
        self.alpha = alpha
        self.in_features = in_features
        self.out_features = out_features
        self.off_diag = off_diag
        self.pattern = pattern
        
        # Create U, S, V matrices (not trainable) - for storing SVD decomposition of original weight
        self.u = nn.Parameter(torch.zeros((out_features, self.r)), requires_grad=False)
        self.s_pre = nn.Parameter(torch.zeros(self.r), requires_grad=False)
        self.v = nn.Parameter(torch.zeros((self.r, in_features)), requires_grad=False)
        
        # Create sparse matrix indices
        if pattern == "banded":
            indices = []
            for i in range(self.r):
                for j in range(max(0, i-off_diag), min(self.r, i+off_diag+1)):
                    indices.append([i, j])
            indices = torch.tensor(indices, dtype=torch.long)
        elif pattern == "random" or pattern == "top_k":
            k = min(self.r * (2 * off_diag + 1) - off_diag * (off_diag + 1), self.r * self.r)
            rows = torch.randint(0, self.r, (k,))
            cols = torch.randint(0, self.r, (k,))
            indices = torch.stack([rows, cols], dim=1)
        
        # Register sparse matrix indices and values
        self.register_buffer('sparse_indices', indices)
        self.s = nn.Parameter(torch.zeros(indices.size(0)))
        
        # Gating mechanism
        self.gate = nn.Parameter(torch.tensor([0.], dtype=torch.float32))
        
        # Optional bias parameter
        if bias:
            self.lora_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('lora_bias', None)
        
        # Scaling parameter
        self.scaling = self.alpha / self.r
        
        # Initialize model parameters
        self._init_weights(init_option)
        
        # Initialization flag
        self.initialized = False
    
    def _init_weights(self, init_option):
        """Initialize weights"""
        if init_option == 'zero':
            nn.init.zeros_(self.s)
        elif init_option == 'normal':
            nn.init.normal_(self.s, mean=0.0, std=0.02)
        else:
            nn.init.kaiming_uniform_(self.s, a=math.sqrt(5))
    
    def _create_sparse_matrix(self, indices, values, shape):
        """Create dense form of sparse matrix"""
        matrix = torch.zeros(shape, device=values.device)
        for idx, (i, j) in enumerate(indices):
            if i < shape[0] and j < shape[1]:  # Ensure indices don't exceed bounds
                matrix[i, j] = values[idx]
        return matrix
    
    def initialize_from_weight(self, weight):
        """Initialize SVD decomposition parameters from weight"""
        if self.initialized:
            return

        # Check if weight shape matches expected
        if weight.shape[0] != self.out_features or weight.shape[1] != self.in_features:
            warnings.warn(f"Weight shape {weight.shape} does not match expected [{self.out_features}, {self.in_features}], performing dimension adjustment")

            # Create correctly sized weight matrix
            adjusted_weight = torch.zeros((self.out_features, self.in_features), device=weight.device)
            
            # Copy common portion
            min_out = min(weight.shape[0], self.out_features)
            min_in = min(weight.shape[1], self.in_features)
            adjusted_weight[:min_out, :min_in] = weight[:min_out, :min_in]
            weight = adjusted_weight

        # Perform SVD decomposition
        try:
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            rank = min(self.r, u.shape[1], s.shape[0], vh.shape[0])
            u_r = u[:, :rank]
            s_r = s[:rank]
            vh_r = vh[:rank, :]
        except Exception as e:
            warnings.warn(f"SVD computation failed: {e}, using random initialization")
            u_r = torch.zeros((self.out_features, self.r), device=weight.device)
            s_r = torch.ones(self.r, device=weight.device) * 0.01
            vh_r = torch.zeros((self.r, self.in_features), device=weight.device)
            
            # Use orthogonal initialization
            nn.init.orthogonal_(u_r)
            nn.init.orthogonal_(vh_r)

        # Update model parameters
        with torch.no_grad():
            self.u.copy_(u_r)
            self.s_pre.copy_(s_r)
            self.v.copy_(vh_r)
        
        self.initialized = True
    
    def forward(self, x, original_weight, original_bias=None):
        """
        Forward pass
        x: Input tensor [batch_size, *, in_features] or [batch_size, num_patches, in_features]
        original_weight: Original weight matrix [out_features, in_features]
        original_bias: Bias vector [out_features]
        """
        # Save original shape
        original_shape = x.shape
        
        # Ensure last dimension is feature dimension
        if len(original_shape) > 2:
            x_flat = x.reshape(-1, original_shape[-1])
        else:
            x_flat = x
            
        # Check if input and weight dimensions match
        if original_weight.shape[1] != x_flat.shape[1]:
            min_dim = min(original_weight.shape[1], x_flat.shape[1])
            warnings.warn(f"Input feature dimension {x_flat.shape[1]} does not match weight feature dimension {original_weight.shape[1]}")

            # Adjust weight dimensions
            if original_weight.shape[1] > min_dim:
                adjusted_weight = original_weight[:, :min_dim]
            else:
                adjusted_weight = torch.zeros(original_weight.shape[0], x_flat.shape[1], 
                                             device=original_weight.device)
                adjusted_weight[:, :min_dim] = original_weight[:, :min_dim]
            
            original_weight = adjusted_weight
        
        # Original linear transformation
        result = torch.matmul(x_flat, original_weight.T)
        
        # Initialize SVD parameters (if needed)
        if not self.initialized:
            self.initialize_from_weight(original_weight)
        
        # Adjust V matrix to match input
        if self.v.shape[1] != x_flat.shape[1]:
            min_dim = min(self.v.shape[1], x_flat.shape[1])
            v_adapted = torch.zeros(self.v.shape[0], x_flat.shape[1], device=self.v.device)
            v_adapted[:, :min_dim] = self.v[:, :min_dim]
        else:
            v_adapted = self.v
        
        # Create diagonal matrix S_pre
        s_pre_diag = torch.diag(self.s_pre)

        # Create delta matrix S (sparse update)
        gate_value = torch.sigmoid(self.gate)
        s_matrix = self._create_sparse_matrix(
            self.sparse_indices, 
            self.s * gate_value, 
            (self.r, self.r)
        )
        
        # Combine two matrices (original singular values + sparse update)
        S_combined = s_pre_diag + s_matrix
        
        # Compute x @ (V.T @ S_combined @ U.T) - SVFT's core computation
        # Equivalent to x @ (original weight + low-rank update)
        VS = torch.matmul(v_adapted.T, S_combined)
        VSU = torch.matmul(VS, self.u.T)
        
        # Compute delta and add to result
        delta = torch.matmul(x_flat, VSU) * self.scaling
        result = result + delta
        
        # Add bias
        if original_bias is not None:
            result = result + original_bias
        if self.lora_bias is not None:
            result = result + self.lora_bias
        
        # Restore original shape
        if len(original_shape) > 2:
            new_shape = original_shape[:-1] + (original_weight.shape[0],)
            result = result.reshape(new_shape)
        
        return result

    def get_effective_weight(self, original_weight=None):
        """Compute effective weight matrix (original weight + SVFT delta)"""
        # Ensure initialized
        if not self.initialized and original_weight is not None:
            self.initialize_from_weight(original_weight)
            
        if not self.initialized:
            return None
        
        # Create diagonal matrix S_pre
        s_pre_diag = torch.diag(self.s_pre)

        # Create delta matrix S
        gate_value = torch.sigmoid(self.gate)
        s_matrix = self._create_sparse_matrix(
            self.sparse_indices, 
            self.s * gate_value, 
            (self.r, self.r)
        )
        
        # Combine two matrices
        S_combined = s_pre_diag + s_matrix
        
        # Compute U @ S_combined @ V
        US = torch.matmul(self.u, S_combined)
        USV = torch.matmul(US, self.v)
        
        return USV * self.scaling
    
    def reshape_for_fft(self, x_ft, original_shape, target_shape):
        """
        Safely reshape frequency domain data to target shape, handling dimension mismatch
        """
        try:
            return x_ft.reshape(target_shape)
        except RuntimeError as e:
            # Get current data size and target size
            current_size = x_ft.numel()
            target_size = np.prod(target_shape)
            warnings.warn(f"Cannot reshape tensor with shape {x_ft.shape} to {target_shape}, "
                          f"current size {current_size}, target size {target_size}")

            # Create zero tensor with appropriate size as replacement
            if hasattr(x_ft, 'real'):  # Handle complex tensor
                result = torch.zeros(target_shape, dtype=torch.complex64, device=x_ft.device)
                # Copy original data as much as possible
                flat_target = result.reshape(-1)
                flat_source = x_ft.reshape(-1)
                copy_size = min(flat_target.numel(), flat_source.numel())
                flat_target[:copy_size] = flat_source[:copy_size]
            else:
                result = torch.zeros(target_shape, dtype=x_ft.dtype, device=x_ft.device)
                # Copy original data as much as possible
                flat_target = result.reshape(-1)
                flat_source = x_ft.reshape(-1)
                copy_size = min(flat_target.numel(), flat_source.numel())
                flat_target[:copy_size] = flat_source[:copy_size]
            
            return result