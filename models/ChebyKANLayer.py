import torch
import torch.nn as nn


# This is inspired by Kolmogorov-Arnold Networks but using Chebyshev polynomials instead of splines coefficients
class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))
        
        # Add shape attribute for compatibility with standard tensors
        self.shape = self.cheby_coeffs.shape

        # Add debug flag
        self.debug = True

    def forward(self, x):
        # Log input state
        if self.debug and torch.isnan(x).any():
            print("ChebyKANLayer input contains NaN!")

        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)

        # Check values after tanh
        if self.debug and torch.isnan(x).any():
            print("NaN appeared after tanh!")

        # Ensure values are strictly in [-1, 1] range to avoid numerical issues with acos
        x = torch.clamp(x, -0.999, 0.999)

        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)

        # Check values after reshape
        if self.debug and torch.isnan(x).any():
            print("NaN appeared after reshape!")

        # Apply acos
        x = x.acos()

        # Check values after acos
        if self.debug and torch.isnan(x).any():
            print("NaN appeared after acos!")

        # Multiply by arange [0 .. degree]
        arange = self._buffers['arange']  # Access directly from _buffers dictionary
        x *= arange

        # Check values after multiplying by arange
        if self.debug and torch.isnan(x).any():
            print("NaN appeared after multiplying by arange!")

        # Apply cos
        x = x.cos()

        # Check values after cos
        if self.debug and torch.isnan(x).any():
            print("NaN appeared after cos!")

        # Check coefficients
        if self.debug and torch.isnan(self.cheby_coeffs).any():
            print("cheby_coeffs contains NaN!")

        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)

        # Check einsum result
        if self.debug and torch.isnan(y).any():
            print("NaN appeared after einsum computation!")

        y = y.view(-1, self.outdim)

        # Check final output
        if self.debug and torch.isnan(y).any():
            print("ChebyKANLayer final output contains NaN!")

        return y

    # Modify size method for full compatibility with standard tensors
    def size(self, dim=None):
        if dim is None:
            return self.cheby_coeffs.size()
        else:
            return self.cheby_coeffs.size(dim)

    # Add dim method to return the number of tensor dimensions
    def dim(self):
        return self.cheby_coeffs.dim()

    # Modify __getattr__ method to correctly handle special attributes
    def __getattr__(self, name):
        # Special handling for buffer attributes
        if name in self._buffers:
            return self._buffers[name]

        if name == 'cheby_coeffs':
            # Avoid recursive calls
            return super(ChebyKANLayer, self).__getattr__(name)

        # Forward attribute access to cheby_coeffs parameter
        try:
            return getattr(self.cheby_coeffs, name)
        except AttributeError:
            raise AttributeError(f"'ChebyKANLayer' object has no attribute '{name}'")