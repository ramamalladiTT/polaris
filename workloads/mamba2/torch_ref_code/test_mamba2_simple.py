from mamba2_simple import Mamba2Simple
import torch
d_model_sz = 128

model = Mamba2Simple(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=d_model_sz, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
)
# Define input tensor x with shape (batch_size, sequence_length, d_model)
batch_size = 1
sequence_length = 16
x = torch.randn(batch_size, sequence_length, d_model_sz)

y = model(x)
print(f'input shape is {x.shape} and output shape is {y.shape}')
assert y.shape == x.shape, 'Test failed!'
print('Test passed!')
