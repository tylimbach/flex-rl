import triton
import triton.language as tl
import torch

@triton.jit
def mlp_fused_forward_kernel(
    # Pointers to matrices
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Fused matrix multiply and ReLU for policy network MLP layers using Tensor Cores"""
    pid = tl.program_id(axis=0)
    
    # Create block pointer for A (input) and B (weights)
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr, shape=(batch_size, in_features), 
        strides=(in_features, 1), offsets=(pid * BLOCK_SIZE_M, 0), 
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0)
    )
    w_block_ptr = tl.make_block_ptr(
        base=W_ptr, shape=(in_features, out_features), 
        strides=(out_features, 1), offsets=(0, 0), 
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0)
    )
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matmul loop
    for k in range(0, in_features, BLOCK_SIZE_K):
        x = tl.load(x_block_ptr, boundary_check=(0, 1))
        w = tl.load(w_block_ptr, boundary_check=(0, 1))
        
        # Matrix multiplication
        acc += tl.dot(x, w)
        
        # Advance block pointers
        x_block_ptr = tl.advance(x_block_ptr, offsets=(0, BLOCK_SIZE_K))
        w_block_ptr = tl.advance(w_block_ptr, offsets=(BLOCK_SIZE_K, 0))
    
    # Load bias
    bias = tl.load(B_ptr + tl.arange(0, BLOCK_SIZE_N))
    
    # Add bias and apply ReLU
    acc = acc + bias
    acc = tl.maximum(acc, 0.0)
    
    # Write output
    output_offset = pid * BLOCK_SIZE_M * out_features
    for m in range(BLOCK_SIZE_M):
        for n in range(BLOCK_SIZE_N):
            Y_ptr[output_offset + m * out_features + n] = acc[m, n]

# Python wrapper class
class TritonPolicyMLP(torch.nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes=[64, 64]):
        super().__init__()
        self.layers = []
        layer_sizes = [in_features] + hidden_sizes + [out_features]
        
        for i in range(len(layer_sizes) - 1):
            weights = torch.nn.Parameter(torch.empty(layer_sizes[i], layer_sizes[i+1]))
            bias = torch.nn.Parameter(torch.zeros(layer_sizes[i+1]))
            self.register_parameter(f'weight_{i}', weights)
            self.register_parameter(f'bias_{i}', bias)
            self.layers.append((weights, bias))
            
        # Initialization
        for i, (weight, bias) in enumerate(self.layers):
            torch.nn.init.orthogonal_(weight, gain=2.0 if i < len(self.layers)-1 else 0.01)
    
    def forward(self, x):
        batch_size = x.shape[0]
        for i, (weight, bias) in enumerate(self.layers[:-1]):
            # Use Triton kernel for hidden layers
            in_features, out_features = weight.shape
            y = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
            
            grid = (triton.cdiv(batch_size, 128),)
            mlp_fused_forward_kernel[grid](
                x, weight, bias, y, 
                batch_size, in_features, out_features,
                BLOCK_SIZE_M=128, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
            )
            x = y
            
        # Final layer (standard PyTorch for simplicity)
        weight, bias = self.layers[-1]
        return torch.nn.functional.linear(x, weight, bias)

# Integration with Stable Baselines3
def replace_policy_mlp_with_triton(model):
    """Replace the standard MLP in the policy with our Triton implementation"""
    if hasattr(model, 'policy') and hasattr(model.policy, 'mlp_extractor'):
        in_dim = model.policy.observation_space.shape[0]
        action_dim = model.policy.action_space.shape[0]
        
        # Replace policy network
        policy_net = TritonPolicyMLP(
            in_features=in_dim,
            out_features=action_dim,
            hidden_sizes=model.policy.net_arch['pi']
        )
        
        # Copy weights from existing network
        for i, (w_old, b_old) in enumerate(zip(
            model.policy.mlp_extractor.policy_net,
            model.policy.action_net
        )):
            if hasattr(w_old, 'weight'):
                getattr(policy_net, f'weight_{i}').data.copy_(w_old.weight.data)
                getattr(policy_net, f'bias_{i}').data.copy_(w_old.bias.data)
        
        # Replace in model
        model.policy.mlp_extractor.policy_net = policy_net
