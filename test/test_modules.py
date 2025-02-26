import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as torch_nn
import pytest
from typing import Tuple, Dict, Any, Optional, List, Callable
import jaxtorch
from jaxtorch import nn, Context, init, PRNG

# Set seed for reproducibility
SEED = 42

# Tolerance for numerical comparisons
RTOL = 1e-5
ATOL = 1e-5


class TestBase:
    """Base class with helper methods for testing."""
    
    def setup_method(self, method):
        """Initialize test resources before each test method is run."""
        # Initialize RNGs
        self.jax_rng = PRNG(jax.random.PRNGKey(SEED))
        torch.manual_seed(SEED)
    
    def check_forward(self, jax_module: jaxtorch.Module, torch_module: torch.nn.Module, 
                      jax_input: jax.Array, mode: str = "train", 
                      rtol: float = RTOL, atol: float = ATOL):
        """Test that forward passes produce equivalent results."""
        # Initialize jaxtorch module
        params = jax_module.initialize(self.jax_rng.split())
        
        # Transfer parameters from torch to jaxtorch
        self._transfer_params_torch_to_jax(torch_module, jax_module, params)
        
        # Convert jax input to torch
        torch_input = torch.tensor(np.array(jax_input))
        
        # Run forward passes
        cx = Context(params, self.jax_rng.split(), mode=mode)
        jax_output = jax_module(cx, jax_input)
        
        # Set torch module to appropriate mode
        if mode == "train":
            torch_module.train()
        else:
            torch_module.eval()
            
        torch_output = torch_module(torch_input)
        
        # Compare outputs
        np.testing.assert_allclose(
            np.array(jax_output),
            torch_output.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
    
    def check_init(self, jax_module: jaxtorch.Module, torch_module: torch.nn.Module):
        """Test that initialization produces statistically similar distributions."""
        # Initialize jaxtorch module
        params = jax_module.initialize(self.jax_rng.split())
        
        # Compare parameter stats
        self._compare_param_stats(jax_module, torch_module, params)
    
    def _transfer_params_torch_to_jax(self, torch_module, jax_module, jax_params):
        """Transfer parameters from torch module to jaxtorch params."""
        jax_params_dict = {}
        for jax_name, jax_param in jax_module.named_parameters():
            jax_params_dict[jax_name] = jax_param
        
        for torch_name, torch_param in torch_module.named_parameters():
            # Try different name formats
            candidates = [
                torch_name,
                torch_name.replace(".", "_"),
                # Add more transformations if needed
            ]
            
            for candidate in candidates:
                if candidate in jax_params_dict:
                    jax_param = jax_params_dict[candidate]
                    jax_params[jax_param.name] = jnp.array(torch_param.detach().cpu().numpy())
                    break
    
    def _compare_param_stats(self, jax_module, torch_module, jax_params):
        """Compare statistics of parameters between jaxtorch and torch modules."""
        jax_params_dict = {}
        for jax_name, jax_param in jax_module.named_parameters():
            jax_value = jax_params[jax_param.name]
            jax_stats = self._compute_stats(jax_value)
            jax_params_dict[jax_name] = (jax_param, jax_stats)
        
        for torch_name, torch_param in torch_module.named_parameters():
            # Try different name formats
            candidates = [
                torch_name,
                torch_name.replace(".", "_"),
                # Add more transformations if needed
            ]
            
            for candidate in candidates:
                if candidate in jax_params_dict:
                    jax_param, jax_stats = jax_params_dict[candidate]
                    torch_stats = self._compute_stats(torch_param)
                    
                    # Compare statistics (allowing for small differences)
                    assert abs(jax_stats["mean"] - torch_stats["mean"]) < 0.1, \
                        f"Mean differs too much for {candidate}: {jax_stats['mean']} vs {torch_stats['mean']}"
                    assert abs(jax_stats["std"] - torch_stats["std"]) < 0.1, \
                        f"Std differs too much for {candidate}: {jax_stats['std']} vs {torch_stats['std']}"
                    break
    
    def _compute_stats(self, tensor: jax.Array | torch.Tensor) -> Dict[str, float]:
        """Compute basic statistics for a tensor."""
        if isinstance(tensor, jax.Array):
            tensor = np.array(tensor)
        elif isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        return {
            "mean": float(np.mean(tensor)),
            "std": float(np.std(tensor)),
            "min": float(np.min(tensor)),
            "max": float(np.max(tensor)),
        }


class TestLinear(TestBase):
    @pytest.mark.parametrize("in_features,out_features,bias", [
        (10, 20, True),   # Standard case
        (10, 20, False),  # No bias
    ])
    def test_linear(self, in_features, out_features, bias):
        jax_linear = nn.Linear(in_features, out_features, bias=bias)
        torch_linear = torch_nn.Linear(in_features, out_features, bias=bias)
        
        # Test initialization
        self.check_init(jax_linear, torch_linear)
        
        # Test forward pass
        batch_size = 5
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, in_features])
        self.check_forward(jax_linear, torch_linear, input_data)


class TestConv2d(TestBase):
    @pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding,dilation,groups", [
        (3, 6, 3, 1, 1, 1, 1),      # Standard case
        (3, 6, 3, 2, 1, 1, 1),      # With stride
        (3, 6, 3, 1, 0, 1, 1),      # No padding
        (3, 6, 3, 1, 1, 2, 1),      # With dilation
        (6, 6, 3, 1, 1, 1, 2),      # With groups
        (3, 6, (3, 5), 1, 1, 1, 1), # Rectangular kernel
    ])
    def test_conv2d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        jax_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
        torch_conv = torch_nn.Conv2d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        # Test initialization
        self.check_init(jax_conv, torch_conv)
        
        # Test forward pass
        batch_size, height, width = 2, 16, 16
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, in_channels, height, width])
        self.check_forward(jax_conv, torch_conv, input_data)


class TestLayerNorm(TestBase):
    @pytest.mark.parametrize("normalized_shape,eps,elementwise_affine", [
        ((20,), 1e-5, True),         # Standard case, 1D
        ((20,), 1e-5, False),        # Without affine parameters
        ((20,), 1e-1, True),         # Larger epsilon
        ((10, 20), 1e-5, True),      # 2D shape
        ((5, 10, 20), 1e-5, True),   # 3D shape
    ])
    def test_layer_norm(self, normalized_shape, eps, elementwise_affine):
        jax_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        torch_norm = torch_nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.check_init(jax_norm, torch_norm)
        batch_size = 5
        input_shape = [batch_size, *normalized_shape] if len(normalized_shape) > 1 else [batch_size, normalized_shape[0]]
        input_data = jax.random.normal(self.jax_rng.split(), input_shape)
        self.check_forward(jax_norm, torch_norm, input_data)

class TestDropout(TestBase):
    @pytest.mark.parametrize("p,eval_mode", [
        (0.2, False),     # Train mode, 20% dropout
        (0.5, False),     # Train mode, 50% dropout
        (0.8, False),     # Train mode, 80% dropout
        (0.5, True),      # Eval mode, should not apply dropout
    ])
    def test_dropout(self, p, eval_mode):
        jax_dropout = nn.Dropout(p)
        torch_dropout = torch_nn.Dropout(p)
        
        # Test forward pass with stochastic behavior
        batch_size, dim = 1000, 1000  # Large to get good statistics
        input_data = jnp.ones((batch_size, dim))
        
        # When testing in train mode, we can't directly compare outputs due to randomness
        # Instead, we'll verify the statistical properties
        if not eval_mode:
            # Initialize jaxtorch module
            params = jax_dropout.initialize(self.jax_rng.split())
            
            # Run in train mode
            cx = Context(params, self.jax_rng.split(), mode="train")
            jax_output = jax_dropout(cx, input_data)
            
            # Convert to numpy for analysis
            jax_output_np = np.array(jax_output)
            
            # Check that around p fraction are zeros
            zero_fraction = np.sum(jax_output_np == 0) / jax_output_np.size
            assert abs(zero_fraction - p) < 0.05, f"Expected ~{p} zeros, got {zero_fraction}"
            
            # Check that non-zero values are scaled appropriately
            non_zero_mean = np.mean(jax_output_np[jax_output_np != 0])
            expected_mean = 1.0 / (1.0 - p)
            assert abs(non_zero_mean - expected_mean) < 0.05, \
                f"Expected mean of ~{expected_mean}, got {non_zero_mean}"
        else:
            # In eval mode, we can directly compare since the output should be deterministic
            self.check_forward(jax_dropout, torch_dropout, input_data, mode="eval")


class TestGroupNorm(TestBase):
    @pytest.mark.parametrize("num_groups,num_channels,eps,affine", [
        (4, 16, 1e-5, True),      # Standard case
        (4, 16, 1e-5, False),     # Without affine parameters
        (1, 16, 1e-5, True),      # Single group (equivalent to LayerNorm)
        (16, 16, 1e-5, True),     # Each channel in its own group (similar to InstanceNorm)
        (4, 16, 1e-10, True),     # Smaller epsilon
    ])
    def test_group_norm(self, num_groups, num_channels, eps, affine):
        jax_norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        torch_norm = torch_nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        
        # Test initialization
        self.check_init(jax_norm, torch_norm)
        
        # Test forward pass
        batch_size, height, width = 2, 8, 8
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, num_channels, height, width])
        self.check_forward(jax_norm, torch_norm, input_data)


class TestEmbedding(TestBase):
    @pytest.mark.parametrize("num_embeddings,embedding_dim", [
        (100, 32),      # Standard case
        (10, 256),      # Small vocabulary, large embeddings
        (1000, 8),      # Large vocabulary, small embeddings
    ])
    def test_embedding(self, num_embeddings, embedding_dim):
        jax_embed = nn.Embedding(num_embeddings, embedding_dim)
        torch_embed = torch_nn.Embedding(num_embeddings, embedding_dim)
        
        # Test initialization
        self.check_init(jax_embed, torch_embed)
        
        # Test forward pass
        batch_size, seq_len = 8, 10
        input_data = jax.random.randint(self.jax_rng.split(), (batch_size, seq_len), 0, num_embeddings)
        self.check_forward(jax_embed, torch_embed, input_data)


class TestConv1d(TestBase):
    @pytest.mark.parametrize("in_channels,out_channels,kernel_size,stride,padding,dilation,groups", [
        (4, 8, 3, 1, 1, 1, 1),      # Standard case
        (4, 8, 3, 2, 1, 1, 1),      # With stride
        (4, 8, 3, 1, 0, 1, 1),      # No padding
        (4, 8, 3, 1, 1, 2, 1),      # With dilation
        (8, 8, 3, 1, 1, 1, 2),      # With groups
    ])
    def test_conv1d(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        jax_conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                          stride=stride, padding=padding, dilation=dilation, groups=groups)
        torch_conv = torch_nn.Conv1d(in_channels, out_channels, kernel_size, 
                                  stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        # Test initialization
        self.check_init(jax_conv, torch_conv)
        
        # Test forward pass
        batch_size, length = 3, 32
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, in_channels, length])
        self.check_forward(jax_conv, torch_conv, input_data)


class TestSimpleActivations(TestBase):
    """Test for activations with no parameters"""
    @pytest.mark.parametrize("activation_class,torch_class", [
        (nn.ReLU, torch_nn.ReLU),
        (nn.Tanh, torch_nn.Tanh),
        (nn.Sigmoid, torch_nn.Sigmoid),
        (nn.SiLU, torch_nn.SiLU),
        (nn.LeakyReLU, torch_nn.LeakyReLU),
    ])
    def test_activation(self, activation_class, torch_class):
        jax_act = activation_class()
        torch_act = torch_class()
        batch_size, dim = 5, 20
        input_data = jax.random.uniform(self.jax_rng.split(), [batch_size, dim], minval=-2.0, maxval=2.0)
        self.check_forward(jax_act, torch_act, input_data)

class TestGELU(TestBase):
    """Test for GELU with different approximation methods"""
    @pytest.mark.parametrize("approximate", [
        "none",
        "tanh",
        None,
    ])
    def test_gelu(self, approximate):
        jax_gelu = nn.GELU(approximate=approximate) if approximate else nn.GELU()
        torch_gelu = torch_nn.GELU(approximate=approximate) if approximate else torch_nn.GELU()
        batch_size, dim = 5, 20
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, dim])
        self.check_forward(jax_gelu, torch_gelu, input_data)


class TestSequential(TestBase):
    def test_sequential_basic(self):
        """Test a basic sequential with Linear + ReLU layers."""
        in_features, hidden_size, out_features = 10, 20, 5
        
        jax_seq = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_features)
        )
        
        torch_seq = torch_nn.Sequential(
            torch_nn.Linear(in_features, hidden_size),
            torch_nn.ReLU(),
            torch_nn.Linear(hidden_size, out_features)
        )
        
        # Test initialization
        self.check_init(jax_seq, torch_seq)
        
        # Test forward pass
        batch_size = 3
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, in_features])
        self.check_forward(jax_seq, torch_seq, input_data)
    
    def test_sequential_with_dropout(self):
        """Test sequential with dropout, ensuring it's properly handled in train/eval modes."""
        in_features, hidden_size, out_features = 10, 20, 5
        dropout_rate = 0.5
        
        jax_seq = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_features)
        )
        
        torch_seq = torch_nn.Sequential(
            torch_nn.Linear(in_features, hidden_size),
            torch_nn.ReLU(),
            torch_nn.Dropout(dropout_rate),
            torch_nn.Linear(hidden_size, out_features)
        )
        
        # Test initialization
        self.check_init(jax_seq, torch_seq)
        
        # Test forward pass in eval mode (deterministic)
        batch_size = 3
        input_data = jax.random.normal(self.jax_rng.split(), [batch_size, in_features])
        self.check_forward(jax_seq, torch_seq, input_data, mode="eval")


class TestModuleList(TestBase):
    def test_module_list_access(self):
        """Test ModuleList's indexing and iteration capabilities."""
        # Create ModuleList with several layers
        jax_module_list = nn.ModuleList(
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 40)
        )
        
        # Test indexing
        assert isinstance(jax_module_list[0], nn.Linear)
        assert jax_module_list[0].out_features == 20
        assert jax_module_list[1].out_features == 30
        assert jax_module_list[2].out_features == 40
        
        # Test iteration
        layers = [layer for layer in jax_module_list]
        assert len(layers) == 3
        assert layers[0].out_features == 20
        assert layers[1].out_features == 30
        assert layers[2].out_features == 40
        
        # Test slicing
        sliced = jax_module_list[1:3]
        assert len(sliced) == 2
        assert sliced[0].out_features == 30
        assert sliced[1].out_features == 40


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])