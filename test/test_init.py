import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.init as torch_init

from jaxtorch import PRNG, init

# Set seeds for reproducibility
SEED = 42

# Sample sizes for distribution tests
SHAPE = (1000, 1000)  # Large enough for good statistics

# Tolerance for statistical tests
STAT_RTOL = 0.1  # 10% relative tolerance for statistics


def test_zeros():
    """Test zeros initialization."""
    # Initialize jaxtorch
    jax_rng = PRNG(jax.random.PRNGKey(SEED))
    jax_param = init.zeros(*SHAPE)
    jax_value = jax_param.initializer(jax_rng.split())
    assert jnp.allclose(jax_value, 0.0)


def test_ones():
    """Test ones initialization."""
    # Initialize jaxtorch
    jax_rng = PRNG(jax.random.PRNGKey(SEED))
    jax_param = init.ones(*SHAPE)
    jax_value = jax_param.initializer(jax_rng.split())
    assert jnp.allclose(jax_value, 1.0)


def test_normal():
    """Test normal initialization."""
    stddev = 0.5
    jax_rng = PRNG(jax.random.PRNGKey(SEED))
    jax_param = init.normal(*SHAPE, stddev=stddev)
    jax_value = jax_param.initializer(jax_rng.split())
    jax_mean = float(jnp.mean(jax_value))
    jax_std = float(jnp.std(jax_value))
    expected_mean = 0.0
    expected_std = stddev
    assert abs(jax_mean - expected_mean) < 0.01
    assert abs(jax_std - expected_std) < STAT_RTOL * expected_std


def test_uniform():
    """Test uniform initialization."""
    min_val, max_val = -0.5, 0.5
    jax_rng = PRNG(jax.random.PRNGKey(SEED))
    jax_param = init.uniform(*SHAPE, min=min_val, max=max_val)
    jax_value = jax_param.initializer(jax_rng.split())
    assert float(jnp.min(jax_value)) >= min_val
    assert float(jnp.max(jax_value)) <= max_val
    jax_mean = float(jnp.mean(jax_value))
    expected_mean = (min_val + max_val) / 2
    expected_var = (max_val - min_val) ** 2 / 12
    abs_tol_mean = 0.01
    assert abs(jax_mean - expected_mean) < abs_tol_mean
    jax_var = float(jnp.var(jax_value))
    assert abs(jax_var - expected_var) < STAT_RTOL * expected_var


def test_glorot_normal():
    """Test glorot_normal initialization."""
    in_features, out_features = 100, 200
    shape = (out_features, in_features)
    jax_rng = PRNG(jax.random.PRNGKey(SEED))
    jax_param = init.glorot_normal(*shape)
    jax_value = jax_param.initializer(jax_rng.split())
    
    torch_tensor = torch.empty(*shape)
    torch_init.xavier_normal_(torch_tensor)
    
    fan_in, fan_out = in_features, out_features
    expected_std = np.sqrt(2.0 / (fan_in + fan_out))
    
    jax_std = float(jnp.std(jax_value))
    torch_std = float(torch.std(torch_tensor))
    
    assert abs(jax_std - expected_std) < STAT_RTOL * expected_std
    assert abs(torch_std - expected_std) < STAT_RTOL * expected_std


def test_kaiming_uniform():
    """Test kaiming_uniform initialization."""
    in_features, out_features = 100, 200
    shape = (out_features, in_features)
    
    jax_rng = PRNG(jax.random.PRNGKey(SEED))
    jax_param = init.kaiming_uniform(*shape)
    jax_value = jax_param.initializer(jax_rng.split())
    
    torch_tensor = torch.empty(*shape)
    torch_init.kaiming_uniform_(torch_tensor, a=0)
    
    fan_in = in_features
    gain = np.sqrt(2.0)  # For ReLU with a=0
    expected_bound = gain * np.sqrt(3.0 / fan_in)
    
    jax_min, jax_max = float(jnp.min(jax_value)), float(jnp.max(jax_value))
    torch_min, torch_max = float(torch.min(torch_tensor)), float(torch.max(torch_tensor))
    
    assert abs(jax_min + expected_bound) < STAT_RTOL * expected_bound
    assert abs(jax_max - expected_bound) < STAT_RTOL * expected_bound
    assert abs(torch_min + expected_bound) < STAT_RTOL * expected_bound
    assert abs(torch_max - expected_bound) < STAT_RTOL * expected_bound


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])