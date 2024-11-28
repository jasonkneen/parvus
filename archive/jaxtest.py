import jax
import jax.numpy as jnp

# Check available devices
print("Devices available:", jax.devices())

# Perform a simple computation
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
print("Sum:", a + b)