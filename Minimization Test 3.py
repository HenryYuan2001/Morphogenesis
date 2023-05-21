import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize

def f(x):
    x1, x2, x3 = x
    return x1**2 + x2**2 + x3 ** 2 + x1 + 2*x2+ 3*x3 + x1*x2 + x2*x3

grad_f = grad(f)
def equations(z_vals):
    return jnp.sum(jnp.abs(grad_f(z_vals)))

x0 = jnp.zeros(3)
result = minimize(equations, x0, method='BFGS')

print('The minimum of the function is at:', result.x)

