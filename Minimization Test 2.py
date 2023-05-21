import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize

def f(x):
    x1, x2 = x
    return (x1**3)/3 - 2*x1**2 + 4*x1 + (x2**3)/3 - 3*x2**2 + 9*x2

grad_f = grad(f)
def equations(z_vals):
    return jnp.sum(jnp.abs(grad_f(z_vals)))

x0 = jnp.zeros(2)
result = minimize(equations, x0, method='BFGS')

print('The minimum of the function is at:', result.x)


