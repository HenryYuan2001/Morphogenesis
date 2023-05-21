import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize

f = lambda x: jnp.sum(x**2 + 3*x + 1)

dfdx = grad(f)

sq_grad = lambda x: jnp.sum(dfdx(x)**2)

x0 = jnp.array([0.0])

result = minimize(sq_grad, x0, method='BFGS')

print(result.x)


