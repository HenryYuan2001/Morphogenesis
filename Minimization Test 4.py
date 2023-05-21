import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize

def computation(m):
    def f(x):
        sum = 0
        for i in range(m):
            sum += (x[i]**3)/3 - 2* x[i]**2 + 4 * x[i]
        return sum

    grad_f = grad(f)

    def equations(z_vals):
        return jnp.sum(jnp.abs(grad_f(z_vals)))

    x0 = jnp.zeros(m)
    result = minimize(equations, x0, method='BFGS')

    print('The minimum of the function is at:', result.x)

def main():
    n = 10
    computation(n)
if __name__ == "__main__":
    main()
