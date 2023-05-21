import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from fenics import Point
from mshr import generate_mesh, Circle
from dolfin import cells
from jax.scipy.optimize import minimize


def generate_3d_mesh(R, N):
    domain = Circle(Point(0., 0.), R)
    mesh = generate_mesh(domain, N)

    mesh_coordinates = jnp.array(mesh.coordinates())
    z_noise = 0.01 * np.random.normal(loc=0.0, scale=0.1, size=mesh_coordinates.shape[0])
    mesh_coordinates_3d = jnp.insert(mesh_coordinates, 2, z_noise, axis=1)

    return mesh, mesh_coordinates_3d

def spherical_para(R, mesh_coordinates_3d):
    x, y, _ = jnp.hsplit(mesh_coordinates_3d, 3)
    z = jnp.sqrt(R**2 - x**2 - y**2)
    z = jnp.where(jnp.isnan(z), 0, z)  
    z = z.ravel()

    G = create_base_matrix(len(mesh_coordinates_3d))
    G = G.at[:, :2].set(mesh_coordinates_3d[:, :2])
    G = G.at[:, 2].set(z)

    return G

def calculate_adjacent_points(mesh):
    adjacent_points = {}

    for cell in cells(mesh):
        vertices = cell.entities(0)
        for i in vertices:
            if i not in adjacent_points:
                adjacent_points[i] = []
            for j in vertices:
                if i != j and j not in adjacent_points[i]:
                    adjacent_points[i].append(j)
    return adjacent_points

def z_free_matrix(mesh_coordinates_3d):
    N = len(mesh_coordinates_3d)
    G = create_base_matrix(N)
    G = G.at[:, :2].set(mesh_coordinates_3d[:, :2])
    return G

def create_base_matrix(N):
    M = jnp.empty((N, 3))
    return M

def distance_3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def spher_potential_computation_numerical(G, G2):
    N = len(G)

    def F(z):
        total = 0.0
        for i in range(N):
            total += (z[i]-G2[i, 2])**2
        return total

    z_guess = jnp.ones(N)

    result = minimize(F, z_guess, method='BFGS')
    z_solutions = result.x

    G = G.at[:, 2].set(z_solutions)

    return G

def plot_3d(coordinates):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = coordinates[:, 0]
    ys = coordinates[:, 1]
    zs = coordinates[:, 2]
    ax.scatter(xs, ys, zs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

R = 1
N = 5
mesh, mesh_coordinates_3d = generate_3d_mesh(R, N)
G1 = z_free_matrix(mesh_coordinates_3d)
print(G1)
adjacent_points = calculate_adjacent_points(mesh)
G2 = spherical_para(R,G1)
print(G2)
G3 = spher_potential_computation_numerical(G1, G2)
print(G3)
plot_3d(G2)
plot_3d(G3)