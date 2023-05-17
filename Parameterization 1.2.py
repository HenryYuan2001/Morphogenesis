import jax.numpy as jnp
from jax import grad, jit
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

def create_base_matrix(N):
    M = jnp.empty((N, 3))
    return M

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

def calculate_distances(mesh_coordinates_3d, adjacent_points):
    distances = {}

    for point, adjacent in adjacent_points.items():
        distances[point] = []
        for adj in adjacent:
            x1, y1, z1 = mesh_coordinates_3d[point]
            x2, y2, z2 = mesh_coordinates_3d[adj]
            distance = jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
            distances[point].append(distance)

    return distances

def spherical_para(R, mesh_coordinates_3d):
    x, y, _ = jnp.hsplit(mesh_coordinates_3d, 3)
    z = jnp.sqrt(R**2 - x**2 - y**2)
    z = z.ravel()  # Flatten the array

    G = create_base_matrix(len(mesh_coordinates_3d))
    G = G.at[:, :2].set(mesh_coordinates_3d[:, :2])
    G = G.at[:, 2].set(z)

    return G


def average_distances(distances):
    avg_distances = {}

    for point, dist_list in distances.items():
        avg_distances[point] = jnp.mean(jnp.array(dist_list))

    return avg_distances

def distance_3d(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

def potential_computation_numerical(G, adjacent_points, avg_distance):
    N = len(G)

    def F(z):
        total = 0.0
        for i in range(N):
            p1 = (G[i, 0], G[i, 1], z[i])
            for j in adjacent_points[i]:
                p2 = (G[j, 0], G[j, 1], z[j])
                total += (distance_3d(p1, p2))**2 - (avg_distance[i])**2
        return total

    dF_dz = jit(grad(F))

    # Initial guess for the z-values
    z_guess = jnp.zeros(N)

    # Define a function to pass to minimize
    def equations(z_vals):
        return jnp.sum(jnp.abs(dF_dz(z_vals)))

    # Solve the system of PDEs numerically
    res = minimize(equations, z_guess, method='BFGS')
    z_solutions = res.x

    # Update the z-values in the mesh
    G = G.at[:, 2].set(z_solutions)

    return G

def spher_potential_computation_numerical(G, G2, adjacent_points):
    N = len(G)

    def F(z):
        total = 0.0
        for i in range(N):
            p1 = (G[i, 0], G[i, 1], z[i])
            p3 = (G2[i, 0], G2[i, 1], G[i, 2])
            for j in adjacent_points[i]:
                p2 = (G[j, 0], G[j, 1], z[j])
                p4 = (G2[j, 0], G2[j, 1], G[j, 2])
                total += (distance_3d(p1, p2))**2- (distance_3d(p3, p4)**2)**2
        return total

    dF_dz = jit(grad(F))
    # Initial guess for the z-values
    z_guess = jnp.zeros(N)

    def equations(z_vals):
        return jnp.sum(jnp.abs(dF_dz(z_vals)))
    # Solve the system of PDEs numerically
    res = minimize(equations, z_guess, method='BFGS')
    z_solutions = res.x

    # Update the z-values in the mesh
    G = G.at[:, 2].set(z_solutions)

    return G



def plot_3d_mesh(mesh, mesh_coordinates_3d, adjacent_points, distances):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(mesh.num_vertices()):
        x, y, z = mesh_coordinates_3d[i]
        ax.text(x, y, z, str(i), ha="center", va="center")
        print("Point ", i, ": (", x, ", ", y, ", ", z, ")")
        print("Distances to adjacent points: ", distances[i])

    for point, adjacent in adjacent_points.items():
        for adj in adjacent:
            x_values = [mesh_coordinates_3d[point][0], mesh_coordinates_3d[adj][0]]
            y_values = [mesh_coordinates_3d[point][1], mesh_coordinates_3d[adj][1]]
            z_values = [mesh_coordinates_3d[point][2], mesh_coordinates_3d[adj][2]]
            ax.plot(x_values, y_values, z_values, c='b')

    sc = ax.scatter(mesh_coordinates_3d[:, 0], mesh_coordinates_3d[:, 1], mesh_coordinates_3d[:, 2], c=mesh_coordinates_3d[:, 2])

    plt.title("Mesh of a Circle in 3D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    fig.colorbar(sc, ax=ax, label='z-coordinate')

    plt.show()

def main():
    R = 1
    N = 5
    mesh, mesh_coordinates_3d = generate_3d_mesh(R, N)

    G1 = z_free_matrix(mesh_coordinates_3d)
    adjacent_points = calculate_adjacent_points(mesh)
    distances = calculate_distances(mesh_coordinates_3d, adjacent_points)
    avg_distance = average_distances(distances)

    G2 = potential_computation_numerical(G1, adjacent_points, avg_distance)
    distances = calculate_distances(mesh_coordinates_3d, adjacent_points)

    G3 = spherical_para(R,G1)
    distance_sphere = calculate_distances(G3, adjacent_points)
    #G4 = potential_computation_numerical(G1, adjacent_points, distance_sphere)
    plot_3d_mesh(mesh, mesh_coordinates_3d, adjacent_points,distances)
    plot_3d_mesh(mesh, G2, adjacent_points, avg_distance)
    plot_3d_mesh(mesh, G3, adjacent_points, distance_sphere)

if __name__ == "__main__":
    main()
