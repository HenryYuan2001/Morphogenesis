import numpy as np
import matplotlib.pyplot as plt
import copy
from fenics import Point
from mshr import generate_mesh, Circle
from dolfin import cells
import random
import matplotlib.gridspec as gridspec

def generate_circle_mesh(R, resolution):
    domain = Circle(Point(0., 0.), R)
    mesh = generate_mesh(domain, resolution)
    mesh_coordinates = np.array(mesh.coordinates())
    return mesh, mesh_coordinates


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

def plot_mesh_and_adjacency(mesh_coordinates, adjacent_points):
    fig, ax = plt.subplots()

    ax.scatter(mesh_coordinates[:, 0], mesh_coordinates[:, 1])

    # plot lines connecting each point to its adjacent points
    for point, adjacents in adjacent_points.items():
        for adjacent in adjacents:
            ax.plot(*zip(mesh_coordinates[point], mesh_coordinates[adjacent]), color='blue')

    # label each point with its index
    for i, coords in enumerate(mesh_coordinates):
        ax.text(coords[0], coords[1], str(i), color="black", fontsize=12)

    ax.set_aspect('equal', 'box')
    plt.show()


def get_boundary_points(mesh_coordinates, R):
    tolerance = 1E-1
    boundary_points = []

    for i, coords in enumerate(mesh_coordinates):
        distance = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
        if abs(distance - R) < tolerance:
            boundary_points.append(i)
    return boundary_points

def division_selection(mesh_coordinates, adjacent_points, P, R, division_dict):
    selected_points = random.sample(range(len(mesh_coordinates)), P)
    boundary_points = get_boundary_points(mesh_coordinates, R)
    print(boundary_points)
    for N in selected_points:
        N_coords = mesh_coordinates[N]
        adjacent_N = adjacent_points[N]
        adjacent_coords_N = [mesh_coordinates[i] for i in adjacent_N]

        if N in boundary_points:
            if len(adjacent_N) % 2 == 0:
                boundary_even(N, N_coords, adjacent_coords_N, division_dict)
            else:
                boundary_odd(N, N_coords, adjacent_coords_N, division_dict)
        else:
            if len(adjacent_N) % 2 == 0:
                mesh_coordinates, adjacent_points = not_boundary_even(N, N_coords, adjacent_N, adjacent_coords_N, mesh_coordinates, adjacent_points, division_dict)
            else:
                mesh_coordinates, adjacent_points = not_boundary_odd(N, N_coords, adjacent_N, adjacent_coords_N, mesh_coordinates, adjacent_points, division_dict)
    print(f"new adjacent{adjacent_points}")
    return mesh_coordinates, adjacent_points
def boundary_even(point, point_coords, adjacent_coords, division_dict):
    print(f"Boundary even point: {point}, coordinates: {point_coords}, adjacent coordinates: {adjacent_coords}")

def boundary_odd(point, point_coords, adjacent_coords,division_dict):
    print(f"Boundary odd point: {point}, coordinates: {point_coords}, adjacent coordinates: {adjacent_coords}")


def not_boundary_even(N, adjacent_N, adjacent_coords_N, mesh_coordinates, adjacent_points, division_dict):
    # First chain
    chain1_points = [adjacent_N[1]]
    chain1_coords = [adjacent_coords_N[1]]

    n1 = adjacent_N[0]
    chain1_points.append(n1)
    chain1_coords.append(mesh_coordinates[n1])

    while len(chain1_points) < len(adjacent_N) // 2 :
        for n2 in adjacent_points[n1]:
            if n2 not in chain1_points and N in adjacent_points[n2]:
                chain1_points.append(n2)
                chain1_coords.append(mesh_coordinates[n2])
                n1 = n2
                break

    # Second chain
    chain2_points = [point for point in adjacent_N if point not in chain1_points]
    chain2_coords = [mesh_coordinates[point] for point in chain2_points]
    print(chain1_points)
    print(chain2_points)

    end_points_chain1 = []
    for i in chain1_points:
        count = 0
        M = adjacent_points[i]
        for j in chain1_points:
            if j in M:
                print(f"M={M}J={j}i={i}")
                count += 1
        if count == 1:
            end_points_chain1.append(i)

    print(end_points_chain1)

    N1, N2 = end_points_chain1

    end_points_chain2 = []
    for i in chain2_points:
        count = 0
        M = adjacent_points[i]
        for j in chain2_points:
            if j in M:
                print(f"M={M}J={j}i={i}")
                count += 1
        if count == 1:
            end_points_chain2.append(i)

    print(end_points_chain2)

    N3, N4 = end_points_chain2

    print(end_points_chain1)
    print(end_points_chain2)
    # Checking adjacency and appending points and their coordinates accordingly
    if N3 in adjacent_points[N1] and N4 in adjacent_points[N2]:
        chain1_points.append(N3)
        chain1_coords.append(mesh_coordinates[N3])
        chain2_points.append(N2)
        chain2_coords.append(mesh_coordinates[N2])
    elif N4 in adjacent_points[N1] and N3 in adjacent_points[N2]:
        chain1_points.append(N4)
        chain1_coords.append(mesh_coordinates[N4])
        chain2_points.append(N2)
        chain2_coords.append(mesh_coordinates[N2])

    # Create New1 and New2, add them to the list of points, and define their adjacent points
    New1_coords = np.mean(chain1_coords, axis=0)
    New2_coords = np.mean(chain2_coords, axis=0)

    New1_adjacents = chain1_points + [len(mesh_coordinates)]
    New2_adjacents = chain2_points + [N]

    # Replace N with New1 in the list of points and update its adjacent points
    mesh_coordinates[N] = New1_coords
    adjacent_points[N] = New1_adjacents


    # Add New2 to the list of points and define its adjacent points
    mesh_coordinates = mesh_coordinates.tolist()
    mesh_coordinates.append(New2_coords)
    mesh_coordinates = np.array(mesh_coordinates)
    adjacent_points[len(mesh_coordinates) - 1] = New2_adjacents

    # Find common adjacent points for New1 and New2
    common_adjacents = list(set(New1_adjacents) & set(New2_adjacents))

    # Update the adjacency list of common points
    for i in common_adjacents:
        if i != N:
            adjacent_points[i].append(len(mesh_coordinates) - 1)

    # Find points that are adjacent to New2 but not New1
    unique_adjacents_New2 = list(set(New2_adjacents) - set(common_adjacents) - {N})

    # Replace New1's index (which is N) with New2's index in the adjacency list of these points
    for j in unique_adjacents_New2:
        adjacent_points[j] = [len(mesh_coordinates)-1 if x==N else x for x in adjacent_points[j]]

    # Record the division in the division_dict
    division_dict[N] = [N, len(mesh_coordinates) - 1]
    print(f"Not boundary even point chain 1: {chain1_points}, chain coordinates: {chain1_coords}")
    print(f"Not boundary even point chain 2: {chain2_points}, chain coordinates: {chain2_coords}")
    print(f"New point 1 coordinates: {New1_coords}, New point 1 adjacents: {New1_adjacents}")
    print(f"New point 2 coordinates: {New2_coords}, New point 2 adjacents: {New2_adjacents}")
    print(f"Division: {N} -> {division_dict[N]}")
    return mesh_coordinates, adjacent_points
def not_boundary_odd(N, adjacent_N, adjacent_coords_N, mesh_coordinates, adjacent_points, division_dict):
    # First chain
    chain1_points = [adjacent_N[1]]
    chain1_coords = [adjacent_coords_N[1]]

    n1 = adjacent_N[0]
    chain1_points.append(n1)
    chain1_coords.append(mesh_coordinates[n1])

    while len(chain1_points) < len(adjacent_N) // 2 + 1:
        for n2 in adjacent_points[n1]:
            if n2 not in chain1_points and N in adjacent_points[n2]:
                chain1_points.append(n2)
                chain1_coords.append(mesh_coordinates[n2])
                n1 = n2
                break

    # Second chain
    chain2_points = [point for point in adjacent_N if point not in chain1_points]
    chain2_coords = [mesh_coordinates[point] for point in chain2_points]
    print(chain1_points)
    print(chain2_points)

    end_points_chain1 = []
    for i in chain1_points:
        count = 0
        M = adjacent_points[i]
        for j in chain1_points:
            if j in M:
                print(f"M={M}J={j}i={i}")
                count += 1
        if count == 1:
            end_points_chain1.append(i)

    print(end_points_chain1)

    N1, N2 = end_points_chain1

    end_points_chain2 = []
    for i in chain2_points:
        count = 0
        M = adjacent_points[i]
        for j in chain2_points:
            if j in M:
                print(f"M={M}J={j}i={i}")
                count += 1
        if count == 1:
            end_points_chain2.append(i)

    print(end_points_chain2)

    N3, N4 = end_points_chain2

    print(end_points_chain1)
    print(end_points_chain2)
    # Checking adjacency and appending points and their coordinates accordingly
    if N3 in adjacent_points[N1] and N4 in adjacent_points[N2]:
        chain1_points.append(N3)
        chain1_coords.append(mesh_coordinates[N3])
        chain2_points.append(N2)
        chain2_coords.append(mesh_coordinates[N2])
    elif N4 in adjacent_points[N1] and N3 in adjacent_points[N2]:
        chain1_points.append(N4)
        chain1_coords.append(mesh_coordinates[N4])
        chain2_points.append(N2)
        chain2_coords.append(mesh_coordinates[N2])

    # Create New1 and New2, add them to the list of points, and define their adjacent points
    New1_coords = np.mean(chain1_coords, axis=0)
    New2_coords = np.mean(chain2_coords, axis=0)

    New1_adjacents = chain1_points + [len(mesh_coordinates)]
    New2_adjacents = chain2_points + [N]

    # Replace N with New1 in the list of points and update its adjacent points
    mesh_coordinates[N] = New1_coords
    adjacent_points[N] = New1_adjacents

    # Add New2 to the list of points and define its adjacent points
    mesh_coordinates = mesh_coordinates.tolist()
    mesh_coordinates.append(New2_coords)
    mesh_coordinates = np.array(mesh_coordinates)
    adjacent_points[len(mesh_coordinates) - 1] = New2_adjacents

    # Find common adjacent points for New1 and New2
    common_adjacents = list(set(New1_adjacents) & set(New2_adjacents))

    # Update the adjacency list of common points
    for i in common_adjacents:
        if i != N:
            adjacent_points[i].append(len(mesh_coordinates) - 1)

    # Find points that are adjacent to New2 but not New1
    unique_adjacents_New2 = list(set(New2_adjacents) - set(common_adjacents) - {N})

    # Replace New1's index (which is N) with New2's index in the adjacency list of these points
    for j in unique_adjacents_New2:
        adjacent_points[j] = [len(mesh_coordinates) - 1 if x == N else x for x in adjacent_points[j]]

    division_dict[N] = [N, len(mesh_coordinates) - 1]

    print(f"Not boundary odd point chain 1: {chain1_points}, chain coordinates: {chain1_coords}")
    print(f"Not boundary odd point chain 2: {chain2_points}, chain coordinates: {chain2_coords}")
    print(f"New point 1 coordinates: {New1_coords}, New point 1 adjacents: {New1_adjacents}")
    print(f"New point 2 coordinates: {New2_coords}, New point 2 adjacents: {New2_adjacents}")
    print(f"Division: {N} -> {division_dict[N]}")
    return mesh_coordinates, adjacent_points


def plot_two_generations_and_dict(initial_mesh_coordinates, initial_adjacent_points,
                                  Gen1_mesh_coordinates, Gen1_adjacent_points, division_dict):
    gs = gridspec.GridSpec(1, 5)

    divided_points = [item for sublist in division_dict.values() for item in sublist]

    # Plot Initial Cells
    ax0 = plt.subplot(gs[0, :2])
    for point in range(len(initial_mesh_coordinates)):
        color = 'r' if point in division_dict else 'b'
        text_color = 'r' if point in division_dict else 'k'
        ax0.scatter(initial_mesh_coordinates[point, 0], initial_mesh_coordinates[point, 1], color=color)
        ax0.text(initial_mesh_coordinates[point, 0], initial_mesh_coordinates[point, 1], str(point), color=text_color,
                 fontsize=9)
        for adjacent in initial_adjacent_points[point]:
            ax0.plot(*zip(initial_mesh_coordinates[point], initial_mesh_coordinates[adjacent]), color='k',
                     linewidth=0.5)
    ax0.set_title('Initial Cells')

    # Plot First Generation
    ax1 = plt.subplot(gs[0, 2:4])
    for point in range(len(Gen1_mesh_coordinates)):
        color = 'r' if point in divided_points else 'b'
        text_color = 'r' if point in divided_points else 'k'
        ax1.scatter(Gen1_mesh_coordinates[point, 0], Gen1_mesh_coordinates[point, 1], color=color)
        ax1.text(Gen1_mesh_coordinates[point, 0], Gen1_mesh_coordinates[point, 1], str(point), color=text_color,
                 fontsize=9)
        for adjacent in Gen1_adjacent_points[point]:
            ax1.plot(*zip(Gen1_mesh_coordinates[point], Gen1_mesh_coordinates[adjacent]), color='k', linewidth=0.5)
    ax1.set_title('First Generation')

    ax2 = plt.subplot(gs[0, 4])
    ax2.axis('off')
    ax2.set_title('Division Dictionary')
    dict_text = "\n".join([f'{k} -> {v}' for k, v in division_dict.items()])
    ax2.text(0.1, 0.5, dict_text, fontsize=12)

    plt.tight_layout()
    plt.show()

def main():
    R = 1
    N = 5
    division_dict = {}
    mesh, mesh_coordinates = generate_circle_mesh(R, N)
    adjacent_points = calculate_adjacent_points(mesh)
    print(adjacent_points)
    plot_mesh_and_adjacency(mesh_coordinates, adjacent_points)
    initial_mesh_coordinates = copy.deepcopy(mesh_coordinates)
    initial_adjacent_points = copy.deepcopy(adjacent_points)
    boundary = get_boundary_points(mesh_coordinates, R)
    Gen1_mesh_coordinates, Gen1_adjacent_points = division_selection(mesh_coordinates, adjacent_points, 10, 1, division_dict)
    plot_mesh_and_adjacency(initial_mesh_coordinates, initial_adjacent_points)
    plot_mesh_and_adjacency(Gen1_mesh_coordinates, Gen1_adjacent_points)
    print(division_dict)
    print(adjacent_points)
    plot_two_generations_and_dict(initial_mesh_coordinates, initial_adjacent_points,
                                 Gen1_mesh_coordinates, Gen1_adjacent_points,
                                 division_dict)
if __name__ == "__main__":
    main()
