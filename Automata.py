import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def lorenz96(initial_state, nsteps, constants=(1/101, 100, 8)):
   

    x = np.array(initial_state, dtype=float)
    alpha, beta, gamma = constants
    N = len(x)
    for _ in range(nsteps):
        # Using np.roll for periodic boundary conditions
        x_ip1 = np.roll(x, -1)  # Shift left by 1 (i+1)
        x_im1 = np.roll(x, 1)   # Shift right by 1 (i-1)
        x_im2 = np.roll(x, 2)   # Shift right by 2 (i-2)
        x = alpha * (beta * x + (x_im2 - x_ip1) * x_im1 + gamma)

    return x

def life(initial_state, nsteps, periodic=False):
    grid = np.array(initial_state, dtype=bool)
    for step in range(nsteps):
        neighbours = count_neighbours(grid, periodic=periodic)

        new_grid = (((grid == True) & (neighbours > 1) & (neighbours < 4)) | 
                    ((grid == False) & (neighbours == 3))).astype(bool)
        print(f"Conway's Game of Life Grid at step {step + 1}:")
        print(new_grid)  # Print the grid
        grid = new_grid
    return grid

KERNEL = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

def count_neighbours(grid, periodic=False):
    mode = 'wrap' if periodic else 'constant'
    return convolve(grid.astype(int), KERNEL, mode=mode)

# Example initial state for Conway's Game of Life (using boolean grid)
initial_state_life = np.array([
    [False, True,  False, True, True],
    [False, False, True,  False, False],
    [True,  True,  True,  False, False],
    [False, False, False, False, False],
    [False, False, False, True, False]
])

def print_grid(grid):
    for row in grid:
        print("".join(['#' if cell else '.' for cell in row]))
final_state_life = life(initial_state_life, nsteps=1, periodic=True)

def plot_lorenz96(data, label=None, color=None):


    offset = 8
    data = np.asarray(data)
    theta = 2 * np.pi * np.arange(len(data)) / len(data)

    vector = np.empty((len(data), 2))
    vector[:, 0] = (data + offset) * np.sin(theta)
    vector[:, 1] = (data + offset) * np.cos(theta)

    theta = np.linspace(0, 2 * np.pi)

    rings = np.arange(int(np.floor(min(data))-1),
                      int(np.ceil(max(data))) + 2)
    for ring in rings:
        plt.plot((ring + offset) * np.cos(theta),
                 (ring + offset) * np.sin(theta), 'k:')

    fig_ax = plt.gca()
    fig_ax.spines['left'].set_position(('data', 0.0))
    fig_ax.spines['bottom'].set_position(('data', 0.0))
    fig_ax.spines['right'].set_color('none')
    fig_ax.spines['top'].set_color('none')
    plt.xticks([])
    plt.yticks(rings + offset, rings)
    plt.fill(vector[:, 0], vector[:, 1],
             label=label, fill=False)
    plt.scatter(vector[:, 0], vector[:, 1], 20)

def plot_array(data, show_axis=False, cmap='seismic', **kwargs):
    plt.pcolormesh(data[::-1, :].astype(int), edgecolor='y', vmin=-2, vmax=2, cmap=plt.get_cmap(cmap), **kwargs)
    plt.axis('equal')
    if show_axis:
        plt.axis('on')
    else:
        plt.axis('off')
    plt.show()

def simulate_and_plot_life(initial_state, nsteps, periodic=False):

    grid = np.array(initial_state, dtype=bool)
    
    for step in range(nsteps):
        print(f"Step {step + 1}:\n{grid}")
        plot_array(grid, show_axis=True, cmap='seismic')  # Plot the grid at each step
        grid = life(grid, nsteps=1, periodic=periodic)

def life_3d(initial_state, nsteps, periodic=False):
    kernel = np.ones((3, 3, 3), dtype=int)
    kernel[1, 1, 1] = 0 

    grid = np.array(initial_state, dtype=bool)

    for _ in range(nsteps):
        mode = 'wrap' if periodic else 'constant'
        neighbors = convolve(grid.astype(int), kernel, mode=mode)

        new_grid = np.zeros_like(grid, dtype=bool)
        new_grid[(grid == True) & ((neighbors == 5) | (neighbors == 6))] = True  # Survive
        new_grid[(grid == False) & (neighbors == 4)] = True

        grid = new_grid
    
    return grid

initial_state_3d = np.random.random((5, 5, 5)) > 0.7
final_state_3d = life_3d(initial_state_3d, nsteps=3, periodic=True)

print("Final state of 3D grid after 3 steps:")
print(final_state_3d.astype(int))
simulate_and_plot_life(initial_state_life, nsteps=10, periodic=True)
