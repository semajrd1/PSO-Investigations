import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyswarm import pso

# Noisy Egg Holder Function
def noisy_egg_holder(x, y, t):
    noise = 0 * np.random.randn()  # Adjust the noise magnitude as needed
    return -(y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47)))) + noise

# Analytical Minimum of the Noisy Egg Holder Function
absolute_minimum = np.array([512, 404.2319])  # Actual minimum values, adjust if needed

# PSO optimization function for Noisy Egg Holder
def optimize_noisy_egg_holder(t):
    lb = [-512, -512]  # Lower bounds for variables
    ub = [512, 512]    # Upper bounds for variables

    objective_function = lambda x: noisy_egg_holder(x[0], x[1], t)
    best_position, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=50)

    return best_position

# Calculate the difference between PSO and optimal solutions
def calculate_difference(optimal, pso_solution):
    return np.linalg.norm(optimal - pso_solution)

# Visualize the optimization progress for the Noisy Egg Holder function with separate figures for each iteration
def visualize_optimization_progress_noisy_egg_holder_separate_figures():
    num_iterations = 10

    for t in range(num_iterations):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        fig.suptitle(f'Optimization Progress - Iteration {t+1}')

        # Store the PSO solution of the current iteration as a reference
        optimal_solution = optimize_noisy_egg_holder(t)

        best_position = optimize_noisy_egg_holder(t)

        x_vals = np.linspace(-512, 512, 100)
        y_vals = np.linspace(-512, 512, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = noisy_egg_holder(X, Y, t)

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.scatter(best_position[0], best_position[1], noisy_egg_holder(best_position[0], best_position[1], t),
                   color='red', marker='o', s=100, label='PSO Predicted Minimum')
        ax.scatter(absolute_minimum[0], absolute_minimum[1], noisy_egg_holder(*absolute_minimum, t),
                   color='blue', marker='o', s=100, label='Absolute Minimum')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')

        # Calculate and display the difference between PSO and optimal solutions
        difference = calculate_difference(optimal_solution, best_position)
        ax.text(300, -300, -500, f'Difference:\n{difference:.4f}', color='black', fontsize=8)

        plt.legend()
        plt.show()

if __name__ == "__main__":
    visualize_optimization_progress_noisy_egg_holder_separate_figures()
