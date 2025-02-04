import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyswarm import pso

# Dynamic and Noisy Rastrigin Function
def dynamic_noisy_rastrigin(x, y, t, noise):
    a = 10
    c = 2 * np.pi * t
    noise_term = noise * np.random.randn()
    return a * 2 + x**2 - a * np.cos(c * x) + y**2 - a * np.cos(c * y) + noise_term

# PSO optimization function
def optimize_dynamic_rastrigin(t, noise):
    lb = [-5.12, -5.12]  # Lower bounds for variables
    ub = [5.12, 5.12]    # Upper bounds for variables

    objective_function = lambda x: dynamic_noisy_rastrigin(x[0], x[1], t, noise)
    best_position, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=50)

    return best_position

# Calculate the difference between PSO and optimal solutions
def calculate_difference(optimal, pso_solution):
    return np.linalg.norm(optimal - pso_solution)

# Visualize the optimization progress as subplots for the Dynamic and Noisy Rastrigin function
def visualize_optimization_progress_difference(noise_levels):
    num_iterations = 10

    fig, axs = plt.subplots(2, 5, figsize=(15, 6), subplot_kw={'projection': '3d'})
    fig.suptitle('Optimization Progress Over Time - Dynamic and Noisy Rastrigin Function')

    # Store the PSO solution of the first iteration as a reference
    optimal_solution = optimize_dynamic_rastrigin(0, noise_levels[0])

    for t, ax in zip(range(num_iterations), axs.flatten()):
        best_position = optimize_dynamic_rastrigin(t, noise_levels[0])

        x_vals = np.linspace(-5.12, 5.12, 100)
        y_vals = np.linspace(-5.12, 5.12, 100)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = dynamic_noisy_rastrigin(X, Y, t, noise_levels[0])

        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.scatter(best_position[0], best_position[1], dynamic_noisy_rastrigin(best_position[0], best_position[1], t, noise_levels[0]),
                   color='red', marker='o', s=100, label='Best Position')
        ax.set_title(f'Iteration {t+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Function Value')

        # Calculate and display the difference between PSO and optimal solutions
        difference = calculate_difference(optimal_solution, best_position)
        ax.text(3, -5, -5, f'Difference:\n{difference:.4f}', color='black', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Separate figure to show how noise impacts prediction accuracy
def visualize_noise_impact(noise_levels):
    # Store the PSO solution of the first iteration as a reference
    optimal_solution = optimize_dynamic_rastrigin(0, noise_levels[0])

    plt.figure(figsize=(8, 6))
    plt.plot(noise_levels, [calculate_difference(optimal_solution, optimize_dynamic_rastrigin(0, noise)) for noise in noise_levels], marker='o')
    plt.title('Impact of Noise on Prediction Accuracy')
    plt.xlabel('Noise Level')
    plt.ylabel('Prediction Accuracy (Difference)')
    plt.show()

if __name__ == "__main__":
    # Define noise_levels here
    noise_levels = np.linspace(0.01,10,100)  # Different noise magnitudes to consider

    # Run the visualization functions
    visualize_optimization_progress_difference(noise_levels)
    visualize_noise_impact(noise_levels)
