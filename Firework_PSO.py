import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# import mplcursors
from matplotlib import rcParams, font_manager
import time
import random


font_manager.findSystemFonts()
rcParams['font.family'] = 'EB Garamond'

# Set style parameters
rcParams['axes.linewidth'] = 0.8
rcParams['lines.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8

# New grid settings
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.5
rcParams['grid.color'] = 'gray'
rcParams['grid.linestyle'] = '--'
rcParams['grid.linewidth'] = 0.5

def eggholder(x, y, noise_strength=100):
    noise_shape = x.shape
    noise = noise_strength * np.random.uniform(-1, 1, size=noise_shape)
    return - (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47)))) + noise

def explore_rotated_positions_avg(current_position, circle_size, search_range, angle_increment=120):
    best_fitness = eggholder(*current_position)

    rotated_positions = []
    rotated_fitness= []
    r1_position = None
    rotated_weight_values = []
    start_position = current_position
    temp = []
    new_position = [0, 0]

    for rotation_angle in range(0, 360, angle_increment):
        moved_position = [
            current_position[0] + np.cos(np.radians(rotation_angle)) * circle_size * random.random(),
            current_position[1] + np.sin(np.radians(rotation_angle)) * circle_size * random.random()
        ]
        moved_position = np.clip(moved_position, -search_range, search_range)
        # print(f"Before clip: {moved_position}")

        if r1_position is None:
            r1_position = moved_position

        rotated_positions.append(moved_position)

        # Evaluate at the moved position
        fitness_moved = eggholder(*moved_position)
        rotated_fitness.append(fitness_moved)

        # Update best rotated position
        # if fitness_moved < best_fitness:
        #     best_fitness = fitness_moved
        # print(f"After clip: {moved_position}")

    # Weighted sum calculation
    for val in rotated_fitness:
        rotated_weight_values.append(val/sum(rotated_fitness))
    # print(sum(rotated_weight_values))
        
    for i in range(len(rotated_positions)):
        temp.append(rotated_weight_values[i]*rotated_positions[i])
    # print(temp)

    for val in temp:
        new_position[0] += val[0]
        new_position[1] += val[1]

    new_position = np.clip(new_position, -search_range, search_range)
    # print(new_position)

    return r1_position, rotated_positions, new_position, start_position

def explore_rotated_positions_min(current_position, circle_size, angle_increment=90):
    new_position = current_position
    best_fitness = eggholder(*current_position)

    rotated_positions = []
    r1_position = None
    for rotation_angle in range(0, 360, angle_increment):
        moved_position = [
            current_position[0] + np.cos(np.radians(rotation_angle)) * circle_size,
            current_position[1] + np.sin(np.radians(rotation_angle)) * circle_size
        ]
        moved_position = np.clip(moved_position, -search_range, search_range)

        if r1_position is None:
            r1_position = moved_position

        rotated_positions.append(moved_position)

        # Evaluate at the moved position
        fitness_moved = eggholder(*moved_position)

        # Update best rotated position
        if fitness_moved < best_fitness:
            new_position = moved_position
            best_fitness = fitness_moved

    start_position = current_position

    return r1_position, rotated_positions, new_position, start_position

def particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight):
    particles = np.random.uniform(low=-search_range, high=search_range, size=(swarm_size, 2))
    velocities = np.zeros_like(particles)
    personal_new_positions = particles.copy()
    global_new_position = particles[np.argmin([eggholder(*p) for p in particles])]
    global_best_fitness = eggholder(*global_new_position)
    best_fitness_values = []
    new_positions = []
    cumulative_evaluations = []

    for iteration in range(iterations):
        for i in range(swarm_size):
            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_weight * np.random.rand() * (personal_new_positions[i] - particles[i]) +
                             social_weight * np.random.rand() * (global_new_position - particles[i]))
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], -search_range, search_range)

            # Adjust circle size based on iteration
            circle_size = 500 - (iteration / iterations) * (500 - 0)
            
            original_pos, rotated_pos, best_pos, middle_pos  = explore_rotated_positions_avg(particles[i], circle_size=circle_size, search_range=search_range)
            # print(best_pos)

            if  -512 > best_pos[0] < 512:
                print('code broken: out of bounds!')

            if  -512 > best_pos[1] < 512:
                print('code broken: out of bounds!')

            if eggholder(*best_pos) < eggholder(*personal_new_positions[i]):
                personal_new_positions[i] = best_pos
                # print(personal_new_positions[i])

            if eggholder(*best_pos) < global_best_fitness:
                global_new_position = best_pos
                global_best_fitness = eggholder(*best_pos)

        best_fitness_values.append(global_best_fitness)
        new_positions.append(global_new_position)
        # print(new_positions)
        cumulative_evaluations.append((iteration + 1) * swarm_size * (len(rotated_pos)+1))

        # if iteration == 50:
        #     random_particle_index = np.random.randint(0, swarm_size)
        #     random_particle = particles[random_particle_index]
        #     angle_for_position_rotations = 30

        #     fig_example = plt.figure(figsize=(8, 6))
        #     ax_example = fig_example.add_subplot(111)

        #     ax_example.scatter(random_particle[0], random_particle[1], color='blue', label='Particle Position')

        #     _, rotated_positions, _, _ = explore_rotated_positions(random_particle, circle_size=0.05, angle_increment=angle_for_position_rotations)

        #     rotated_positions = np.array(rotated_positions)
        #     ax_example.scatter(rotated_positions[:, 0], rotated_positions[:, 1], color='orange', label='Rotated Positions')

        #     ax_example.set_title('Example Particle and Rotated Positions')
        #     ax_example.set_xlabel('X')
        #     ax_example.set_ylabel('Y')
        #     ax_example.legend()

        #     plt.show()

    return new_positions, best_fitness_values, cumulative_evaluations

def normal_particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight):
    particles = np.random.uniform(low=-search_range, high=search_range, size=(swarm_size, 2))
    velocities = np.zeros_like(particles)
    personal_new_positions = particles.copy()
    global_new_position = particles[np.argmin([eggholder(*p) for p in particles])]
    global_best_fitness = eggholder(*global_new_position)
    best_fitness_values = []
    new_positions = []
    cumulative_evaluations = []

    for iteration in range(iterations):
        for i in range(swarm_size):
            velocities[i] = (inertia_weight * velocities[i] +
                             cognitive_weight * np.random.rand() * (personal_new_positions[i] - particles[i]) +
                             social_weight * np.random.rand() * (global_new_position - particles[i]))
            
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], -search_range, search_range)

            fitness_current = eggholder(*particles[i])
            fitness_personal_best = eggholder(*personal_new_positions[i])

            if fitness_current < fitness_personal_best:
                personal_new_positions[i] = particles[i]

            if fitness_current < global_best_fitness:
                global_new_position = particles[i]
                global_best_fitness = fitness_current

        best_fitness_values.append(global_best_fitness)
        new_positions.append(global_new_position)
        cumulative_evaluations.append((iteration + 1) * swarm_size)

    return new_positions, best_fitness_values, cumulative_evaluations

# Parameters
iterations = 200
swarm_size = 20
search_range = 512
inertia_weight = 0.5
cognitive_weight = 1.5
social_weight = 1.5

# Run the proposed PSO
new_positions_proposed, best_fitness_values_proposed, cumulative_evaluations_proposed = particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight)

# Run the normal PSO
new_positions_normal, best_fitness_values_normal, cumulative_evaluations_normal = normal_particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight)

# Specify the width ratios for the subplots
width_ratios = [3, 2, 2]

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': width_ratios}, figsize=(15, 6))
ax2.axis('off')
ax3.axis('off')

ax1.plot(cumulative_evaluations_proposed, best_fitness_values_proposed, label='Proposed PSO', color='blue')
ax1.plot(cumulative_evaluations_normal, best_fitness_values_normal, label='Normal PSO', color='red')
ax1.set_xlabel('Cumulative Objective Function Evaluations')
ax1.set_ylabel('Best Fitness Value')
ax1.set_title('Proposed PSO vs Normal PSO')
ax1.legend()

x = np.linspace(-512, 512, 500)
y = np.linspace(-512, 512, 500)
X, Y = np.meshgrid(x, y)
Z = eggholder(X, Y)

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='plasma', alpha=0.3, label='eggholder Function')

actual_optimum = np.array([512, 404.2319])
actual_optimum_value = eggholder(*actual_optimum)
ax2.scatter(actual_optimum[0], actual_optimum[1], actual_optimum_value, color='green', alpha=.4, s=100, label='Actual Optimum')
ax2.scatter(new_positions_proposed[-1][0], new_positions_proposed[-1][1], best_fitness_values_proposed[-1], color='blue', s=40, label='Proposed PSO Optimum')

new_positions_proposed = np.array(new_positions_proposed)
ax2.plot(new_positions_proposed[:, 0], new_positions_proposed[:, 1], best_fitness_values_proposed, color='blue', label='Proposed PSO Position Progression')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Eggholder Function Optimization \n with Proposed PSO', y=1.02)
ax2.legend()
ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4))

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='plasma', alpha=0.3, label='eggholder Function')

actual_optimum = np.array([512, 404.2319])
actual_optimum_value = eggholder(*actual_optimum)
ax3.scatter(actual_optimum[0], actual_optimum[1], actual_optimum_value, color='green', alpha=.4, s=100, label='Actual Optimum')
ax3.scatter(new_positions_normal[-1][0], new_positions_normal[-1][1], best_fitness_values_normal[-1], color='blue', s=40, label='Normal PSO Optimum')

new_positions_normal = np.array(new_positions_normal)
ax3.plot(new_positions_normal[:, 0], new_positions_normal[:, 1], best_fitness_values_normal, color='blue', label='Normal PSO Position Progression')

ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Eggholder Function Optimization \n with Normal PSO', y=1.02)
ax3.legend()
ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4))

# Add mplcursors to make legends draggable
# mplcursors.cursor(hover=True)

# Add figure captions at the bottom of the subplots
# fig.text(0.5, 0.02, 'X', ha='center')
fig.text(0.08, 0.5, 'Y', va='center', rotation='vertical')
fig.text(0.92, 0.5, 'Z', va='center', rotation='vertical')

plt.show()

# # Initialize variables to track points for each PSO and total evaluations
# points_proposed = 0
# points_normal = 0
# total_evaluations_proposed = 0
# total_evaluations_normal = 0

# # Run PSO 20 times for each
# for run in range(20):
#     # Run the proposed PSO
#     new_positions_proposed, best_fitness_values_proposed, cumulative_evaluations_proposed = particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight)
#     # print(cumulative_evaluations_proposed)

#     # Run the normal PSO
#     new_positions_normal, best_fitness_values_normal, cumulative_evaluations_normal = normal_particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight)
#     # print(cumulative_evaluations_normal)

#     # Update total evaluations
#     total_evaluations_proposed += cumulative_evaluations_proposed[-1]
#     total_evaluations_normal += cumulative_evaluations_normal[-1]

#     # Check which PSO performs better in this run and update points
#     if best_fitness_values_proposed[-1] < best_fitness_values_normal[-1]:
#         points_proposed += 1
#     elif best_fitness_values_proposed[-1] > best_fitness_values_normal[-1]:
#         points_normal += 1

#     print('Firework PSO score:',points_proposed)
#     print('Normal PSO score:',points_normal)

# # Normalize points based on total evaluations
# normalized_points_proposed = points_proposed / total_evaluations_proposed
# normalized_points_normal = points_normal / total_evaluations_normal

# # Print the normalized points for each PSO
# print(f"Proposed PSO Normalized Points: {normalized_points_proposed}")
# print(f"Normal PSO Normalized Points: {normalized_points_normal}")

# # Create a new figure for normalized points comparison
# fig_normalized_points_comparison, ax_normalized_points_comparison = plt.subplots()

# # Plot the normalized points for each PSO
# ax_normalized_points_comparison.bar(['Proposed PSO', 'Normal PSO'], [normalized_points_proposed, normalized_points_normal], color=['blue', 'red'])
# ax_normalized_points_comparison.set_ylabel('Normalized Points')
# ax_normalized_points_comparison.set_xlabel('PSO Type')
# ax_normalized_points_comparison.set_title('Normalized Points Comparison between Proposed PSO and Normal PSO')

# plt.show()

# Initialize variables to track points for each PSO and total evaluations
points_proposed = 0
points_normal = 0
total_evaluations_proposed = 0
total_evaluations_normal = 0

# Lists to store data for line graphs
evaluations_list_proposed = []
evaluations_list_normal = []
points_list_proposed = []
points_list_normal = []

# Run PSO 20 times for each
for run in range(200):
    # Run the proposed PSO
    new_positions_proposed, best_fitness_values_proposed, cumulative_evaluations_proposed = particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight)

    # Run the normal PSO
    new_positions_normal, best_fitness_values_normal, cumulative_evaluations_normal = normal_particle_swarm_optimization(iterations, swarm_size, search_range, inertia_weight, cognitive_weight, social_weight)

    # Update total evaluations
    total_evaluations_proposed += cumulative_evaluations_proposed[-1]
    total_evaluations_normal += cumulative_evaluations_normal[-1]

    # Check which PSO performs better in this run and update points
    if best_fitness_values_proposed[-1] < best_fitness_values_normal[-1]:
        points_proposed += 1
    elif best_fitness_values_proposed[-1] > best_fitness_values_normal[-1]:
        points_normal += 1

    # Append data for line graphs
    evaluations_list_proposed.append(total_evaluations_proposed)
    evaluations_list_normal.append(total_evaluations_normal)
    points_list_proposed.append(points_proposed)
    points_list_normal.append(points_normal)

    print('Firework PSO score:', points_proposed)
    print('Normal PSO score:', points_normal)

# Calculate the ratio of points to cumulative evaluations for each run
ratio_proposed = np.array(points_list_proposed) / np.array(evaluations_list_proposed)
ratio_normal = np.array(points_list_normal) / np.array(evaluations_list_normal)

# Create a new figure for the line graphs
fig_ratio_vs_run, ax_ratio_vs_run = plt.subplots()

# Plot the ratio for each PSO run
ax_ratio_vs_run.plot(range(1, 201), ratio_proposed, label='Proposed PSO', color='blue', marker='o')
ax_ratio_vs_run.plot(range(1, 201), ratio_normal, label='Normal PSO', color='red', marker='o')

ax_ratio_vs_run.set_xlabel('Run')
ax_ratio_vs_run.set_ylabel('Number of Points / Cumulative Evaluations')
ax_ratio_vs_run.set_title('Ratio of Points to Cumulative Evaluations vs Run')
ax_ratio_vs_run.legend()

plt.show()