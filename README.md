# PSO Investigations

## Overview
PSO Investigations is a research-driven exploration into various implementations and optimizations of Particle Swarm Optimization (PSO) algorithms. This repository serves as a technical portfolio showcasing advanced PSO variants, problem-specific adaptations, and comparative analyses against other optimization techniques.

The project investigates:
- Standard PSO and its limitations
- Hybrid and modified PSO versions
- PSO's effectiveness across different optimization landscapes
- Comparative benchmarking against alternative metaheuristic algorithms

## Project Goals
- Implement and analyze different PSO variations, including inertia weight, constriction factor, and adaptive PSO.
- Apply PSO to real-world optimization problems, such as function minimization, engineering design, and hyperparameter tuning.
- Evaluate the performance of different PSO strategies through empirical analysis and benchmarking.
- Compare PSO with alternative algorithms such as Genetic Algorithms, Differential Evolution, and Bayesian Optimization.

## Implemented Variants
- **Standard PSO**: Canonical implementation based on the original formulation.
- **Inertia Weight PSO**: Introduces a dynamic inertia parameter to balance exploration and exploitation.
- **Constriction Factor PSO**: Implements a mathematically derived constriction coefficient for stability.
- **Adaptive PSO**: Adjusts parameters dynamically based on swarm behavior.
- **Hybrid PSO**: Integrates PSO with other metaheuristics for enhanced optimization.

## Methodology
- Algorithmic implementations are tested using benchmark functions such as Sphere, Rastrigin, Rosenbrock, and Ackley.
- Empirical evaluation is conducted using performance metrics such as convergence speed, solution quality, and computational efficiency.
- Comparative studies analyze PSO’s behavior in different optimization landscapes.
- Code is written in Python with libraries such as NumPy, SciPy, and Matplotlib for analysis and visualization.

## Usage
To run the PSO experiments, clone the repository and execute the relevant scripts.

```sh
git clone https://github.com/semajrd1/PSO-Investigations.git
cd PSO-Investigations
python run_experiment.py
