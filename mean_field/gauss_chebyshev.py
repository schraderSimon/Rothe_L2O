import numpy as np
import matplotlib.pyplot as plt

# Define the Gauss-Chebyshev quadrature points and weights on [0, R]
def gauss_chebyshev_points_weights(n, R):
    # Chebyshev nodes and weights for [0, π]
    nodes = np.cos((2 * np.arange(1, n + 1) - 1) * np.pi / (2 * n))
    weights = np.pi / n * np.ones(n)
    
    # Transform nodes and weights to [0, R]
    transformed_nodes = (nodes + 1) * R / 2
    transformed_weights = weights * R / 2
    return transformed_nodes, transformed_weights

# Parameters
n = 20
R = 10

# Calculate points and weights
nodes, weights = gauss_chebyshev_points_weights(n, R)

# Visualization
plt.figure(figsize=(8, 5))
plt.scatter(nodes, np.zeros_like(nodes), color='red', label='Quadrature Nodes')
plt.vlines(nodes, 0, weights, color='blue', alpha=0.5, label='Weights')
plt.xlabel('x')
plt.ylabel('Weight (scaled)')
plt.title(f'Gauss-Chebyshev Quadrature Points and Weights on [0, {R}] (n={n})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
