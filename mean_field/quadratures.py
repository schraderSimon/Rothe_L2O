import numpy as np
from scipy.special import roots_legendre

def tanh_sinh_quadrature(a, b, n):
    """
    Compute quadrature points and weights for tanh-sinh quadrature on interval [a, b].

    Parameters:
        a (float): Lower limit of integration.
        b (float): Upper limit of integration.
        n (int): Number of nodes on one side of zero (total nodes will be 2n + 1).

    Returns:
        points (numpy.ndarray): Quadrature points in the interval [a, b].
        weights (numpy.ndarray): Corresponding weights.
    """
    # Step size
    h = 1.0 / n

    # Create array of k values (from -n to n)
    k = np.arange(-n, n + 1)

    # Compute t_k values
    t = h * k

    # Compute phi(t) = (π/2) * sinh(t)
    phi = (np.pi / 2) * np.sinh(t)

    # Compute x_k = tanh(φ(t))
    x = np.tanh(phi)

    # Map x from [-1, 1] to [a, b]
    points = (b - a) / 2 * x + (a + b) / 2

    # Compute derivative φ'(t) = (π/2) * cosh(t)
    dphi_dt = (np.pi / 2) * np.cosh(t)

    # Compute x'(t) = φ'(t) * sech^2(φ(t))
    dx_dt = dphi_dt * (1 / np.cosh(phi)) ** 2

    # Compute weights w_k = h * x'(t_k)
    weights = h * dx_dt * (b - a) / 2

    return points, weights
def gaussian_quadrature(a, b, n):
    x, w = roots_legendre(n)
    points = 0.5 * (b - a) * x + 0.5 * (b + a)
    weights = 0.5 * (b - a) * w   
    return points, weights
def trapezoidal_quadrature(a, b, n):
    # Generate equally spaced points in the interval [a, b]
    points = np.linspace(a, b, n)
    
    # Calculate the spacing between points
    dx = points[1] - points[0]
    
    # Calculate weights for the trapezoidal rule
    weights = np.full(n, dx)
    weights[0] *= 0.5  # Half weight for the first point
    weights[-1] *= 0.5  # Half weight for the last point
    
    return points, weights
def simpson_quadrature(a, b, n):
    # Simpson's rule requires an odd number of points
    if n % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points (n must be odd).")
    
    # Generate equally spaced points in the interval [a, b]
    points = np.linspace(a, b, n)
    
    # Calculate the spacing between points
    dx = (b - a) / (n - 1)
    
    # Calculate weights for Simpson's rule
    weights = np.zeros(n)
    weights[0] = weights[-1] = dx / 3  # First and last points
    weights[1:-1:2] = 4 * dx / 3  # Odd-indexed points
    weights[2:-2:2] = 2 * dx / 3  # Even-indexed points
    
    return points, weights
def clenshaw_curtis_quadrature(a, b, n):
    # Generate Chebyshev nodes in [-1, 1]
    k = np.arange(n)
    theta = np.pi * k / (n - 1)
    x = np.cos(theta)
    
    # Transform the nodes from [-1, 1] to [a, b]
    points = 0.5 * (b - a) * (x + 1) + a
    
    # Calculate weights for Clenshaw-Curtis quadrature
    weights = np.zeros(n)
    weights[0] = 1 / (n - 1)
    weights[-1] = weights[0]
    
    for j in range(1, n - 1):
        weights[j] = 2 * np.sum(1 - np.cos(j * theta)) / (n - 1)
    
    return points, weights
