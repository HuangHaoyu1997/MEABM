import numpy as np

def pareto(alpha, m, size=1):
    """Generate random samples from a Pareto distribution."""
    u = np.random.rand(size)  # Uniform random numbers
    samples = m / np.power(u, 1.0/alpha)
    return samples

# Example usage:
alpha = 3.0  # Shape parameter
m = 1.0  # Scale parameter
samples = pareto(alpha, m, size=10)
print(samples)


def beta_dist(size=1):
    alpha, beta = 1.5, 2
    s = np.random.beta(alpha, beta, size) * 168 * 500
    if s <= 500:
        s = [500]
    return s