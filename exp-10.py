import numpy as np
from scipy.stats import multivariate_normal

def initialize_parameters(X, k):
    n, d = X.shape
    weights = np.ones(k) / k
    means = X[np.random.choice(n, k, False)]
    covariances = np.array([np.eye(d)] * k)
    return weights, means, covariances

def e_step(X, weights, means, covariances):
    n, k = X.shape[0], len(weights)
    responsibilities = np.zeros((n, k))
    for i in range(k):
        responsibilities[:, i] = weights[i] * multivariate_normal.pdf(X, means[i], covariances[i])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def m_step(X, responsibilities):
    n, d = X.shape
    k = responsibilities.shape[1]
    Nk = responsibilities.sum(axis=0)
    weights = Nk / n
    means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]
    covariances = np.zeros((k, d, d))
    for i in range(k):
        diff = X - means[i]
        covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / Nk[i]
    return weights, means, covariances

def log_likelihood(X, weights, means, covariances):
    n, k = X.shape[0], len(weights)
    log_likelihood = 0
    for i in range(k):
        log_likelihood += weights[i] * multivariate_normal.pdf(X, means[i], covariances[i])
    return np.log(log_likelihood).sum()

def em_algorithm(X, k, max_iter=100, tol=1e-6):
    weights, means, covariances = initialize_parameters(X, k)
    log_likelihoods = []
    for _ in range(max_iter):
        responsibilities = e_step(X, weights, means, covariances)
        weights, means, covariances = m_step(X, responsibilities)
        log_likelihoods.append(log_likelihood(X, weights, means, covariances))
        if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            break
    return weights, means, covariances, log_likelihoods

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([np.random.multivariate_normal(mean, np.eye(2), 100) for mean in [(0, 0), (5, 5), (0, 5)]])
    k = 3
    weights, means, covariances, log_likelihoods = em_algorithm(X, k)
    print("Weights:\n", weights)
    print("Means:\n", means)
    print("Covariances:\n", covariances)
