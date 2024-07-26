import numpy as np

def perceptron(x1, x2):
    w0 = 2
    w1 = 0.1
    w2 = -1
    value = w0 + w1 * x1 + w2 * x2
    if value > 0:
        return "Setosa"
    else:
        return "Versicolor"

# Example usage:
predicted_species = perceptron(5.1, 1.4)
print(f"Predicted species: {predicted_species}")
