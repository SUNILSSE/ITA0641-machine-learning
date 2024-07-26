import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode target labels to one-hot encoding
num_classes = len(np.unique(y))
y_one_hot = np.zeros((len(y), num_classes))
y_one_hot[np.arange(len(y)), y] = 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)


# Define activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)
input_dim = X.shape[1]
hidden_dim = 8
output_dim = num_classes
learning_rate = 0.1
epochs = 1000

# Initialize weights and biases
np.random.seed(0)
weights_input_hidden = np.random.randn(input_dim, hidden_dim)
bias_hidden = np.zeros((1, hidden_dim))
weights_hidden_output = np.random.randn(hidden_dim, output_dim)
bias_output = np.zeros((1, output_dim))

# Train the neural network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X_train, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y_train - output_layer_output
    d_output = error * sigmoid_derivative(output_layer_output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate

    weights_input_hidden += X_train.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Predictions on test set
hidden_layer_input = np.dot(X_test, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output_layer_output = sigmoid(output_layer_input)

# Convert probabilities to class labels
y_pred = np.argmax(output_layer_output, axis=1)

# Calculate accuracy
accuracy = accuracy_score(np.argmax(y_test, axis=1), y_pred)
print("Accuracy:", accuracy)
# Sample data for prediction
new_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sepal Length, Sepal Width, Petal Length, Petal Width

# Standardize the new sample using the previously fitted scaler
new_sample_scaled = scaler.transform(new_sample)

# Perform forward propagation to predict the class probabilities
hidden_layer_input = np.dot(new_sample_scaled, weights_input_hidden) + bias_hidden
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
output_layer_output = sigmoid(output_layer_input)

# Convert probabilities to class label
predicted_class = np.argmax(output_layer_output, axis=1)

# Print the predicted class
print("Predicted class for the new sample:", iris.target_names[predicted_class])
