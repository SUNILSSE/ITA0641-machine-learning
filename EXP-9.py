# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston Housing dataset
boston = load_iris()
X = boston.data
y = boston.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions with the Linear Regression model
y_pred_lr = lr.predict(X_test)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Transform the features to polynomial features
poly_features = PolynomialFeatures(degree=2)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Initialize and train the Polynomial Regression model
poly_lr = LinearRegression()
poly_lr.fit(X_poly_train, y_train)

# Make predictions with the Polynomial Regression model
y_pred_poly = poly_lr.predict(X_poly_test)

# Evaluate the Polynomial Regression model
mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# Print results
print("Linear Regression")
print(f"Mean Squared Error: {mse_lr}")
print(f"R² Score: {r2_lr}\n")

print("Polynomial Regression")
print(f"Mean Squared Error: {mse_poly}")
print(f"R² Score: {r2_poly}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, color='blue', edgecolor='w', label='Linear Regression')
plt.scatter(y_test, y_pred_poly, color='green', edgecolor='w', label='Polynomial Regression (Degree=2)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2, linestyle='--', label='Ideal Fit')
plt.title('Linear vs Polynomial Regression')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
