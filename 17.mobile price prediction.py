import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score

# Load your dataset (replace with your actual dataset)
data = pd.read_csv(r"C:\Users\Teja\Desktop\ml\cv files\mobile_prices.csv")

# Define features (X) and target variable (y)
X = data.drop(columns=["price_range"])
y = data["price_range"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model (Random Forest in this example)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Now you can use the trained model to predict prices for new mobile phones
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')
