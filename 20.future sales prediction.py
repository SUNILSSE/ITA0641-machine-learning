# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Read the CSV file
df = pd.read_csv(r"C:\Users\Teja\Desktop\ml\cv files\futuresale prediction.csv")

# Split the dataset into features (X) and target (y)
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a linear regression model
lr = LinearRegression()

# Train the model on the training set
lr.fit(X_train, y_train)

# Predict the sales for the test set
y_pred = lr.predict(X_test)
# Evaluate the model accuracy
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Use the trained model to predict future sales
new_data = pd.DataFrame([[500, 100, 50]], columns=X.columns)
prediction = lr.predict(new_data)
print("Predicted Sales:", prediction[0])
#TV	Radio	Newspaper	Sales
