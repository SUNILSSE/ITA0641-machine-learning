import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample data: Replace this with your actual dataset
data = {
    'Income': [50000, 60000, 70000, 80000, 90000, 100000],
    'Age': [25, 35, 45, 55, 65, 75],
    'Loan Amount': [2000, 3000, 4000, 5000, 6000, 7000],
    'Credit Score': [1, 0, 1, 0, 1, 0]  # 1: Good, 0: Bad
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df[['Income', 'Age', 'Loan Amount']]
y = df['Credit Score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
