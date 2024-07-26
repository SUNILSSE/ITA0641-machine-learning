import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Import Libraries

# Step 2: Create Sample DataFrame
data = pd.DataFrame({
    'Age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 37],
    'Income': [50000, 100000, 75000, 120000, 45000, 80000, 150000, 90000, 70000, 85000],
    'LoanAmount': [20000, 50000, 30000, 60000, 15000, 40000, 70000, 45000, 25000, 35000],
    'CreditScore': [600, 700, 650, 750, 580, 690, 800, 710, 640, 660],
    'Approved': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
})

# Step 3: Data Preprocessing
# Convert categorical columns to numerical (if any)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Define features and target variable
X = data.drop('Approved', axis=1)
y = data['Approved']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Naive Bayes Model
model = GaussianNB()
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test)

# Print accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Print confusion matrix
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# Print classification report
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
