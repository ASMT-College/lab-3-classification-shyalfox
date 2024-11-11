import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (same as above)
data = {
    'Pregnancies': [6, 1, 8, 1, 0, 5, 3, 10, 2, 8],
    'Glucose': [148, 85, 183, 89, 137, 116, 78, 115, 197, 125],
    'BloodPressure': [72, 66, 64, 66, 40, 74, 50, 0, 70, 96],
    'SkinThickness': [35, 29, 0, 23, 35, 0, 32, 0, 45, 0],
    'Insulin': [0, 0, 0, 94, 168, 0, 88, 0, 543, 0],
    'BMI': [33.6, 26.6, 23.3, 28.1, 43.1, 25.6, 31.0, 35.3, 30.5, 30.0],
    'DiabetesPedigreeFunction': [0.627, 0.351, 0.672, 0.167, 2.288, 0.201, 0.248, 0.134, 0.158, 0.232],
    'Age': [50, 31, 32, 21, 33, 30, 26, 29, 53, 54],
    'Outcome': [1, 0, 1, 0, 1, 0, 0, 0, 1, 1]
}

# Load the dataset
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop(columns='Outcome')
y = df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))
