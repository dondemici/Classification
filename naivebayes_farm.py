import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

# Sample data
data = [
    ["<=30", "High", "No", "Fair", "No"],
    ["<=30", "High", "No", "Excellent", "No"],
    ["31 to 40", "High", "No", "Fair", "Yes"],
    [">40", "Medium", "No", "Fair", "Yes"],
    [">40", "Low", "Yes", "Fair", "Yes"],
    [">40", "Low", "Yes", "Excellent", "No"],
    ["31 to 40", "Low", "Yes", "Excellent", "Yes"],
    ["<=30", "Medium", "No", "Fair", "No"],
    ["<=30", "Low", "Yes", "Fair", "Yes"],
    [">40", "Medium", "Yes", "Fair", "Yes"],
    ["<=30", "Medium", "Yes", "Excellent", "Yes"],
    ["31 to 40", "Medium", "No", "Excellent", "Yes"],
    ["31 to 40", "High", "Yes", "Fair", "Yes"],
    [">40", "Medium", "No", "Excellent", "No"]
]

# Create DataFrame
df = pd.DataFrame(data, columns=["Age", "Income", "JobSatisfaction", "Desire", "Enrolls"])

# Encode categorical features
encoder = LabelEncoder()
for col in df.columns:
    df[col] = encoder.fit_transform(df[col])

# Split data into features and target
X = df.drop(columns=["Enrolls"])  # Features
y = df["Enrolls"]  # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Naïve Bayes model
model = CategoricalNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
