import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix

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

# Display Encoded Table
print("Encoded Data:")
print(df)

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

# ==========================
# 📊 **VISUALIZATION**
# ==========================

# 1️⃣ **Class Distribution**
plt.figure(figsize=(6,4))
sns.countplot(x=df["Enrolls"], palette="pastel")
plt.title("Class Distribution of Enrollment")
plt.xticks(ticks=[0,1], labels=["No", "Yes"])
plt.xlabel("Enrolls")
plt.ylabel("Count")
plt.show()

# 2️⃣ **Confusion Matrix**
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 3️⃣ **Feature Importance (Naïve Bayes Log Probabilities)**
features = X.columns
importances = model.feature_log_prob_[1].sum(axis=0)  # Log Probabilities of class 1

plt.figure(figsize=(8,5))
sns.barplot(x=features, y=importances, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Feature Importance Based on Log Probabilities")
plt.xlabel("Features")
plt.ylabel("Log Probability Sum")
plt.show()
