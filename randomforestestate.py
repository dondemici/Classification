import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 📂 Load the dataset
file_path = r"C:\Users\PCAdmin\Downloads\Melbourne_RealEstate.csv"
df = pd.read_csv(file_path)

# 📌 Drop unnecessary columns
df = df.drop(columns=['Address', 'Date', 'Postcode'])

# 📌 Handle missing values
df['Landsize'] = df['Landsize'].fillna(df['Landsize'].median())
df['BuildingArea'] = df['BuildingArea'].fillna(df['BuildingArea'].median())
df['YearBuilt'] = df['YearBuilt'].fillna(df['YearBuilt'].median())

# 📌 Encode categorical variables
# categorical_cols = ['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
# le = LabelEncoder()
# for col in categorical_cols:
#     df[col] = le.fit_transform(df[col])

# 📌 Encode categorical variables and save the mappings
categorical_cols = ['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']
le = LabelEncoder()

# Create a dictionary to store the mappings
category_mappings = {}

# Apply LabelEncoder and store mappings
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
    category_mappings[col] = dict(zip(le.classes_, range(len(le.classes_))))

# 📌 Create a DataFrame to show the mappings (legend)
mapping_data = []

for col, mapping in category_mappings.items():
    mapping_data.append([col, mapping])

mapping_df = pd.DataFrame(mapping_data, columns=['Feature', 'Mapping'])

# 📌 Save the cleaned data and mapping to Excel with two sheets
output_path = r"C:\Users\PCAdmin\Downloads\Cleaned_Melbourne_RealEstate_with_Mappings.xlsx"

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    # Write the cleaned data to the first sheet
    df.to_excel(writer, sheet_name='Cleaned Data', index=False)
    
    # Write the category mappings to the second sheet
    mapping_df.to_excel(writer, sheet_name='Category Mappings', index=False)


# 📌 Save the cleaned DataFrame to an Excel file
# output_path = r"C:\Users\PCAdmin\Downloads\Cleaned_Melbourne_RealEstate.xlsx"
# df.to_excel(output_path, index=False)


# 📌 Split data into training and testing sets
X = df.drop(columns=['Price'])  # Features
y = df['Price']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Train a Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# 📌 Model Evaluation
y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Fixed RMSE calculation

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# 📌 Feature Importance Plot
feature_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Importance of Features in Random Forest Model")
plt.show()

# 📌 Bar Chart: Average Price by Region
region_avg_price = df.groupby('Regionname')['Price'].mean().sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=region_avg_price.index, y=region_avg_price.values, palette='viridis')
plt.xlabel("Region")
plt.ylabel("Average Price")
plt.title("Average Price by Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()