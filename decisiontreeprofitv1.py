from itertools import product
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

# Define given data
profit_per_acre = {'Corn': 190, 'Soybeans': 170}
loss_per_acre = {'Corn': -190, 'Soybeans': -170}
insurance_cost = {'Corn': 35000, 'Soybeans': 20000}
fertilizer_cost = {'Corn': 30000, 'Soybeans': 10000}
probability = {
    (True, True): 1.00,
    (True, False): 0.90,
    (False, True): 0.95,
    (False, False): 0.85
}

# Generate all possible decision combinations
decisions = list(product(crops, insurance_options, fertilizer_options))

data = []
for crop, insurance, fertilizer in decisions:
    prob = probability[(insurance, fertilizer)]
    revenue = profit_per_acre[crop] * 1000 * prob + loss_per_acre[crop] * 1000 * (1 - prob)
    cost = (insurance_cost[crop] if insurance else 0) + (fertilizer_cost[crop] if fertilizer else 0)
    net_profit = revenue - cost
    data.append([crop, insurance, fertilizer, net_profit])

# Create a DataFrame
df = pd.DataFrame(data, columns=['Crop', 'Insurance', 'Fertilizer', 'Profit'])

# Encode categorical variables
X = pd.get_dummies(df[['Crop', 'Insurance', 'Fertilizer']], drop_first=True)
y = df['Profit']

# Train a decision tree model
tree = DecisionTreeClassifier()
tree.fit(X, y.idxmax())

# Find the best decision
best_decision = df.loc[df['Profit'].idxmax()]
print("Best Decision:")
print(best_decision)
