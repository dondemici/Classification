from itertools import product
import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# Define the possible choices
crops = ['Corn', 'Soybeans']
insurance_options = [True, False]
fertilizer_options = [True, False]

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
#tree.fit(X, y.idxmax())
tree.fit(X, y.values.ravel())

# Visualize the decision tree
dot_data = export_graphviz(tree, out_file=None, 
                           feature_names=X.columns, 
                           class_names=[str(p) for p in y.unique()],
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
# Save the decision tree as a PDF to the specified location
graph.render(r"C:\Users\PCAdmin\Documents\Douglas_College\3rd_Sem\2_Data_Analytics\Decision_Tree\decision_tree")

# Find the best decision
best_decision = df.loc[df['Profit'].idxmax()]
print("Best Decision:")
print(best_decision)