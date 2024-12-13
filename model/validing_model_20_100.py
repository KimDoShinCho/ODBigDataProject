import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'D:\\study\\class\\빅데\\random_forest_label_8_대피.csv'  # Replace with your actual file path
rf_data = pd.read_csv(file_path)

# Splitting features and target
X = rf_data[['gender', 'age', 'purpose', 'dest_hdong_cd']]
y = rf_data['score']

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

# R^2 score for varying number of trees (n_estimators)
tree_range = np.arange(100, 1100, 100)
tree_r2_scores = []

for n_trees in tree_range:
    print(n_trees, 'ntree')
    model = RandomForestRegressor(n_estimators=n_trees, max_depth=10, random_state=31)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    tree_r2_scores.append(r2)

# R^2 score for varying max_depth
depth_range = np.arange(5, 45, 5)
depth_r2_scores = []

for max_depth in depth_range:
    print(max_depth, 'depth')
    model = RandomForestRegressor(n_estimators=100, max_depth=max_depth, random_state=31)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    depth_r2_scores.append(r2)

# Optimal number of trees and depth
optimal_trees = tree_range[np.argmax(tree_r2_scores)]
optimal_depth = depth_range[np.argmax(depth_r2_scores)]

# Plot R^2 score vs number of trees
plt.figure(figsize=(10, 6))
plt.plot(tree_range, tree_r2_scores, marker='o', label='R^2 Score')
plt.xlabel("Number of Trees")
plt.ylabel("R^2 Score")
plt.title("R^2 Score vs Number of Trees (Depth = 10)")
plt.grid(True)
plt.legend()
plt.show()

# Plot R^2 score vs max depth
plt.figure(figsize=(10, 6))
plt.plot(depth_range, depth_r2_scores, marker='o', label='R^2 Score')
plt.xlabel("Max Depth")
plt.ylabel("R^2 Score")
plt.title("R^2 Score vs Max Depth (Trees = 100)")
plt.grid(True)
plt.legend()
plt.show()

# R^2 score distribution with different random seeds
seeds = range(1, 21)
r2_scores_seeds = []

for seed in seeds:
    model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores_seeds.append(r2)

# Plot R^2 score distribution
plt.figure(figsize=(10, 6))
plt.boxplot(r2_scores_seeds, vert=False, patch_artist=True)
plt.xlabel("R^2 Score")
plt.title(f"R^2 Score Distribution (Trees = 100, Depth = 20)")
plt.grid(True)
plt.show()

# Summary of optimal parameters and distribution
print(f"Optimal Number of Trees: {optimal_trees}")
print(f"Optimal Max Depth: {optimal_depth}")
print(f"Mean R^2 Score (20 runs): {np.mean(r2_scores_seeds):.4f}")
print(f"Std Dev of R^2 Scores (20 runs): {np.std(r2_scores_seeds):.4f}")
