import numpy as np

# Dataset
study_hours = np.array([2, 4, 6, 8, 10])
pass_fail = np.array([0, 0, 1, 1, 1])

# Function to calculate Gini impurity for a group
def gini_impurity(y):
    classes = np.unique(y)
    impurity = 1.0
    for cls in classes:
        p = np.sum(y == cls) / len(y)
        impurity -= p**2
    return impurity

# Function to calculate weighted Gini impurity for a split point
def weighted_gini(split):
    left_group = pass_fail[study_hours <= split]
    right_group = pass_fail[study_hours > split]
    n = len(pass_fail)
    gini_left = gini_impurity(left_group)
    gini_right = gini_impurity(right_group)
    weighted = len(left_group)/n * gini_left + len(right_group)/n * gini_right
    return weighted

# Possible splits
split_candidates = [(study_hours[i] + study_hours[i+1]) / 2 for i in range(len(study_hours)-1)]

# Calculate Gini impurity for each split
split_ginis = {s: weighted_gini(s) for s in split_candidates}

# Find the best split with minimum Gini impurity
best_split = min(split_ginis, key=split_ginis.get)
best_gini = split_ginis[best_split]

# Print results
print("Possible splits and their Gini impurities:")
for split, gini in split_ginis.items():
    print(f"Split at {split}: Gini impurity = {gini:.3f}")

print(f"Best split found at {best_split} with minimum Gini impurity = {best_gini:.3f}")