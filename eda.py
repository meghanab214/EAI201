import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set plot style for better visuals
sns.set_style('whitegrid')

# Load the standard Titanic dataset from the seaborn library.
df = sns.load_dataset('titanic')
print("Standard Titanic dataset loaded successfully.")


# -------------------------------------------
# 1. Exploratory Data Analysis (EDA)
# -------------------------------------------
print("\n1. Exploratory Data Analysis")

print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\n" + "="*50 + "\n")


# Task: Summarize missing values and data types
print(" Data Types and Missing Values")
print("Data Types:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
print("\n" + "="*50 + "\n")


# Task: Visualize distributions of key features
print("Visualizing Key Feature Distributions")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distributions of Key Features', fontsize=16)

sns.histplot(df['age'].dropna(), kde=True, ax=axes[0, 0], bins=30).set_title('Age Distribution')
sns.countplot(x='sex', data=df, ax=axes[0, 1]).set_title('Gender Distribution (Sex)')
sns.countplot(x='pclass', data=df, ax=axes[1, 0]).set_title('Passenger Class (Pclass) Distribution')
sns.histplot(df['fare'], kde=True, ax=axes[1, 1], bins=40).set_title('Fare Distribution')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('feature_distributions.png')
print("Plots for feature distributions saved to 'feature_distributions.png'.\n")


# Task: Analyze relationships between features and survival rates
print("Analyzing Relationships with Survival Rate")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Survival Rate by Key Features', fontsize=16)

sns.barplot(x='sex', y='survived', data=df, ax=axes[0]).set_title('Survival Rate by Sex')
sns.barplot(x='pclass', y='survived', data=df, ax=axes[1]).set_title('Survival Rate by Passenger Class')
sns.barplot(x='embarked', y='survived', data=df, ax=axes[2]).set_title('Survival Rate by Port of Embarkation')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('survival_analysis.png')
print("Plots for survival analysis saved to 'survival_analysis.png'.")
print("\n" + "="*50 + "\n")


# -------------------------------------------
# 2. Data Cleaning and Imputation
# -------------------------------------------
print("2. Data Cleaning and Imputation")

# Impute missing values (updated to avoid FutureWarning)
df['age'] = df['age'].fillna(df['age'].median())
print(f"Missing 'age' values filled with median: {df['age'].median()}")

df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
print(f"Missing 'embarked' values filled with mode: '{df['embarked'].mode()[0]}'")

# Drop columns that are irrelevant or have too many missing values.
# CORRECTED: 'ticket' is removed from this list.
df.drop(columns=['deck', 'embark_town'], inplace=True)
print("Dropped irrelevant columns: 'deck', 'embark_town'.")
print("\n" + "="*50 + "\n")


# -------------------------------------------
# 3. Feature Engineering
# -------------------------------------------
print("3. Feature Engineering")

# Create FamilySize feature
df['family_size'] = df['sibsp'] + df['parch'] + 1
df.drop(columns=['sibsp', 'parch'], inplace=True)
print("Created 'family_size'; dropped 'sibsp' and 'parch'.")

# Create Title feature. Note: This is a simplified version for demonstration
# since the seaborn dataset doesn't have the original 'name' column.
df['title'] = df.apply(lambda row: 'Miss' if row['sex'] == 'female' and row['age'] < 18 else ('Mrs' if row['sex'] == 'female' else ('Master' if row['sex'] == 'male' and row['age'] < 18 else 'Mr')), axis=1)
print("Created a simplified 'title' feature.")

# The seaborn dataset has some redundant categorical columns. We'll drop them before encoding.
df.drop(columns=['who', 'adult_male', 'alive', 'alone'], inplace=True, errors='ignore')

# Convert remaining categorical features to numeric
df = pd.get_dummies(df, columns=['sex', 'embarked', 'class', 'title'], drop_first=True)
print("Converted categorical features to numeric using one-hot encoding.")
print("\n" + "="*50 + "\n")


# -------------------------------------------
# 4. Prepare Data for Modeling
# -------------------------------------------
print("4. Prepare Data for Modeling")

# Define features (X) and target (y)
X = df.drop('survived', axis=1)
y = df['survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data split into training and testing sets (80/20 split).")

# Check data readiness
print("\nData Readiness Check")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape:  {X_test.shape}")
print("\nFirst 5 rows of the final training data (X_train):")
print(X_train.head())

# Save the processed data to CSV files
X_train.to_csv('titanic_train_features.csv', index=False)
X_test.to_csv('titanic_test_features.csv', index=False)
y_train.to_csv('titanic_train_target.csv', index=False)
y_test.to_csv('titanic_test_target.csv', index=False)
print("\nProcessed training and testing data saved to CSV files.")