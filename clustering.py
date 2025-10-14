# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
# Make sure the CSV file is in the same directory as your script or notebook
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
except FileNotFoundError:
    print("Please download the dataset and place it in the correct directory.")
    print("Dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    exit()

# --- Part A: Preprocess Data ---
print("--- Part A: Preprocessing Data ---")

# Convert 'TotalCharges' to numeric, handling errors by coercing to NaN, then drop rows with NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.reset_index(drop=True, inplace=True)

# Drop the customerID column as it's not a useful feature for clustering
df_processed = df.drop('customerID', axis=1)

# Identify numeric and categorical features
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in df_processed.columns if col not in numeric_features + ['Churn']]

# Create preprocessing pipelines for numeric and categorical data
# Numeric features will be scaled, categorical features will be one-hot encoded
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Separate features (X) and target (y)
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn'].map({'No': 0, 'Yes': 1})

# Create and apply the preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
X_processed = pipeline.fit_transform(X)

print("Data preprocessing complete.")
print(f"Shape of processed data: {X_processed.shape}")


# --- Part B: PCA ---
print("\n--- Part B: Principal Component Analysis (PCA) ---")

# Apply PCA to reduce the data to 2 components
pca = PCA(n_components=2, random_state=42)
principal_components = pca.fit_transform(X_processed)
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Print the explained variance
explained_variance = pca.explained_variance_ratio_
total_variance = np.sum(explained_variance)
print(f"Variance explained by PC1: {explained_variance[0]:.2%}")
print(f"Variance explained by PC2: {explained_variance[1]:.2%}")
print(f"Total variance explained: {total_variance:.2%}")

# --- Part C: K-Means on PCA data ---
print("\n--- Part C: K-Means Clustering ---")

# Determine the optimal number of clusters (K) using the Elbow Method and Silhouette Score
sse = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(df_pca)
    sse.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_pca, kmeans.labels_))

# Set the final number of clusters based on the analysis
final_k = 4
print(f"The final K chosen is {final_k}.")

# Run K-Means with the chosen K
kmeans = KMeans(n_clusters=final_k, n_init=10, random_state=42)
df_pca['Cluster'] = kmeans.fit_predict(df_pca)

# --- Part D: Business Analysis ---
print("\n--- Part D: Business Analysis & Insights ---")

# Add the cluster labels back to the original dataframe
df['Cluster'] = df_pca['Cluster']
df['Churn_numerical'] = y

# Analyze the clusters
cluster_summary_numeric = df.groupby('Cluster')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean().round(2)
print("\n--- Average Numeric Values per Cluster ---")
print(cluster_summary_numeric)

churn_rate = df.groupby('Cluster')['Churn_numerical'].value_counts(normalize=True).unstack().fillna(0)
churn_rate.columns = ['Churn_No', 'Churn_Yes']
churn_rate['Churn_Rate_%'] = (churn_rate['Churn_Yes'] * 100).round(2)
print("\n--- Churn Rate per Cluster ---")
print(churn_rate)