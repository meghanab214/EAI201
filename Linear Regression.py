import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sns.set_style('whitegrid')

print(" Task A1: Data Loading and Exploration ")

boston = fetch_openml(name='boston', version=1, as_frame=True)
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target

num_samples, num_features = df.shape
print(f"Dataset contains {num_samples} samples (houses).")
print(f"Dataset contains {num_features - 1} features.")
print("\nDataFrame structure (first 5 rows):")
print(df.head())

price_stats = df['PRICE'].describe()
print("\nHouse Price Statistics:")
print(f"Minimum Price: ${price_stats['min']*1000:,.2f}")
print(f"Maximum Price: ${price_stats['max']*1000:,.2f}")
print(f"Mean Price:    ${price_stats['mean']*1000:,.2f}")

print("\n\n Task A2: Exploratory Data Analysis ")

plt.figure(figsize=(10, 6))
sns.histplot(df['PRICE'], bins=30, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price ($1000s)')
plt.ylabel('Frequency')
plt.show()

correlation_matrix = df.corr()
price_correlation = correlation_matrix['PRICE'].sort_values(ascending=False)
print("\nCorrelation of features with price:")
print(price_correlation)

strongest_corr_feature = price_correlation.index[1]
print(f"\nFeature with strongest positive correlation: '{strongest_corr_feature}'")

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[strongest_corr_feature], y=df['PRICE'])
plt.title(f'Price vs. {strongest_corr_feature} (Most Correlated Feature)')
plt.xlabel(f'{strongest_corr_feature}')
plt.ylabel('Price ($1000s)')
plt.show()

print("\nAnalysis of Price Distribution:")
print("The price distribution is right-skewed (positively skewed).")
print("This suggests that while most houses are clustered around the lower-to-mid price range, there are a few houses that are significantly more expensive, creating a tail on the right side of the distribution.")
print("\n\nTask A3: Model Building and Evaluation")

X = df.drop('PRICE', axis=1)
y = df['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

model = LinearRegression()
model.fit(X_train, y_train)
print("\nLinear Regression model trained successfully.")

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Model Performance")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} (in $1000s)")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs. Predicted Prices')
plt.xlabel('Actual Prices ($1000s)')
plt.ylabel('Predicted Prices ($1000s)')
plt.show()

print("\n Final Analysis ")
print(f"What is your model's R² score? Is this considered good performance?")
print(f"The R² score is {r2:.4f}. This means our model explains approximately {r2:.1%} of the variance in house prices. For this classic dataset, a score in this range (0.6-0.7) is considered decent but not great. It shows a moderately effective model but indicates that a simple linear model cannot capture all the complexities of the housing market.")
print("\nBased on the actual vs predicted plot, does your model perform better on certain price ranges?")
print("Yes. The model performs better for lower to mid-range priced houses, where the data points are clustered more tightly around the red diagonal line. For more expensive houses (e.g., above $35k), the predictions are more spread out and less accurate. The model tends to underestimate the price of very expensive homes.")