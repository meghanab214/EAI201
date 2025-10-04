import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

sns.set_style("whitegrid")

# B1: Data Loading and Preparation
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y
df['binary_target'] = np.where(df['target'] == 0, 1, 0)

class_counts = df['binary_target'].value_counts()
print("Class Distribution:")
print(class_counts)
print("\n")

# B2: Exploratory Data Analysis
plt.figure(figsize=(8, 6))
sns.countplot(x='binary_target', data=df)
plt.title('Class Distribution (1: Setosa, 0: Non-Setosa)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Setosa', 'Setosa'])
plt.show()

plt.figure(figsize=(10, 7))
sns.scatterplot(x='petal length (cm)', y='petal width (cm)', hue='binary_target', data=df, palette='viridis', s=100)
plt.title('Petal Length vs. Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Class', labels=['Setosa', 'Non-Setosa'])
plt.show()

feature_means = df.groupby('binary_target')[iris.feature_names].mean()
print("Mean Feature Values for Each Class:")
print(feature_means)
print("\n")

# B3: Model Building and Evaluation
X_features = df[iris.feature_names]
y_target = df['binary_target']

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=42, stratify=y_target)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Setosa', 'Setosa'], yticklabels=['Non-Setosa', 'Setosa'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

coefficients = pd.DataFrame(log_reg.coef_[0], X_features.columns, columns=['Coefficient'])
print("Feature Coefficients (Importance):")
print(coefficients)
print("\n")
print(f"Model Accuracy: {accuracy*100:.2f}%")