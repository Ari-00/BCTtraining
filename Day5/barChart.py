from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(script_dir, "New Superstore Data.xlsx")

df=pd.read_excel(excel_path)
#print(df)

# Features and target
X = df[['Profit', 'Discount', 'Quantity']] #categorical value
y = df['Sales']#targeted value

# Train-test split - THIS WAS MISSING!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest = Bagging of Decision Trees (using Regressor for continuous target)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy (R² score for regression)
print("R² Score:", model.score(X_test, y_test))

# visualization
# ----------------------------
# 🔹 1. Pairplot – feature relationships
# ----------------------------
# Using a column that exists in the Superstore data instead of "species"
if 'Segment' in df.columns:
    sns.pairplot(df, hue="Segment", corner=True)
elif 'Category' in df.columns:
    sns.pairplot(df, hue="Category", corner=True)
else:
    sns.pairplot(df, corner=True)
plt.suptitle("Superstore Data Feature Relationships", y=1.02)
plt.show()
