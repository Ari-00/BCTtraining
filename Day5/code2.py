# --- Import Libraries ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# --- Load Dataset ---
df = pd.read_excel("Day5/New Superstore Data.xlsx")

# --- Feature Selection ---
df = df[['Segment', 'Category', 'Sales', 'Discount', 'Profit']]

# --- Create Target Variable ---
df['High_Profit'] = np.where(df['Profit'] > df['Profit'].median(), 1, 0)

# --- Encode Categorical Columns ---
le = LabelEncoder()
df['Segment'] = le.fit_transform(df['Segment'])
df['Category'] = le.fit_transform(df['Category'])

# --- Split Data ---
X = df[['Segment', 'Category', 'Sales', 'Discount']]
y = df['High_Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Ensemble Models ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
adb = AdaBoostClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# --- Stacking Ensemble ---
estimators = [
    ('rf', rf),
    ('adb', adb),
    ('gb', gb)
]
stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

# --- Train Models ---
rf.fit(X_train, y_train)
adb.fit(X_train, y_train)
gb.fit(X_train, y_train)
stack_model.fit(X_train, y_train)

# --- Predictions ---
y_pred_rf = rf.predict(X_test)
y_pred_adb = adb.predict(X_test)
y_pred_gb = gb.predict(X_test)
y_pred_stack = stack_model.predict(X_test)

# --- Accuracy Scores ---
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_adb = accuracy_score(y_test, y_pred_adb)
acc_gb = accuracy_score(y_test, y_pred_gb)
acc_stack = accuracy_score(y_test, y_pred_stack)

# --- Display Results ---
print("Random Forest Accuracy:", acc_rf)
print("AdaBoost Accuracy:", acc_adb)
print("Gradient Boosting Accuracy:", acc_gb)
print("Stacking Ensemble Accuracy:", acc_stack)

# --- Visualization: Compare All Ensemble Methods ---
models = ['Random Forest', 'AdaBoost', 'Gradient Boosting', 'Stacking']
accuracy = [acc_rf, acc_adb, acc_gb, acc_stack]

plt.figure(figsize=(7,5))
sns.barplot(x=models, y=accuracy, palette='viridis')
plt.title("Comparison of Ensemble Methods on Superstore Dataset")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.savefig("Day5/ensemble_comparison.png", dpi=100, bbox_inches='tight')
plt.close()

# --- Confusion Matrix for Stacking Model ---
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm')
plt.title("Confusion Matrix - Stacking Ensemble")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Day5/confusion_matrix.png", dpi=100, bbox_inches='tight')
plt.close()

# --- Classification Report for Stacking Model ---
print("\nStacking Ensemble Classification Report:\n")
print(classification_report(y_test, y_pred_stack))