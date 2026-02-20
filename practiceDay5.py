import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "age": [22, 25, 47, 52, 46, 56, 55, 60],
    "salary": [30000, 40000, 50000, 60000, 52000, 58000, 62000, 72000],
    "approved": [0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)


X = df[['age', 'salary']]
y = df['approved']  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

