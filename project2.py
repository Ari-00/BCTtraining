import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_excel("New Superstore Data.xlsx")

# Features and target
X = df[['Profit', 'Discount', 'Quantity']] #categorical value
y = df['Sales']#targeted value

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)