import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset
data = pd.DataFrame({
    'Experience': [1,2,3,4,5,6,7,8,9,10],
    'Salary': [30,35,40,45,50,55,60,65,70,75]
})

X = data[['Experience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
print("R2 Score:", r2_score(y_test, y_pred))


# Plot original data points
plt.scatter(X, y)

# Plot regression line
plt.plot(X, model.predict(X))

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary (Linear Regression)")
plt.grid()
plt.show()