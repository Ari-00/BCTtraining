# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn import tree
# import matplotlib.pyplot as plt
# import pandas as pd

# # # Load dataset
# # X, y = load_iris(return_X_y=True)
# # Load dataset
# df = pd.read_excel("New Superstore Data.xlsx")
# # Features and target
# X = df[['Profit', 'Discount', 'Quantity']] #categorical value
# y = df['Sales']#targeted value


# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Create model
# clf = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)

# # Train
# clf.fit(X_train, y_train)

# # Accuracy
# print("Accuracy:", clf.score(X_test, y_test))

# # Plot the tree
# plt.figure(figsize=(12,6))
# tree.plot_tree(clf, feature_names=load_iris().feature_names,
#                class_names=load_iris().target_names,
#                filled=True)
# plt.show()
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd


df=pd.read_excel("Day5/New Superstore Data.xlsx")
#print(df)

# Features and target
X = df[['Profit', 'Discount', 'Quantity']] #categorical value
y = df['Sales']#targeted value

# Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Load dataset
#X, y = load_iris(return_X_y=True)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create model
clf = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=42)

# Train
clf.fit(X_train, y_train)

# Accuracy
print("Accuracy:", clf.score(X_test, y_test))

# Plot the tree
plt.figure(figsize=(12,6))
tree.plot_tree(clf, feature_names=['Profit', 'Discount', 'Quantity'],
               filled=True)
plt.show()
