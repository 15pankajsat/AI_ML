import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# smooth line
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range_pred = model.predict(X_range)

# plot
plt.scatter(X, y, label="Actual Data")
plt.plot(X_range, y_range_pred, label="Regression Line")
plt.legend()
plt.show()