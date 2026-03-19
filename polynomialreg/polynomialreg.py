import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# pipeline model
model = Pipeline([
    ("poly", PolynomialFeatures(degree=2)),
    ("linear", LinearRegression())
])

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# smooth curve
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range = model.predict(X_range)

# plot
plt.scatter(X, y, label="Actual Data")
plt.plot(X_range, y_range, label="Polynomial Curve")
plt.legend()
plt.show()