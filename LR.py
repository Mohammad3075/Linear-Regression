# "Linear Rgression example, Author: Mohammad Gharibi"
# "This script demonstrates how to generate synthetic linear regression data,
# fit a regression model manually (using the normal equation) and using
# scikit-learn, visualize the data and regression line, and compute the
# Mean Squared Error (MSE)."

# Depedencies : "numpy, matplotlib, scikit-learn"


import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
random.seed(888)
n_feature = 1
n_sample = 100

x = np.random.normal(loc=30, scale=10, size=(n_sample, n_feature))
# print(x)
beta = np.random.normal(loc=10, scale=2, size=(n_feature, 1))
# print(beta)
epsilon = np.random.normal(loc=0, scale=1, size=(n_sample, 1))
# print(epsilon)
y = np.dot(x, beta) + epsilon
print(y)
print(y.shape)
x_flat = x.flatten()
y_flat = y.flatten()
# plt.scatter(x_flat, y_flat)
# plt.xlabel("feature X")
# plt.ylabel("Target Y")
# plt.title("Scatter plot of synthetic regression Data")
# plt.show()
A = np.dot(x.T, x)
b = np.dot(x.T, y)
max_beta = np.linalg.solve(A, b)
print(max_beta)
lr_model = LinearRegression()
lr_model.fit(x, y)
print("Estimated_coefficient_s:", lr_model.coef_)
print("Estimated intercept:", lr_model.intercept_)
y_pred = lr_model.predict(x)
plt.scatter(x.flatten(), y.flatten(), color='blue')
plt.plot(x.flatten(), y_pred.flatten(), color='red')
plt.show()
mse = mean_squared_error(y_flat, y_pred)
print("MSE via sklearn:", mse)
rmse = np.sqrt(mse)
print("RMSE:", rmse)
