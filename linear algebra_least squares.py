"""Least-Squares."""

import numpy as np
import matplotlib.pyplot as plt

m = 10
n = 3
# Creating data
X = np.random.randn(m, n)  # "design matrix"
y = np.random.randn(m, 1)  # "outcome measures (data)"
print(np.shape(y))

XtX = X.T@X
Xty = X.T@y
normEQ = np.matrix(np.concatenate([XtX, Xty], axis=1))
Xsol = normEQ.rref()
Xsol = Xsol[0]
beta = Xsol[:, -1]

print(np.array(Xsol))
print()
print(beta)

# Comparing to left-inverse
beta2 = np.linalg.inv(XtX) @ Xty
print(beta2)

# Comparing with Python solver
beta3 = np.linalg.solve(XtX, Xty)
print(beta3)


# For instance -
data = np.array([[-4, 0, -3, 1, 2, 8, 5, 8]]).T
N = len(data)
X = np.ones([N, 1])  # matrix
b = np.linalg.solve(X.T@X, X.T@data)  # fitting the model
m = np.mean(data)  # comparing against the mean
print(b, m)  # printing the results

# computing the model-predicted values
yHat = X@b

# plotting data and model prediction
plt.plot(np.arange(1, N+1), data, 'bs-',  label='Data')
plt.plot(np.arange(1, N+1), yHat, 'ro--', label='Model pred.')
plt.legend()
plt.show()

# Matrix with nonlinearity
X = np.concatenate([np.ones([N, 1]), np.array([np.arange(0, N) ** 2]).T], axis=1)  # matrix
b = np.linalg.solve(X.T@X, X.T@data)  # fitting the model
yHat = X@b  # computing the model-predicted values

# plot data and model prediction
plt.plot(np.arange(1, N+1), data, 'bs-', label='Data')
plt.plot(np.arange(1, N+1), yHat, 'ro--', label='Model pred.')
plt.legend()
plt.show()
