"""Matrix addition, subtraction, scalar multiplication, transpose, diagonal, trace."""

# Importing numpy library
import numpy as np

# Different forms of matrix
square_matrix = np.random.randn(4, 4)

rectangular_matrix = np.random.randn(4, 2)

identity_matrix = np.eye(3)

zero_matrix = np.zeros((3, 3))

diagonal_matrix = np.diag([1, 2, 3, 4, 5])

A = np.random.randn(5, 5)
B = np.triu(A)
C = np.tril(A)

A = np.random.randn(3, 2)
B = np.random.randn(4, 4)
C = np.concatenate((A, B), axis=1)
# When conconating, sizes of the matrices must match

# Matrix addition and subtraction
A = np.random.randn(5, 4)
B = np.random.randn(5, 4)
print(A + B)
# must be same shape

# Shifting matrix
lambda_ = 0.05
n = 5  # size of square matrix
d = np.random.randn(n, n)
# can only shift square matrix
ds = d + lambda_ * np.eye(n)

# Scalar Multiplication of Matrices
matrix = np.array([[1, 2], [3, 4]])
s = 4
print(matrix * s)
print(s * matrix)

# Transpose of Matrix
matrix = np.array([[1, 2, 3], [2, 3, 4]])

print(matrix)
print()
print(matrix.T)  # transpose once
print()
print(matrix.T.T)  # double transpose returns original matrix M
print()
print(np.transpose(matrix))  # Using the function transpose

# Diagonal
matrix = np.round(6 * np.random.randn(5, 5))
print(matrix)
d = np.diag(matrix)  # extract the diagonals
d = np.diag(matrix)  # input is matrix, output is vector
dd = np.diag(d)  # input is vector, output is matrix

# Trace as the summation of diagonal elements
trace1 = np.trace(matrix)
trace2 = sum(np.diag(matrix))

















