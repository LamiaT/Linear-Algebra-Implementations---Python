"""Inverse, Left Inverse, Right Inverse Matrices."""

import numpy as np
import matplotlib.pyplot as plt

m = 3  # size of square matrix
A = np.random.randn(m, m)  # random matrix
A_inv = np.linalg.inv(A)  # computing its inverse
idm = A@A_inv  # checking the multiplication
print(idm)

# show in an image
plt.subplot(131)
plt.imshow(A)
plt.title('Matrix A')

plt.subplot(132)
plt.imshow(A_inv)
plt.title('Matrix $A^{-1}$')

plt.subplot(133)
plt.imshow(idm)
plt.title('AA$^{-1}$')

plt.show()

# m > n for left inverse,
# m < n for right inverse
m, n = 6, 3
# creating matrices
A = np.random.randn(m, n)
AtA = A.T@A
AAt = A@A.T
# inspecting ranks
print('Shape of A^TA:', np.shape(AtA))
print('Rank of A^TA:', np.linalg.matrix_rank(AtA))
print('Shape of AA^T:', np.shape(AAt))
print('Rank of AA^T:', np.linalg.matrix_rank(AAt))

A_left_inverse = np.linalg.inv(AtA)@A.T  # left inverse
A_right_inverse = A.T@np.linalg.inv(AAt)  # right inverse

I_left = A_left_inverse @ A
I_right = A @ A_right_inverse
print(I_left)
print()
print(I_right)
print()

# Using the inverse function
AtA_inv = np.linalg.inv(AtA)
I_AtA = AtA_inv @ AtA

AAt_inv = np.linalg.inv(AAt)
I_AAt = AAt_inv @ AAt

print(I_AtA)
print()
print(I_AAt)
