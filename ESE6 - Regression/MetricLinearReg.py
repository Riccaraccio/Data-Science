import numpy as np
import matplotlib.pyplot as plt 

# data
x = np.arange(1,11)
# y = np.array ([0.2, 0.5, 0.3, 0.7, 1.0,
#               1.5, 1.8, 2.0, 2.3, 2.2])
# introduce an outlier
y = np.array([0.2, 0.5, 0.3, 3.5, 1.0, 
              1.5, 1.8, 2.0, 2.3, 2.2])

plt.plot(x, y, 'o')

# y = A[0] + A[1]*x
def max_error(A, x, y):
    return np.max(np.abs((A[0] + A[1]*x) - y))

def l1_norm(A, x, y):
    return np.sum(np.abs((A[0] + A[1]*x) - y))

def l2_norm(A, x, y):
    return np.sum(np.square((A[0] + A[1]*x) - y))

# first guess for A
A = np.array([1, 1])

import scipy.optimize
#fmin(func, x0, args=(additional arguments))

p1 = scipy.optimize.fmin(max_error, A, args=(x, y)) 
p2 = scipy.optimize.fmin(l1_norm, A, args=(x, y))
p3 = scipy.optimize.fmin(l2_norm, A, args=(x, y))

y1 = p1[0] + p1[1]*x
y2 = p2[0] + p2[1]*x
y3 = p3[0] + p3[1]*x

plt.plot(x,y1, label='E_inf')
plt.plot(x,y2, label='E_1')
plt.plot(x,y3, label='E_2')
plt.legend()
plt.show()