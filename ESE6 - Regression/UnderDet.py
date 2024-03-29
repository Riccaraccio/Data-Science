import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Underdetermined
n = 20
m = 100
A = np.random.rand(n,m)
b = np.random.rand(n)

def two_norm(x):
    return np.linalg.norm(x,ord=2)

#constraied are passed as a list of dictionaries
# lamda allows to define an anonymous function
constr = ({'type': 'eq', 'fun': lambda x:  A @ x - b})

x0 = np.random.rand(m)
res = scipy.optimize.minimize(two_norm, x0, method='SLSQP',constraints=constr)
x2 = res.x

def one_norm(x):
    return np.linalg.norm(x,ord=1)

res = scipy.optimize.minimize(one_norm, x0, method='SLSQP',constraints=constr)
x1 = res.x

fig,axs = plt.subplots(2,2)
axs = axs.reshape(-1)

axs[0].bar(range(m),x2)
axs[0].set_title('x2')
axs[1].bar(range(m),x1)
axs[1].set_title('x1')

axs[2].hist(x2,40)
axs[3].hist(x1,40)

plt.show()