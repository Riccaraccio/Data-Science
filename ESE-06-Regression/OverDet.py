import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

n = 300 # number of equations
m = 60 # number of variables

# Ax = b
A = np.random.rand(n, m)
b = np.random.rand(n) 

# pseudoinverse solution, used as initial guess
xdag = np.linalg.pinv(A) @ b 
lam = np.array([0, 0.1, 0.5])

def reg_norm(x, A, b, lam):
    return np.linalg.norm(A @ x - b, ord=2) + lam*np.linalg.norm(x, ord=1)

fig,axs = plt.subplots(len(lam),2)
for j in range(len(lam)):
    res = scipy.optimize.minimize(reg_norm,args=(A,b,lam[j]),x0=xdag)
    x = res.x
    axs[j,0].bar(range(m),x)
    axs[j,0].set_ylabel('lam='+str(lam[j]))
    axs[j,1].hist(x,20)
    axs[j,1].set_xlim(-0.15,0.15)
    
plt.show()