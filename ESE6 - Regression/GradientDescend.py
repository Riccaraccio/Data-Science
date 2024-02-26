import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

h = 0.5
x = np.arange(-4,4+h,h)
y = np.arange(-4,4+h,h)

X,Y = np.meshgrid(x,y)
Fquad = np.power(X,2) + 3*np.power(Y,2)

## Gradient Descent
x = np.zeros(10)
y = np.zeros(10)
f = np.zeros(10)

x[0] = 3  # Initial guess
y[0] = 2 

f[0] = x[0]**2 + 3*y[0]**2 # Initial function value

for j in range(len(x)-1):
    Del = (x[j]**2 + 9*y[j]**2)/(2*x[j]**2 + 54*y[j]**2)
    x[j+1] = (1 - 2*Del)*x[j] # update values
    y[j+1] = (1 - 6*Del)*y[j]
    f[j+1] = x[j+1]**2 + 3*y[j+1]**2
    
    if np.abs(f[j+1]-f[j]) < 10**(-6): # check convergence
        x = x[:j+2]
        y = y[:j+2]
        f = f[:j+2]
        break
fig,ax = plt.subplots(1,1,subplot_kw={'projection': '3d'})
ax.plot_surface(X, Y, Fquad,linewidth=0,color='k',alpha=0.3)
ax.scatter(x,y,f,'o',color='r',s=20)
ax.plot(x,y,f,':',color='k',linewidth=3)
ax.contour(X, Y, Fquad, zdir='z', offset=ax.get_zlim()[0], cmap='gray')
ax.view_init(elev=40, azim=-140)
plt.show()