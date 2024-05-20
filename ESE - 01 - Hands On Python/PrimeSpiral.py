import sympy
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100 #increase resolution of the image
plt.style.use('dark_background')

def get_coordinate(num):
    return num * np.cos(num), num * np.sin(num)
    
primes = sympy.primerange(0, 20000)
nums = np.array(list(primes))
x, y = get_coordinate(nums)
plt.scatter(x, y, s=1)
plt.axis("off")
plt.axis("equal")
plt.show()

