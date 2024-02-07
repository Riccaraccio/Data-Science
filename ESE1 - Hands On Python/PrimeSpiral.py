import sympy
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 100
plt.style.use('dark_background')

def get_coordinate(num):
    return num * np.cos(num), num * np.sin(num)

def create_plot(nums, figsize=8, s=None, show_annot=False):
    nums = np.array(list(nums))
    x, y = get_coordinate(nums)
    plt.figure(figsize=(figsize, figsize))
    plt.axis("off")
    plt.scatter(x, y, s=s)
    plt.show()
    
primes = sympy.primerange(0, 20000)
create_plot(primes, s = 1)
