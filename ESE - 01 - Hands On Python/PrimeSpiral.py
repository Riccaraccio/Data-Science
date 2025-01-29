"""Prime Number Spiral Visualization.

This code creates a mathematical visualization of prime numbers up to 20,000,
where each prime p is plotted as a point with coordinates (p*cos(p), p*sin(p)).
The resulting pattern reveals the distribution and relationships of prime numbers
through a geometric spiral representation.
To learn more: https://www.youtube.com/watch?v=EK32jo7i5LQ
"""

# Import required libraries
import sympy          # For generating prime numbers
import numpy as np    # For numerical operations
import matplotlib.pyplot as plt  # For plotting

# Set the display parameters
plt.rcParams['figure.dpi'] = 100     # Set higher resolution for clearer output
plt.style.use('dark_background')      # Use dark theme for better visibility

# Convert numbers into polar coordinates and return their Cartesian equivalents.
def get_coordinate(num):
    return num * np.cos(num), num * np.sin(num)
    
# Generate prime numbers from 0 to 20000
primes = sympy.primerange(0, 20000)

# Convert the prime numbers iterator to a numpy array for vectorized operations
nums = np.array(list(primes))

# Convert prime numbers to coordinates
x, y = get_coordinate(nums)

# Create the visualization
plt.scatter(x, y, s=1)      # Plot points with size 1
plt.axis("off")             # Hide the axes
plt.axis("equal")           # Ensure equal scaling on both axes
plt.show()                  # Display the plot