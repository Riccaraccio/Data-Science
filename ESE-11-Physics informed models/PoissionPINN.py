"""Physics-Informed Neural Network (PINN) for Differential Equation Solving

This code implements a Physics-Informed Neural Network (PINN) to solve a second-order 
boundary value problem (BVP) of the form:
    u_xx = sin(pi*x)  with boundary conditions u(0) = u(1) = 0

PINNs integrate physical laws into neural network training through custom loss functions
that enforce both the governing differential equations and boundary conditions. Unlike
traditional numerical methods, PINNs learn continuous solutions that automatically
satisfy the underlying physics.

The approach consists of:
1. Defining a neural network architecture to approximate the solution u(x)
2. Formulating a physics-informed loss function that penalizes violations of:
   - The differential equation (u_xx - sin(pi*x) = 0)
   - The boundary conditions (u(0) = 0 and u(1) = 0)
3. Training the neural network to minimize this combined loss

The exact analytical solution to this problem is u(x) = -1/pi^2 * sin(pi*x), which is
used to validate the accuracy of the PINN approximation during training.
"""

import os
# Disable TensorFlow's oneDNN optimizations to ensure consistent results across different hardware
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for reproducibility

# Import required libraries
import tensorflow as tf      # Deep learning framework
import numpy as np           # Numerical computing library
import matplotlib.pyplot as plt  # Plotting library

# Define the neural network architecture for the PINN, inheriting from tf.keras.Model
class PINN(tf.keras.Model):
    def __init__(self):
        # Initialize the parent class (tf.keras.Model)
        super(PINN, self).__init__()
        
        # Define the network architecture - 3 hidden layers with 20 neurons each using tanh activation
        # tanh activation is common in PINNs as it produces smooth outputs suitable for physics problems
        self.hidden_layer1 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden_layer2 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden_layer3 = tf.keras.layers.Dense(20, activation='tanh')
        
        # Output layer with 1 neuron (scalar output) and linear activation
        # Linear activation allows unconstrained output range needed for the solution
        self.output_layer = tf.keras.layers.Dense(1, activation="linear")

    def call(self, x):
        # Define the forward pass (how data flows through the network)
        x = self.hidden_layer1(x)  # Pass input through first hidden layer
        x = self.hidden_layer2(x)  # Pass through second hidden layer
        x = self.hidden_layer3(x)  # Pass through third hidden layer
        return self.output_layer(x)  # Return the output (predicted solution u(x))


# Define the source term f(x) of the differential equation u_xx = f(x)
def f(x):
    # For this problem, f(x) = sin(πx)
    return tf.sin(np.pi * x)


# Define the physics-informed loss function that enforces the differential equation
def custom_loss(model, x):    
    # Use automatic differentiation to compute derivatives of the model output
    with tf.GradientTape() as g:  # Outer tape to compute second derivative
        g.watch(x)  # Tell TensorFlow to track operations on x
        with tf.GradientTape() as gg:  # Inner tape to compute first derivative
            gg.watch(x)  # Tell TensorFlow to track operations on x
            u = model(x)  # Neural network output (predicted solution)
        u_x = gg.gradient(u, x)  # First derivative (du/dx)
    u_xx = g.gradient(u_x, x)  # Second derivative (d²u/dx²)
    
    # Compute the residual: u_xx - f(x) should be approximately zero if the PDE is satisfied
    residual = u_xx - f(x)
    
    # Return mean squared error of the residual (physics-informed component of the loss)
    return tf.reduce_mean(tf.square(residual))


# Define the boundary loss function to enforce boundary conditions u(0) = u(1) = 0
def boundary_loss(model):
    # Evaluate the model at the boundary points
    u_0 = model(tf.constant([[0.0]], dtype=tf.float32))  # Value at x=0
    u_1 = model(tf.constant([[1.0]], dtype=tf.float32))  # Value at x=1
    
    # Return sum of squared errors at boundaries (should approach zero during training)
    return tf.square(u_0) + tf.square(u_1)


# Define the total loss as a combination of the physics-based and boundary losses
def total_loss(model, x):
    # Combine both loss components without specific weighting
    return custom_loss(model, x) + boundary_loss(model)


# Training procedure for the PINN
def train(model, x, epochs):
    # Set up interactive plotting to visualize training progress
    plt.ion()  # Turn on interactive mode for matplotlib
    plt.figure()  # Create a new figure
    
    # Prepare data for plotting the exact solution
    x_exact = np.linspace(0, 1, 100).reshape(-1, 1)  # Evenly spaced points in [0,1]
    u_exact = -1/np.pi**2 * np.sin(np.pi * x_exact)  # Exact analytical solution
    
    # Initialize the Adam optimizer (commonly used for training neural networks)
    optimizer = tf.keras.optimizers.Adam()
    
    # Main training loop
    for epoch in range(epochs):
        # Compute loss and gradients
        with tf.GradientTape() as g:
            loss = total_loss(model, x)  # Compute the combined loss
        # Compute gradients of the loss with respect to trainable variables
        gradients = g.gradient(loss, model.trainable_variables)
        # Update model parameters using the optimizer
        # zip pairs each gradient with its corresponding variable for the update
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Visualize progress every 10 epochs
        if epoch % 10 == 0:
            # Clear the previous plot
            plt.clf()
            
            # Plot current PINN solution and exact solution for comparison
            plt.plot(x, model(x), label="PINN")  # PINN approximation
            plt.plot(x_exact, u_exact, label="Exact")  # Exact solution
            
            # Add plot details
            plt.title(f"Epoch {epoch}, Loss: {loss.numpy()}")
            plt.xlim(-0.5, 1.5)  # Set x-axis limits for better visualization
            plt.ylim(-0.2, 0.1)  # Set y-axis limits for better visualization
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.legend()
            
            # Display the plot and pause briefly to allow update
            plt.show()
            plt.pause(0.01)
            
            # Print loss value to console
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    
    # Turn off interactive plotting after training
    plt.ioff()
    
    # Keep the final plot open for inspection
    plt.show()


# Main execution block
if __name__ == "__main__":
    # Generate collocation points (training data) as evenly spaced points in [0,1]
    # Reshape to column vector and convert to TensorFlow tensor type
    x = tf.convert_to_tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=tf.float32)
    
    # Create an instance of the PINN model
    model = PINN()

    # Train the model for 200 epochs
    train(model, x, epochs=200)