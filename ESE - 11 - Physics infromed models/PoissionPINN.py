import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the neural network, derived from tf.keras.Model
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__() # do the same initialization as the parent class
        self.hidden_layer1 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden_layer2 = tf.keras.layers.Dense(20, activation='tanh')
        self.hidden_layer3 = tf.keras.layers.Dense(20, activation='tanh')
        self.output_layer = tf.keras.layers.Dense(1, activation="linear")

    def call(self, x): # Define the forward pass
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        return self.output_layer(x)


# Define the source term f(x)
def f(x):
    return tf.sin(np.pi * x)


# Define the custom loss function
def custom_loss(model, x):    
    with tf.GradientTape() as g: # compute the gradients
        g.watch(x)
        with tf.GradientTape() as gg:
            gg.watch(x)
            u = model(x)
        u_x = gg.gradient(u, x)
    u_xx = g.gradient(u_x, x)        
    residual = u_xx + f(x) # compute the residual
    return tf.reduce_mean(tf.square(residual)) # compute the mean squared loss


# Define the boundary loss, to impose the boundary conditions
def boundary_loss(model):
    u_0 = model(tf.constant([[0.0]], dtype=tf.float32)) # u(0)
    u_1 = model(tf.constant([[1.0]], dtype=tf.float32)) # u(1)
    return tf.square(u_0) + tf.square(u_1) # squared loss


# Define the total loss
def total_loss(model, x): # total loss = custom loss + boundary loss
    return custom_loss(model, x) + boundary_loss(model)


# Training the PINN
def train(model, x, epochs):
    
    plt.ion()
    plt.figure()
    # plot the exact solution
    x_exact = np.linspace(0, 1, 100).reshape(-1, 1)
    u_exact = 1/np.pi**2 * np.sin(np.pi * x_exact)
    
    optimizer = tf.keras.optimizers.Adam() # Set the optimizer
    for epoch in range(epochs): # Loop over the epochs
        with tf.GradientTape() as g:
            loss = total_loss(model, x) # Compute the loss
        gradients = g.gradient(loss, model.trainable_variables) # Compute the gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Update the weights, zip is used to match the gradients to the variables creating pairs
        
        if epoch % 10 == 0:
            # Plot the current solution
            plt.clf()
            plt.plot(x, model(x), label="PINN")
            plt.plot(x_exact, u_exact, label="Exact")
            plt.title(f"Epoch {epoch}, Loss: {loss.numpy()}")
            plt.xlim(-1,2)
            plt.ylim(-0.1,0.2) 
            plt.xlabel("x")
            plt.ylabel("u(x)")
            plt.legend()
            plt.show()
            plt.pause(0.01)
            print(f"Epoch {epoch}, Loss: {loss.numpy()}")
    plt.ioff()
    # keep the plot open
    plt.show()


# Main execution
if __name__ == "__main__": # Execute the main code
    # Training data (collocation points), shped as a column vector, tf.float32
    x = tf.convert_to_tensor(np.linspace(0, 1, 100).reshape(-1, 1), dtype=tf.float32)
    
    # Instantiate the model
    model = PINN()

    # Train the model
    train(model, x, epochs=200)