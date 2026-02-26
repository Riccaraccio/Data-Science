"""Neural Network Prediction of Henon Map Dynamics.

This code trains a neural network to predict the behavior of the Henon map dynamical system using:
1. Generating training data from the Henon map equations
2. Training a neural network to learn the mapping from current state to next state
3. Predicting trajectories using the trained model
4. Comparing neural network predictions with true Henon map trajectories
5. Visualizing results and calculating error metrics

The program demonstrates how well neural networks can learn chaotic dynamical systems
and visually compares the true vs. predicted behavior of the Henon map attractor.

Data Structure
-------------
- X, Y: Arrays containing the x and y coordinates of the Henon map trajectories
- nn_input: Input data for neural network (current state)
- nn_output: Target output for neural network (next state)
- X_predict, Y_predict: Neural network predicted trajectory
- X_true, Y_true: True trajectory calculated from Henon map equations
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimization for reproducibility

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Model parameters for the Henon map
a = 1.4  # Standard parameter value for chaotic behavior
b = 0.3  # Standard parameter value for chaotic behavior
n = 100  # Number of steps for each trajectory

"""# Visualization: Generate a Henon map animation
# Generate the Henon map
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], 'o', markersize=1, color='black')

# Initial conditions
X = [1]
Y = [1]

# Update function for the animation
def update(frame):
    global X, Y
    if len(X) < n:
        X.append(1 - a*X[-1]**2 + Y[-1])
        Y.append(b*X[-2])
        line.set_data(X, Y)
    return line,

ani = FuncAnimation(fig, update, frames=n, blit=True, interval=1)
plt.show()

plt.plot (range(1, n+1), X, '-o', markersize=1, color='black')
plt.show()
plt.plot (range(1, n+1), Y, '-o', markersize=1, color='red')
plt.show()
"""

# Load the model if it exists, otherwise train it
try:
    net = tf.keras.models.load_model('henon_map.keras')
    model_loaded = True
    print("MODEL LOADED")
except:
    model_loaded = False
    
    # Generate training data from m different initial conditions
    m = 1000  # Number of initial conditions
    
    # Pre-allocate arrays for efficiency
    X = np.zeros((m, n))
    Y = np.zeros((m, n))
    
    # Carefully selected initial conditions to avoid overflow
    # Use stratified sampling to better cover the attractor's basin
    X[:,0] = np.linspace(-0.75, 0.75, m)
    Y[:,0] = np.linspace(-0.75, 0.2, m)
    
    # Generate trajectories using the Henon map equations
    for i in range(1, n):
        X[:,i] = 1 - a*X[:,i-1]**2 + Y[:,i-1]  # Henon map x-component
        Y[:,i] = b*X[:,i-1]                     # Henon map y-component
    
    # Prepare input-output pairs for neural network training
    # Each pair consists of (current state, next state)
    nn_input = np.column_stack((X[:, :-1].reshape(-1, 1), Y[:, :-1].reshape(-1, 1)))
    nn_output = np.column_stack((X[:, 1:].reshape(-1, 1), Y[:, 1:].reshape(-1, 1)))
    
    # Split the data into training and test sets (80%/20%)
    input_train, input_test, output_train, output_test = train_test_split(
        nn_input, nn_output, test_size=0.2, random_state=42
    )
    
    # Create a neural network to learn the Henon map dynamics
    net = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_train.shape[1],)),  # Input layer (x,y)
        tf.keras.layers.Dense(64, activation="relu"),   # 64 neurons with ReLU activation
        tf.keras.layers.Dense(32, activation="relu"),   # 32 neurons with ReLU activation
        tf.keras.layers.Dense(16, activation="relu"),   # 16 neurons with ReLU activation
        tf.keras.layers.Dense(2, activation="linear")   # 2 outputs (next x, next y)
    ])
    
    # Compile the model with Adam optimizer and MSE loss
    net.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]  # Mean Absolute Error
    )
    
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model with validation split
    n_epochs = 100  # Maximum number of epochs
    history = net.fit(
        input_train, output_train, 
        epochs=n_epochs, 
        batch_size=128,  
        validation_split=0.2,  # 20% of training data used for validation
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model on the test set
    loss = net.evaluate(input_test, output_test)
    print(f"Test loss: {loss[0]:.4f}, MAE: {loss[1]:.4f}")

    # Save the trained model for future use
    net.save('henon_map.keras')  
    
# Test the neural network on a new trajectory
# Generate a random initial point within the basin of attraction
X0 = np.random.uniform(-0.75, 0.75)
Y0 = np.random.uniform(-0.75, 0.2)

# Initialize arrays for predicted trajectory
X_predict = np.zeros(n)
Y_predict = np.zeros(n)
X_predict[0] = X0
Y_predict[0] = Y0

# Predict the entire trajectory step-by-step
print("Predicting the test-case series...")
for i in range(1, n):
    # Use the current state to predict the next state
    prediction = net.predict(np.array([[X_predict[i-1], Y_predict[i-1]]]), verbose=0)
    X_predict[i] = prediction[0,0]  # Predicted next x-coordinate
    Y_predict[i] = prediction[0,1]  # Predicted next y-coordinate

# Calculate true trajectory using the Hénon map equations
X_true = np.zeros(n)
Y_true = np.zeros(n)
X_true[0] = X0  # Same initial condition as predicted trajectory
Y_true[0] = Y0 

for i in range(1, n):
    X_true[i] = 1 - a*X_true[i-1]**2 + Y_true[i-1]  # True Henon map x-component
    Y_true[i] = b*X_true[i-1]                        # True Henon map y-component

# Calculate error metrics
mse = np.mean((X_predict - X_true)**2 + (Y_predict - Y_true)**2)
print(f"Mean Squared Error: {mse:.6f}")

# Visualization: Compare first and last g points of trajectories
g = 20  # Number of points to display

plt.figure(figsize=(15, 4))

# First g points - X component
plt.subplot(1, 4, 1)
plt.plot(range(1, g+1), X_predict[:g], '-o', markersize=2, color='black', label='Predicted X')
plt.plot(range(1, g+1), X_true[:g], '-o', markersize=2, color='blue', label='True X')
plt.title('First g points - X')
plt.legend()

# First g points - Y component
plt.subplot(1, 4, 2)   
plt.plot(range(1, g+1), Y_predict[:g], '-o', markersize=2, color='red', label='Predicted Y')
plt.plot(range(1, g+1), Y_true[:g], '-o', markersize=2, color='green', label='True Y')
plt.title('First g points - Y')
plt.legend()

# Last g points - X component
plt.subplot(1, 4, 3)
plt.plot(range(n-g+1, n+1), X_predict[-g:], '-o', markersize=2, color='black', label='Predicted X')
plt.plot(range(n-g+1, n+1), X_true[-g:], '-o', markersize=2, color='blue', label='True X')
plt.title('Last g points - X')
plt.legend()

# Last g points - Y component
plt.subplot(1, 4, 4)   
plt.plot(range(n-g+1, n+1), Y_predict[-g:], '-o', markersize=2, color='red', label='Predicted Y')
plt.plot(range(n-g+1, n+1), Y_true[-g:], '-o', markersize=2, color='green', label='True Y')
plt.title('Last g points - Y')
plt.legend()
plt.tight_layout()
plt.show()

# Display learning curves if model was just trained
if not model_loaded:
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # MAE curves
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.yscale('log')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualization: Compare full predicted vs true Henon map attractors
plt.figure(figsize=(12, 5))

# Predicted attractor
plt.subplot(1, 2, 1)
plt.scatter(X_predict, Y_predict, color='black', s=1, label='Predicted')
plt.title('Predicted Henon Map')
plt.xlabel('X')
plt.ylabel('Y')

# True attractor
plt.subplot(1, 2, 2)
plt.scatter(X_true, Y_true, color='blue', s=1, label='True')
plt.title('True Henon Map')
plt.xlabel('X')
plt.ylabel('Y')
plt.tight_layout()
plt.show()