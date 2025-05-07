"""Diabetes Progression Prediction using Neural Networks and Pseudo-Inverse Regression.

This code performs regression analysis on the diabetes dataset by:
1. Loading the standard diabetes dataset 
2. Training a multi-layer neural network with SELU activations
3. Comparing the neural network's performance with pseudo-inverse regression
4. Visualizing prediction accuracy and training progress

The program demonstrates how a neural network can predict disease progression
and compares its performance against a simpler linear approach using
pseudo-inverse regression.

Data Structure
-------------
- X: Features from the diabetes dataset
- y: Disease progression target values
- history: Training metrics recorded during neural network training
- y_pred: Neural network predictions on test data
- y_pred_pinv: Pseudo-inverse regression predictions on test data
"""
import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Load the diabetes dataset
diabetes_data, diabetes_target = load_diabetes(return_X_y=True)

X = diabetes_data
y = diabetes_target

# Split the dataset into training and test sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
net = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
    tf.keras.layers.Dense(20, activation="selu"),  # Hidden layer with 20 units and selu activation
    tf.keras.layers.Dense(10, activation="selu"),  # Hidden layer with 10 units and selu activation
    tf.keras.layers.Dense(5, activation="selu"),  # Hidden layer with 5 units and selu activation
    tf.keras.layers.Dense(1, activation="linear"),  # Output layer with 1 unit, regression output
])

# Compile the model
net.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mean_absolute_error"])

# Train the model
n_epochs = 100
history = net.fit(X_train, y_train, epochs=n_epochs, batch_size=16) 

# Generate predictions using the trained model
y_pred = net.predict(X_test)

# Create visualization of results
fig, ax = plt.subplots(1,2)
ax[0].plot(y_test, y_pred, 'o')
ax[0].plot([0, 350], [0, 350], 'r-')
ax[0].set_xlabel('True value')  
ax[0].set_ylabel('Predicted value')  
ax[0].set_title('True vs predicted value')

ax[1].plot(history.history['mean_absolute_error'])
ax[1].set_title('Model accuracy')
ax[1].set_ylabel('Mean absolute error') 
ax[1].set_xlabel('Epoch')
plt.show()

# Evaluate neural network performance on test data
mae = net.evaluate(X_test, y_test)

# Perform pseudo-inverse regression for comparison
X_train_pinv = np.linalg.pinv(X_train)
w = X_train_pinv @ y_train
y_pred_pinv = X_test @ w

# Calculate mean absolute error for pseudo-inverse regression
mae_pinv = np.mean(np.abs(y_test - y_pred_pinv))

# Print performance comparison
print("MAE of neural network: ", mae[-1])
print("MAE of pinv regression: ", mae_pinv)