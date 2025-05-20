"""Video Frame Prediction using LSTM with SVD Dimensionality Reduction.
This code performs future frame prediction on a video sequence by:

Loading video frames from a numpy array file
Applying SVD for dimensionality reduction
Creating a sequence prediction dataset (frame t → frame t+1)
Training an LSTM neural network to predict future frames
Visualizing and animating prediction results against ground truth

The program displays training loss curves, side-by-side comparisons of predicted
versus actual frames, and an animation of the prediction sequence. SVD is used for
dimensionality reduction before training the neural network, creating a more
efficient representation of the video data.
Data Structure

frames_array: Original video frames stored in numpy array
phi: Truncated spatial modes from SVD (first r columns of U)
a: Temporal coefficients in the reduced basis
nn_input/nn_output: Sequential frame pairs for training
predicted_frames: Neural network predictions transformed back to image space
true_frames: Ground truth frames for comparison
"""
import os
# Disable OneDNN optimization for reproducibility of TensorFlow results
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  
# Change to directory containing the dataset
os.chdir('ESE-10-Reduced Order Models')

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the frames array from the file
frames_array = np.load('frames_array.npy')

# Flatten the frames array: reshape into a matrix where each column is a flattened frame
# Transpose to get pixels as rows and time steps as columns
flattened_frames = frames_array.reshape(frames_array.shape[0], -1).T
print("Flattened frames shape:", flattened_frames.shape)

# Perform SVD on the flattened frames
# U: spatial modes (pixel patterns), S: singular values (importance), Vt: temporal modes
U, S, Vt = np.linalg.svd(flattened_frames, full_matrices=False)
print("Shapes of U, S, Vt:", U.shape, S.shape, Vt.shape)

# Truncate the SVD to keep only the first r singular values (dimensionality reduction)
# r represents the number of modes to retain
r = 35
phi = U[:, :r]  # Reduced spatial basis

# Project original data onto reduced basis to get temporal coefficients
a = phi.T @ flattened_frames  # Matrix multiplication: a contains the low-dimensional representation
print("a shape:", a.shape)

# Create input-output pairs for sequence prediction
# Input: coefficients at time t, Output: coefficients at time t+1
nn_input = a[:, :-1]  # All columns except the last one
nn_output = a[:, 1:]  # All columns except the first one

print("nn_input shape:", nn_input.shape)

# Split data into training and testing sets (keeping time order with shuffle=False)
input_train, input_test, output_train, output_test = train_test_split(
    nn_input.T, nn_output.T, test_size=0.1, shuffle=False
)

# Standardize the data (mean=0, std=1) for better neural network performance
scaler = StandardScaler()
input_train = scaler.fit_transform(input_train)  # Learn parameters from training data
input_test = scaler.transform(input_test)  # Apply same transformation to test data
output_train = scaler.fit_transform(output_train)  # Scale outputs separately to avoid data leakage
# No need to transform output_test, as it is used only for comparison, not training

print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)    

import tensorflow as tf

# Define the neural network model architecture
model = tf.keras.models.Sequential([
    # Reshape input for LSTM layer
    tf.keras.layers.Reshape((1, r), input_shape=(r,)),  
    # LSTM layer with 128 hidden units
    tf.keras.layers.LSTM(128, return_sequences=False),
    # Dropout for regularization (prevents overfitting)
    tf.keras.layers.Dropout(0.2),
    # Output layer with linear activation (regression problem)
    tf.keras.layers.Dense(r, activation='linear')
])

# Configure model training parameters
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss',  # Monitor training loss
    patience=30,  # Stop if no improvement for 30 epochs
    restore_best_weights=True,  # Keep best weights
    verbose=1  # Print progress
)

# Train the model
history = model.fit(
    input_train, output_train,
    validation_split=0.1,  # 10% of training data for validation
    epochs=1000,  # Maximum number of epochs
    batch_size=16,  # Number of samples per gradient update
    callbacks=[early_stopping],  
    verbose=1  # Print progress
)

# Plot the training and validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.semilogy()  # Logarithmic y-axis for better visualization
plt.legend()
plt.show()

# Generate predictions on test data
prediction = model.predict(input_test)
print("prediction shape:", prediction.shape)

# Convert predictions from standardized scale back to original scale
prediction = scaler.inverse_transform(prediction)

# Convert predicted coefficients back to image space
# Project from low-dimensional space to pixel space using the spatial modes
predicted_frames = phi @ prediction.T
# Reshape to original frame dimensions
predicted_frames = predicted_frames.T.reshape(-1, frames_array.shape[1], frames_array.shape[2])

# Compare prediction with ground truth for the first frame
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(predicted_frames[0], cmap='gray')
plt.title('Prediction')

# Generate ground truth frames for comparison
true_frames = phi @ output_test.T
true_frames = true_frames.T.reshape(-1, frames_array.shape[1], frames_array.shape[2])
print("True frames shape:", true_frames.shape)
plt.subplot(1, 2, 2)
plt.imshow(true_frames[0], cmap='gray')
plt.title('True Frame')

plt.show()

# Animate side-by-side comparison of predictions and ground truth
fig, ax = plt.subplots(1, 2)

plt.ion()  # Turn on interactive mode for animation
for i in range(len(predicted_frames)):
    # Left plot: predicted frame
    ax[0].imshow(predicted_frames[i], label='Prediction', cmap='gray')
    # Right plot: ground truth frame
    ax[1].imshow(true_frames[i], label='Ground Truth', cmap='gray')
    # Hide axes for better visualization
    ax[0].axis('off')
    ax[1].axis('off')
    
    plt.show()
    plt.pause(0.1)  # Pause between frames
    
    # Clear the axes for the next frame
    ax[0].cla()
    ax[1].cla()
    
plt.ioff()  # Turn off interactive mode