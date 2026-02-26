"""Face Recognition with Neural Networks using TensorFlow.

This code performs face recognition on the LFW (Labeled Faces in the Wild) dataset using:
1. Loading and preprocessing facial image data
2. Training a standard neural network model with fully connected layers
3. Training a convolutional neural network (CNN) model
4. Evaluating and comparing model performance
5. Visualizing CNN feature extraction

The program demonstrates two approaches to face recognition: a simple neural
network and a more advanced CNN. It shows how convolutional layers extract
features from images and how pooling layers reduce dimensionality while
preserving important information.

Data Structure
-------------
- X: Flattened pixel values of face images
- y: Labels indicating the identity of each face
- X_train, X_test: Training and test datasets
- y_train, y_test: One-hot encoded identity labels
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable OneDNN optimization for reproducibility

import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# Load the LFW dataset with faces that appear frequently
print("Loading dataset...")
dataset = fetch_lfw_people(min_faces_per_person=70) 
print("Dataset loaded.")

# Extract features and target variables
X = dataset.data  # Flattened pixel values
y = dataset.target  # Identity labels

"""
# Code for examining the shape of the data and displaying a sample image
# Print the shape of the data and look at a sample image
print(X.shape)

index = 1
plt.imshow(X[index].reshape(62, 47), cmap='gray')
plt.title(dataset.target_names[y[index]])
plt.show()
"""

# Transform the target variable to one-hot encoded vectors
y = tf.keras.utils.to_categorical(y, num_classes=len(dataset.target_names))

# Split the dataset into training and test sets (80%/20%)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a standard neural network with fully connected layers
net = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
    tf.keras.layers.Dense(128, activation="relu"),  # Hidden layer with 128 units and relu activation
    tf.keras.layers.Dense(64, activation="relu"),  # Hidden layer with 64 units and relu activation
    tf.keras.layers.Dense(len(dataset.target_names), activation="softmax"),  # Output layer with units equal to the number of different people
])

# Compile the model with appropriate loss function and optimizer
net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model for 15 epochs with batch size of 16
net.fit(X_train, y_train, epochs=15, batch_size=16)

# Evaluate the standard neural network model on the test set
loss, accuracy = net.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")


# Create a convolutional neural network for improved feature extraction
net2 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
    tf.keras.layers.Reshape((62, 47, 1)),  # Reshape flattened data to 2D image format (height, width, channels)
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation="relu"),  # Convolutional layer to extract features
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # Max pooling layer to reduce dimensionality
    tf.keras.layers.Flatten(),  # Flatten layer to convert the 2D output to 1D
    tf.keras.layers.Dense(len(dataset.target_names), activation="softmax"),  # Output layer for classification
])

"""
# Alternative CNN model with dropout for regularization (commented out)

net2 = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((62, 47, 1), input_shape=(X_train.shape[1],)),  # Reshape input to 2D image shape (height, width, channels)
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation="relu"),  # Convolutional layer with 16 filters and relu activation
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # Max pooling layer with pool size of 2
    tf.keras.layers.Dropout(0.25),  # Dropout layer with dropout rate of 25% to prevent overfitting
    tf.keras.layers.Flatten(),  # Flatten layer to convert 2D features to 1D
    tf.keras.layers.Dense(128, activation="relu"),  # Dense hidden layer with 128 units and relu activation
    tf.keras.layers.Dropout(0.5),  # Dropout layer with dropout rate of 50% to prevent overfitting
    tf.keras.layers.Dense(len(dataset.target_names), activation="softmax")  # Output layer with softmax activation
])
"""

# Compile the CNN model with the same optimizer and loss function
net2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the CNN model with the same hyperparameters as the standard network
net2.fit(X_train, y_train, epochs=15, batch_size=16) 

# Evaluate the CNN model on the test set
loss, accuracy = net2.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Visualize intermediate outputs using a simpler approach with eager execution
# Choose image to visualize
index = 1
sample = X_train[index:index+1]  # Keep batch dimension

# Apply each layer individually to get intermediate outputs
reshaped = net2.layers[0](sample)  # Reshape layer
conv_output = net2.layers[1](reshaped)   # Conv2D layer
pool_output = net2.layers[2](conv_output)  # MaxPooling2D layer

# Plot the original image and the feature maps
plt.figure(figsize=(12, 4))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(sample[0].reshape(62, 47), cmap='gray')  # Original image
plt.title('Original Image')
plt.axis('off')

# Convolutional layer output (first filter)
plt.subplot(1, 3, 2)
plt.imshow(conv_output[0, :, :, 0], cmap='inferno')  # First feature map
plt.title('Convolutional Feature Map')
plt.axis('off')

# Pooling layer output (first filter)
plt.subplot(1, 3, 3)
plt.imshow(pool_output[0, :, :, 0], cmap='inferno')  # Pooled feature map
plt.title('Pooled Feature Map')
plt.axis('off')

plt.show()  # Display the visualization