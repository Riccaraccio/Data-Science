import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt

# Load the dataset
print("Loading dataset...")
dataset = fetch_lfw_people(min_faces_per_person=70) 
print("Dataset loaded.")

# Split the dataset
X = dataset.data
y = dataset.target

"""
# Print the shape of the data and look at a sample image
print(X.shape)

index = 1
plt.imshow(X[index].reshape(62, 47), cmap='gray')
plt.title(dataset.target_names[y[index]])
plt.show()
"""

# transform the target variable to a one-hot encoded vector
y = tf.keras.utils.to_categorical(y, num_classes=len(dataset.target_names))

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a neural network 
net = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
    tf.keras.layers.Dense(128, activation="relu"),  # Hidden layer with 128 units and relu activation
    tf.keras.layers.Dense(64, activation="relu"),  # Hidden layer with 64 units and relu activation
    tf.keras.layers.Dense(len(dataset.target_names), activation="softmax"),  # Output layer with units equal to the number of different people
])

# Compile the model
net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
net.fit(X_train, y_train, epochs=15, batch_size=16)

# Evaluate the model on the test set
loss, accuracy = net.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")


# Create a convolutional neural network
net2 = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
    tf.keras.layers.Reshape((62, 47, 1)),  # Reshape layer
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation="relu"),  # Convolutional layer with 32 filters and relu activation
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # Max pooling layer with pool size of 2 and stride of 2
    tf.keras.layers.Flatten(),  # Flatten layer to convert the 2D output to 1D
    tf.keras.layers.Dense(len(dataset.target_names), activation="softmax"),  # Output layer with 1 units
])

"""
# Add dropout layers to the model

net2 = tf.keras.models.Sequential([
    tf.keras.layers.Reshape((62, 47, 1), input_shape=(X_train.shape[1],)),  # Reshape input to 2D image shape (height, width, channels)
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation="relu"),  # Convolutional layer with 16 filters and relu activation
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),  # Max pooling layer with pool size of 2
    tf.keras.layers.Dropout(0.25),  # Dropout layer with dropout rate of 25%
    tf.keras.layers.Flatten(),  # Flatten layer
    tf.keras.layers.Dense(128, activation="relu"),  # Dense hidden layer with 128 units and relu activation
    tf.keras.layers.Dropout(0.5),  # Dropout layer with dropout rate of 50%
    tf.keras.layers.Dense(len(dataset.target_names), activation="softmax")  # Output layer with softmax activation
])
"""

# Compile the model
net2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
net2.fit(X_train, y_train, epochs=15, batch_size=16) 

# Evaluate the model on the test set
loss, accuracy = net2.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

import numpy as np

# Get the output of the convolutional and pooling layers
conv_output = tf.keras.Model(inputs=net2.inputs, outputs=net2.layers[1].output)(X_train)
pool_output = tf.keras.Model(inputs=net2.inputs, outputs=net2.layers[2].output)(X_train)

# Choose the first image from the training set
index = 1

# Plot the original, convoluted, and pooled images
plt.subplot(1, 3, 1)
plt.imshow(X_train[index].reshape(62, 47), cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(conv_output[index, :, :, 0], cmap='inferno')
plt.title('Convoluted Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pool_output[index, :, :, 0], cmap='inferno')
plt.title('Pooled Image')
plt.axis('off')

plt.show()
