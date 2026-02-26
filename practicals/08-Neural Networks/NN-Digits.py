"""Handwritten Digit Classification using Neural Networks.

This code performs multi-class classification on the digits dataset by:
1. Loading the standard digits dataset from scikit-learn
2. Creating a neural network with ReLU activation and softmax output
3. Training the model on handwritten digit images
4. Visualizing the training accuracy over epochs
5. Evaluating classification performance on test data

The program demonstrates how a simple neural network architecture can 
effectively classify handwritten digits into their respective classes (0-9)
using categorical cross-entropy loss.

Data Structure
-------------
- X: Flattened pixel values from digit images
- y: One-hot encoded digit labels (0-9)
- history: Training metrics recorded during model training
- net: Sequential neural network model with dense layers
"""
import tensorflow as tf
from sklearn.datasets import load_digits 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the digits dataset
digits_images, digits_target = load_digits(return_X_y=True)
X = digits_images
y = digits_target

# Convert y to a one-hot encoded vector
y = tf.keras.utils.to_categorical(y, num_classes=10)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
net = tf.keras.models.Sequential([
   tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
   tf.keras.layers.Dense(256, activation="relu"),  # Hidden layer with 256 units and ReLU activation
   tf.keras.layers.Dense(10, activation="softmax"),  # Output layer with 10 units for digit classes
])

# Compile the model
net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
#optimizer is the algorithm to minimize the loss function
#loss is the loss function, loss = function(actual, predicted)
#metrics is the list of metrics to be evaluated by the model during training and testing

#generate summary of the model
net.summary()

# Train the model
n_epochs = 15
history = net.fit(X_train, y_train, epochs=n_epochs, batch_size=32) 
#epochs is the number of times the model is trained on the entire dataset

# Visualize training progress
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.xticks(range(0, n_epochs))
plt.show()

# Evaluate the model on the test set
loss, accuracy = net.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")