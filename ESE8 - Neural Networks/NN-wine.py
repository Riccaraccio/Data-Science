import tensorflow as tf
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the wine dataset
wine_data, wine_target = load_wine(return_X_y=True)

X = wine_data
y = wine_target

# Convert y to a one-hot encoded vector
y = tf.keras.utils.to_categorical(y, num_classes=3)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
net = tf.keras.models.Sequential([
    tf.keras.Input(shape=(X_train.shape[1],)),  # Input layer with shape of the input data
    tf.keras.layers.Dense(13, activation="selu"),  # Hidden layer with 13 units and selu activation
    tf.keras.layers.Dense(5, activation="relu"),  # Hidden layer with 5 units and selu activation
    tf.keras.layers.Dense(3, activation="softmax"),  # Output layer with 1 units
])

# Compile the model
net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
net.fit(X_train, y_train, epochs=100, batch_size=2)

# Evaluate the model on the test set
loss, accuracy = net.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

import numpy as np
# Predict the classes for the test set
y_pred = net.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate the confusion matrix
confusion_mat = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes)
print("Confusion Matrix:")
print(confusion_mat)