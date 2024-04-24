import tensorflow as tf
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the diabetes dataset
diabetes_data, diabetes_target= load_diabetes(return_X_y=True)

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
    tf.keras.layers.Dense(1, activation="linear"),  # Output layer with 1 units
])

# Compile the model
net.compile(optimizer="adam", loss="mean_absolute_error", metrics=["mean_absolute_error"])

# Train the model
n_epochs = 100
history = net.fit(X_train, y_train, epochs=n_epochs, batch_size=16) 

y_pred = net.predict(X_test)

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

mae = net.evaluate(X_test, y_test)

#perfor pinv regression
import numpy as np

X_train_pinv = np.linalg.pinv(X_train)
w = X_train_pinv @ y_train
y_pred_pinv = X_test @ w

mae_pinv = np.mean(np.abs(y_test - y_pred_pinv))

print("MaE of neural network: ", mae[0])
print("MaE of pinv regression: ", mae_pinv)
