import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf

# Model parameters for the Henon map
a = 1.4
b = 0.3
n = 1000  # Number of points

"""
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
except:
    # Generate m vectors of the henon map
    # n x m: 100'000 for light training, 1'000'000 for well training
    m = 1000 # Number of vectors

    X = np.zeros((m,n))
    Y = np.zeros((m,n))

    X[:,0] = np.random.uniform(0.75, -0.75, size=m) # This range avoids overflow
    Y[:,0] = np.random.uniform(0.2, -0.75, size=m)

    for i in range(1,n):
        X[:,i] = 1 - a*X[:,i-1]**2 + Y[:,i-1]
        Y[:,i] = b*X[:,i-1]
    
    """
    # Find the overflow points
    overflow = np.where(np.abs(X[:,]) > 100)[0]  # Get indices of overflow

    # Plot the overflow initial points
    plt.scatter(X[overflow, 0], Y[overflow, 0], marker='o', color='black', s=1)
    plt.show()
    """

    nn_input = np.column_stack((X[0,:-1], Y[0,:-1])) # For the first vector
    nn_output = np.column_stack((X[0,1:], Y[0,1:]))

    # Construct the input and output data for the neural network, x and y pairs for each vector
    for i in range(1,m):
        nn_input = np.vstack((nn_input, np.column_stack((X[i,:-1], Y[i,:-1]))))
        nn_output = np.vstack((nn_output, np.column_stack((X[i,1:], Y[i,1:]))))

    # Split the data into training and test sets
    from sklearn.model_selection import train_test_split
    input_train, input_test, output_train, output_test = train_test_split(nn_input, nn_output, test_size=0.2, random_state=42)

    # Create a neural network
    net = tf.keras.models.Sequential([
        tf.keras.Input(shape=(input_train.shape[1],)),  # Input layer with shape of the input data
        tf.keras.layers.Dense(10, activation="sigmoid"),  # Hidden layer with 10 units and relu activation
        tf.keras.layers.Dense(5, activation="relu"),  # Hidden layer with 5 units and relu activation
        tf.keras.layers.Dense(2, activation="linear")  # Output layer with 2s units
    ])

    # Compile the model
    net.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    n_epochs = 32
    history = net.fit(input_train, output_train, epochs=n_epochs, batch_size=64)

    # Evaluate the model on the test set
    loss = net.evaluate(input_test, output_test)
    print(f"Test loss: {loss:.2f}")
    
    # Save the model
    net.save('henon_map.keras')

# Generate new data using the trained model
# Generate the initial point
X0 = np.random.uniform(0.75, -0.75)
Y0 = np.random.uniform(0.2, -0.75)

X_predict = np.zeros(n)
Y_predict = np.zeros(n)
X_predict[0] = X0
Y_predict[0] = Y0  

# Predict the next point
print("Predicting the test-case series...")
for i in range(1, n):

    prediction = net.predict(np.array([[X_predict[i-1], Y_predict[i-1]]]), verbose=0)
    X_predict[i] = prediction[0,0]
    Y_predict[i] = prediction[0,1]
    
X_true = np.zeros(n)
Y_true = np.zeros(n)
X_true[0] = X0
Y_true[0] = Y0 

# Calculate true value  
for i in range(1, n):
    X_true[i] = 1 - a*X_true[i-1]**2 + Y_true[i-1]
    Y_true[i] = b*X_true[i-1]
    
# Plot the first and last g points   
g=20

plt.subplot(1, 4, 1)
plt.plot(range(1, g+1), X_predict[:g], '-o', markersize=1, color='black')
plt.plot(range(1, g+1), X_true[:g], '-o', markersize=1, color='blue')

plt.subplot(1, 4, 2)   
plt.plot(range(1, g+1), Y_predict[:g], '-o', markersize=1, color='red')
plt.plot(range(1, g+1), Y_true[:g], '-o', markersize=1, color='green')

plt.subplot(1, 4, 3)
plt.plot(range(1, g+1), X_predict[-g:], '-o', markersize=1, color='black')
plt.plot(range(1, g+1), X_true[-g:], '-o', markersize=1, color='blue')

plt.subplot(1, 4, 4)   
plt.plot(range(1, g+1), Y_predict[-g:], '-o', markersize=1, color='red')
plt.plot(range(1, g+1), Y_true[-g:], '-o', markersize=1, color='green')
plt.show()

# Plot the loss function over the epochs
plt.plot(history.history['loss'])
plt.title('Model mean squared error')
plt.yscale('log')
plt.ylabel('mean squared error')
plt.xlabel('Epoch')
plt.xticks(range(0, n_epochs))
plt.show()

# Plot the predicted Henon map
plt.scatter(X_predict, Y_predict, color='black', s=1)
plt.show()

