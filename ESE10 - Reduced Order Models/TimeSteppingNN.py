import numpy as np
import matplotlib.pyplot as plt 

# Load the frames array from the file
frames_array = np.load('frames_array.npy')

# Flatten the frames array
flattened_frames = frames_array.reshape(frames_array.shape[0], -1).T
print("Flattened frames shape:", flattened_frames.shape)

# Perform SVD on the flattened frames
U, S, Vt = np.linalg.svd(flattened_frames, full_matrices=False)
print("Shapes of U, S, Vt:", U.shape, S.shape, Vt.shape)

r = 35
phi = U[:, :r]

a = phi.T @ flattened_frames
print("a shape:", a.shape)

nn_input = a[:, :-1]
nn_output = a[:, 1:]

print("nn_input shape:", nn_input.shape)

from sklearn.model_selection import train_test_split
input_train, input_test, output_train, output_test = train_test_split(nn_input.T, nn_output.T, test_size=0.1, shuffle=False)

print("input_train shape:", input_train.shape)
print("input_test shape:", input_test.shape)    

import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_train.shape[1],)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(input_train.shape[1], activation='linear'),
])

model.compile(optimizer='adam', loss='mean_squared_error')

tf.keras.utils.plot_model(model, show_shapes=True)

history = model.fit(input_train, output_train, epochs=1000, batch_size=16)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Predict the future
# allocate space for the prediction
prediction = model.predict(input_test)
print("prediction shape:", prediction.shape)

# Visualize the first predicted frame
predicted_frames = phi @ prediction.T
predicted_frames = predicted_frames.T.reshape(-1, frames_array.shape[1], frames_array.shape[2])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(predicted_frames[0], cmap='gray')
plt.title('Prediction')

# Optionally, visualize the corresponding true frame for comparison
true_frames = phi @ output_test.T
true_frames = true_frames.T.reshape(-1, frames_array.shape[1], frames_array.shape[2])
print("True frames shape:", true_frames.shape)
plt.subplot(1, 2, 2)
plt.imshow(true_frames[0], cmap='gray')
plt.title('True Frame')

plt.show()


fig, ax = plt.subplots(1, 2)

plt.ion()
for i in range(len(predicted_frames)):
    ax[0].imshow(predicted_frames[i], label='Prediction', cmap='gray')
    ax[1].imshow(true_frames[i], label='Ground Truth', cmap='gray')
    
    plt.axis('off')
    plt.show()
    plt.pause(0.1)
    
    ax[0].cla()
    ax[1].cla()
    
plt.ioff()


