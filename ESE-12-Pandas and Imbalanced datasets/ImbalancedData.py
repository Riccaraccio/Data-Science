import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf

# Load the data from the Excel file
df = pd.read_excel('satellites.xlsx', index_col=0)

# Separate the target column from the data
target = df["Target"].values
data = df.drop("Target", axis=1)

# Convert target values to binary (1 for "Anomaly", 0 "Normal")
target = np.array([1 if t == 'Anomaly' else 0 for t in target])

# Count the number of samples in each class
neg, pos = np.bincount(target)
total = neg + pos

# Print the class distribution
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# Split the data into training and testing sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.2, random_state=0)

# Ensure that the training set has balanced classes
train_prob = np.mean(target_train)
test_prob = np.mean(target_test)

print('Training set positive class probability: {:.2f}%'.format(100 * train_prob))
print('Testing set positive class probability: {:.2f}%'.format(100 * test_prob))

# Normalize each feature
scaler = StandardScaler()

data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

# Look at the data distribution
sns.jointplot(x=data_train[:, 0], y=data_train[:, 1], kind='hex', xlim=(-5, 5), ylim=(-5, 5))
plt.show()

# Define the model and metrics
METRICS = [
    keras.metrics.BinaryCrossentropy(name='cross_entropy'),  # same as model's loss
    keras.metrics.MeanSquaredError(name='Brier_score'),
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),   
]

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)  # convert to tensor
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=(data_train.shape[1],)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

EPOCHS = 100
BATCH_SIZE = 1024  # large batch size to ensure that each batch has a balanced class distribution

# Stop training when a monitored metric has stopped improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='min',  # use 'min' for loss
    restore_best_weights=True)

model = make_model()
model.summary()

# Reshape target arrays to match the model output
target_train = target_train.reshape(-1, 1)
target_test = target_test.reshape(-1, 1)

# Test the model prediction on the training data
print("Initial model predictions: ", model.predict(data_train[:10]))

# Set the correct initial bias
initial_bias = np.log([pos / neg])
model_bias = make_model(output_bias=initial_bias)
print("Model with bias predictions: ", model_bias.predict(data_train[:10]))

# Train both models
no_bias_history = model.fit(data_train, 
                            target_train, 
                            validation_data=(data_test, target_test),
                            batch_size=BATCH_SIZE, 
                            epochs=EPOCHS,
                            verbose=0,
                            callbacks=[early_stopping])

bias_history = model_bias.fit(data_train, 
                              target_train, 
                              validation_data=(data_test, target_test),
                              batch_size=BATCH_SIZE, 
                              epochs=EPOCHS,
                              verbose=0,
                              callbacks=[early_stopping])

# Plot training and validation loss
plt.plot(no_bias_history.history['loss'], label='Training Loss', color='b')
plt.plot(no_bias_history.history['val_loss'], label='Validation Loss', linestyle='--', color='b')
plt.plot(bias_history.history['loss'], label='Training Loss with Bias', color='r')
plt.plot(bias_history.history['val_loss'], label='Validation Loss with Bias', linestyle='--', color='r')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')
plt.show()


from sklearn.metrics import confusion_matrix
predictions = model.predict(data_test)
predictions_bias = model_bias.predict(data_test)

# tresholding the predictions
predictions = (predictions > 0.5).astype(int)
predictions_bias = (predictions_bias > 0.5).astype(int)

# Plot the confusion matrices for the test data
plt.subplot(1, 2, 1)
plt.title('Confusion Matrix - Model without Bias')
cm = confusion_matrix(target_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.subplot(1, 2, 2)
plt.title('Confusion Matrix - Model with Bias')
cm = confusion_matrix(target_test, predictions_bias)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
