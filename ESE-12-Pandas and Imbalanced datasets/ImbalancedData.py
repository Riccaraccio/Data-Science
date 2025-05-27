"""Satellite Anomaly Detection using Neural Networks with Class Imbalance Handling

This code performs binary classification for satellite anomaly detection by:

Loading and preprocessing satellite telemetry data from Excel files
Handling severe class imbalance through multiple techniques:
  - Proper bias initialization based on class proportions
  - Class weighting to penalize misclassification of minority class
  - Oversampling (SMOTE-like) to balance training data
Creating and training neural network models with different imbalance strategies
Comparing model performance across different approaches
Visualizing training metrics to evaluate convergence and generalization

The program implements four distinct approaches to address class imbalance in satellite
anomaly detection, a critical challenge where normal operations vastly outnumber anomalies.
Each method is evaluated through comprehensive metrics including precision, recall, and
loss curves to determine the most effective approach for this domain.

Data Structure:
df: Original satellite telemetry DataFrame with features and binary target
data: Feature matrix after preprocessing and normalization
target: Binary labels (1=Anomaly, 0=Normal)
Various model histories: Training/validation metrics for performance comparison

Approaches Tested:
1. Baseline model without bias correction
2. Bias-corrected model with proper initialization
3. Class-weighted model with penalty-based loss adjustment
4. Oversampled model with balanced training data
"""

# Import required libraries for data manipulation, machine learning, and visualization
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Change to script directory for relative file access
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations for reproducible results

import pandas as pd
import numpy as np
import seaborn as sns  # Statistical data visualization library
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample  # For oversampling minority class
from tensorflow import keras
import tensorflow as tf

# Load satellite telemetry data from Excel file
# The dataset contains various sensor readings and operational parameters
df = pd.read_excel('satellites.xlsx', index_col=0)

# Separate features from target variable for supervised learning
target = df["Target"].values  # Extract target column containing anomaly labels
data = df.drop("Target", axis=1)  # Remove target to create feature matrix

# Convert categorical target labels to binary numerical format
# Transform string labels ("Anomaly", "Normal") to binary (1, 0) for neural network training
target = np.array([1 if t == 'Anomaly' else 0 for t in target])

# Analyze class distribution to understand the imbalance problem
neg, pos = np.bincount(target)  # Count negative (normal) and positive (anomaly) samples
total = neg + pos

# Display class distribution statistics
# This reveals the severity of class imbalance, critical for choosing appropriate techniques
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

# Split data into train/test sets with stratification to preserve class proportions
# Use small test set (10%) to maximize training data for this challenging problem
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1, random_state=0)

# Further split training data to create validation set for hyperparameter tuning
# Validation set helps prevent overfitting and enables early stopping
data_train, data_val, target_train, target_val = train_test_split(data_train, target_train, test_size=0.2, random_state=0)

# Verify that class proportions are maintained across all splits
# Important check to ensure each subset remains representative of the original distribution
train_prob = np.mean(target_train)
test_prob = np.mean(target_test)
val_prob = np.mean(target_val)

print('Training set positive class probability: {:.2f}%'.format(100 * train_prob))
print('Testing set positive class probability: {:.2f}%'.format(100 * test_prob))
print('Validation set positive class probability: {:.2f}%'.format(100 * val_prob))

# Normalize features using StandardScaler for improved neural network training
# Standardization ensures all features have mean=0 and std=1, preventing dominance by large-scale features
scaler = StandardScaler()

# Fit scaler on training data only to prevent data leakage
data_train = scaler.fit_transform(data_train)
# Apply same transformation to test and validation sets
data_test = scaler.transform(data_test)
data_val = scaler.transform(data_val)

print('Training set shape:', data_train.shape)

# Reshape target arrays to match neural network output dimensions
# Convert 1D arrays to 2D column vectors for compatibility with Keras
target_train = target_train.reshape(-1, 1)
target_test = target_test.reshape(-1, 1)
target_val = target_val.reshape(-1, 1)

# Define comprehensive metrics for model evaluation
# Multiple metrics provide different perspectives on classification performance
METRICS = [
    keras.metrics.BinaryCrossentropy(name='cross_entropy'),  # Primary loss function (same as model's loss)
    keras.metrics.MeanSquaredError(name='Brier_score'),      # Probabilistic accuracy measure
    keras.metrics.TruePositives(name='tp'),                  # Correctly identified anomalies
    keras.metrics.FalsePositives(name='fp'),                 # False alarms
    keras.metrics.TrueNegatives(name='tn'),                  # Correctly identified normal cases
    keras.metrics.FalseNegatives(name='fn'),                 # Missed anomalies (critical for safety)
    keras.metrics.BinaryAccuracy(name='accuracy'),           # Overall classification accuracy
    keras.metrics.Precision(name='precision'),               # Precision = TP/(TP+FP)
    keras.metrics.Recall(name='recall'),                     # Recall = TP/(TP+FN), critical for anomaly detection
]

def make_model(metrics=METRICS, output_bias=None):
    """
    Create a neural network model for binary classification
    
    Parameters:
    metrics - List of evaluation metrics to track during training
    output_bias - Initial bias for output layer to handle class imbalance
    
    Returns:
    model - Compiled Keras sequential model ready for training
    """
    # Convert bias value to Keras initializer if provided
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)  # Convert to tensor format
    
    # Build sequential neural network architecture
    model = keras.Sequential([
        keras.layers.InputLayer(shape=(data_train.shape[1],)),  # Input layer matching feature dimensions
        keras.layers.Dense(16, activation='relu'),               # Hidden layer with ReLU activation
        keras.layers.Dropout(0.5),                              # Dropout for regularization (50% rate)
        keras.layers.Dense(1, activation='sigmoid',              # Output layer for binary classification
                          bias_initializer=output_bias),         # Custom bias initialization if specified
    ])

    # Compile model with Adam optimizer and binary crossentropy loss
    # Adam optimizer adapts learning rates for each parameter individually
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

# Training hyperparameters optimized for class imbalance scenarios
EPOCHS = 200      # Maximum training epochs
BATCH_SIZE = 1024 # Large batch size helps ensure balanced representation in each batch

# Early stopping callback to prevent overfitting
# Monitors validation loss and stops training when improvement plateaus
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',      # Metric to monitor for improvement
    verbose=1,               # Print when stopping occurs
    patience=10,             # Number of epochs to wait before stopping
    mode='min',              # Stop when monitored metric stops decreasing (use 'min' for loss)
    restore_best_weights=True) # Restore weights from best epoch

# APPROACH 1: Baseline model without bias correction
model = make_model()

# Test initial predictions to observe bias toward majority class
# Without proper initialization, model likely predicts mostly negative (normal) cases
print("Initial model predictions: ", model.predict(data_train[:10]))

# APPROACH 2: Model with proper bias initialization
# Calculate optimal initial bias based on class frequencies
# This helps the model start with realistic probability estimates
initial_bias = np.log([pos / neg])  # Log-odds ratio of positive to negative class
model_bias = make_model(output_bias=initial_bias)

# Verify that bias initialization produces more realistic initial predictions
print("Model with bias predictions: ", model_bias.predict(data_train[:10]))

# Train baseline model without bias correction
print("Training model without bias...")
no_bias_history = model.fit(data_train, 
                            target_train, 
                            validation_data=(data_val, target_val),
                            batch_size=BATCH_SIZE, 
                            epochs=EPOCHS,
                            verbose=0,                    # Suppress epoch-by-epoch output
                            callbacks=[early_stopping])

# Train bias-corrected model
print("Training model with bias...")
bias_history = model_bias.fit(data_train, 
                              target_train, 
                              validation_data=(data_val, target_val),
                              batch_size=BATCH_SIZE, 
                              epochs=EPOCHS,
                              verbose=0,
                              callbacks=[early_stopping])

# Visualize training progress: Loss curves
# Compare convergence behavior between baseline and bias-corrected models
plt.plot(no_bias_history.history['loss'], label='Training Loss', color='b')
plt.plot(no_bias_history.history['val_loss'], label='Validation Loss', linestyle='--', color='b')
plt.plot(bias_history.history['loss'], label='Training Loss with Bias', color='r')
plt.plot(bias_history.history['val_loss'], label='Validation Loss with Bias', linestyle='--', color='r')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')  # Log scale better shows convergence behavior
plt.show()

# Visualize training progress: Precision curves
# Precision is critical for anomaly detection to minimize false alarms
plt.plot(no_bias_history.history['precision'], label='Training Precision', color='b')
plt.plot(no_bias_history.history['val_precision'], label='Validation Precision', linestyle='--', color='b')
plt.plot(bias_history.history['precision'], label='Training Precision with Bias', color='r')
plt.plot(bias_history.history['val_precision'], label='Validation Precision with Bias', linestyle='--', color='r')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.ylim(0, 1)  # Set y-axis limits from 0 to 1 for precision
plt.show()

# APPROACH 3: Class weighting to handle imbalance during training
# Calculate class weights inversely proportional to class frequencies
# This penalizes misclassification of rare (anomaly) class more heavily
weights_0 = (1.0 / neg) * (total / 2.0)  # Weight for normal class (class 0)
weights_1 = (1.0 / pos) * (total / 2.0)  # Weight for anomaly class (class 1)
class_weight = {0: weights_0, 1: weights_1}

print("Class weights:", class_weight)

# Train model with class weighting applied during loss computation
weigth_model = make_model()
weigthed_history = weigth_model.fit(data_train,
                                    target_train,
                                    validation_data=(data_val, target_val),
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS,
                                    verbose=0,
                                    callbacks=[early_stopping],
                                    class_weight=class_weight)  # Apply class weights during training

# APPROACH 4: Oversampling the minority class (Data Augmentation)
# Import resampling utility for minority class augmentation
from sklearn.utils import resample

# Combine training data and target for resampling operations
# Create DataFrame from normalized training data for easier manipulation
train_data = pd.DataFrame(data_train, columns=data.columns)
# Add target column to enable class-based filtering
train_data['Target'] = target_train

# Separate majority and minority class samples for individual processing
majority = train_data[train_data['Target'] == 0]  # Normal operations (majority class)
minority = train_data[train_data['Target'] == 1]  # Anomalies (minority class)

# Oversample minority class to match majority class size
# This balances the dataset but may lead to overfitting on minority class patterns
minority_upsampled = resample(minority, 
                              replace=True,           # Sample with replacement (bootstrapping)
                              n_samples=len(majority), # Match size of majority class
                              random_state=0)         # Ensure reproducible results

# Combine majority class with upsampled minority class to create balanced training set
train_upsampled = pd.concat([majority, minority_upsampled])

# Extract features and target from the resampled DataFrame
# Separate features and target back into arrays for model training
data_train = train_upsampled.drop('Target', axis=1).values
target_train = train_upsampled['Target'].values

# Verify that oversampling achieved the desired class balance
neg_upsampled, pos_upsampled = np.bincount(target_train)
total_upsampled = neg_upsampled + pos_upsampled
print('Upsampled Training set:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total_upsampled, pos_upsampled, 100 * pos_upsampled / total_upsampled)) 

# Reshape target array to match model input requirements
target_train = target_train.reshape(-1, 1)

# Display final dimensions of training data after upsampling
print(data_train.shape, target_train.shape)

# Train a new model on the upsampled/balanced training data
upsampled_model = make_model()
upsampled_history = upsampled_model.fit(data_train,
                                         target_train,
                                         validation_data=(data_val, target_val),  # Use original validation set
                                         batch_size=BATCH_SIZE,
                                         epochs=EPOCHS,
                                         verbose=0,
                                         callbacks=[early_stopping])

# Compare performance of class weighting vs oversampling approaches
# Both methods address class imbalance but through different mechanisms

# Plot training and validation loss for comparison between upsampled and weighted models
plt.plot(upsampled_history.history['loss'], label='Training Loss (Upsampled)', color='g')
plt.plot(upsampled_history.history['val_loss'], label='Validation Loss (Upsampled)', linestyle='--', color='g')
plt.plot(weigth_model.history.history['loss'], label='Training Loss (Weighted)', color='m')
plt.plot(weigth_model.history.history['val_loss'], label='Validation Loss (Weighted)', linestyle='--', color='m')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.yscale('log')  # Log scale for better visualization of convergence patterns
plt.show()

# Plot training and validation precision for comparison between upsampled and weighted models
# Precision is especially important in anomaly detection to minimize false positives
plt.plot(upsampled_history.history['precision'], label='Training Precision (Upsampled)', color='g')
plt.plot(upsampled_history.history['val_precision'], label='Validation Precision (Upsampled)', linestyle='--', color='g')
plt.plot(weigth_model.history.history['precision'], label='Training Precision (Weighted)', color='m')
plt.plot(weigth_model.history.history['val_precision'], label='Validation Precision (Weighted)', linestyle='--', color='m')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.ylim(0, 1)  # Constrain y-axis to valid precision range
plt.show()

# Summary of approaches implemented:
# 1. Baseline: Standard neural network without imbalance handling
# 2. Bias initialization: Proper starting weights based on class frequencies  
# 3. Class weighting: Penalty-based approach during loss computation
# 4. Oversampling: Data augmentation to balance training set
# Each approach offers different trade-offs between precision, recall, and computational efficiency