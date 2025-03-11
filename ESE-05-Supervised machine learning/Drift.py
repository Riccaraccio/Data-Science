import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("ESE-05-Supervised machine learning/processed_data/merged_batches.pkl", "rb") as f:
    data = pickle.load(f)

# Extract data from the first batch
batch1 = data.where(data["batch_id"] == 1).dropna()

batch1_y = batch1["gas_class"].to_numpy()
batch1_X = batch1[batch1.columns[:-5]].to_numpy()


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(batch1_X, batch1_y, random_state=0, test_size=0.1)

# Train a random forest classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=0)
model.fit(X_train, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
accuracy = [accuracy_score(y_test, y_pred)]

for i in range(2, 10):
    
    # Extract data from the batch
    batch = data.where(data["batch_id"] == i).dropna()

    batch_y = batch["gas_class"].to_numpy()
    batch_X = batch[batch.columns[:-5]].to_numpy()

    # Predict the gas class
    y_pred = model.predict(batch_X)

    # Evaluate the model
    accuracy = np.append(accuracy, accuracy_score(batch_y, y_pred))

# Plot the accuracy of the model over time
plt.figure(figsize=(10, 6))
plt.plot(range(1, 10), accuracy, marker='o')
plt.xlabel('Batch number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.title('Random Forest Accuracy over Time')
plt.grid(True)
plt.show()
