"""Handwritten Digit Classification using Random Forest.

This code performs digit recognition on the MNIST dataset by:
1. Loading the digits dataset (8x8 pixel images)
2. Visualizing sample digit images in a grid
3. Training a Random Forest Classifier with 1000 estimators
4. Evaluating model performance with a confusion matrix

The program displays a visualization of sample digit images and produces
a confusion matrix heatmap showing classification performance. Random Forest
is used as an ensemble method to achieve high accuracy in digit recognition.

Data Structure
-------------
- digits: Flattened 8x8 pixel images (64 features per sample)
- labels: Integer class labels (0-9) for each digit image
- mat: Confusion matrix showing prediction accuracy for each digit class
"""
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the digits dataset
digits, labels = load_digits(return_X_y=True)

# Set up the figure
fig = plt.figure()  # Create a new figure

# Plot the digits: each image is 8x8 pixels
for i in range(64):
   ax = fig.add_subplot(8, 8, i + 1)  # Add a subplot to the figure
   ax.axis("off")  # Turn off the axis
   ax.imshow(digits[i].reshape(8, 8), cmap="Greys")  # Display the image
plt.show()  # Show the figure

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(digits, labels, random_state=0)

# Create a random forest classifier with 1000 trees
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, Y_train)  # Train the model
ypred = model.predict(X_test)  # Make predictions on the test set

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
mat = confusion_matrix(Y_test, ypred)

# Plot the confusion matrix as a heatmap
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()  # Show the heatmap