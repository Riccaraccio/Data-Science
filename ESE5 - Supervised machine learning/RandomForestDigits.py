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
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])  # Add a subplot to the figure
    ax.imshow(digits[i].reshape(8,8), cmap="Greys")  # Display the image
plt.show()  # Show the figure

# Split the dataset into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits, labels, random_state=0)

# Create a random forest classifier with 1000 trees
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain, Ytrain)  # Train the model
ypred = model.predict(Xtest)  # Make predictions on the test set

import seaborn as sns
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
mat = confusion_matrix(Ytest, ypred)

# Plot the confusion matrix as a heatmap
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()  # Show the heatmap
