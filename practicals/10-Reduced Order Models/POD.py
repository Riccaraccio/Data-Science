"""Video Frame Decomposition and Reconstruction using SVD.
This code performs reduced-order modeling on a video sequence by:

Loading and animating frames from a numpy array file
Flattening and centering the frames
Performing Singular Value Decomposition (SVD)
Visualizing spatial and temporal modes
Reconstructing the video using a subset of modes
Calculating and displaying the reconstruction error

The program displays animations of the original video, visualizations of the
dominant spatial modes, plots of temporal mode dynamics and their variance distribution,
and a side-by-side comparison between the original video, reconstructed version, and
pointwise error. SVD is used to extract the most important patterns in the video for
efficient representation and dimensionality reduction.
Data Structure

frames_array: Original video frames stored in numpy array
U, S, Vt: Results of SVD (spatial modes, singular values, temporal modes)
reconstructed_frames: Video reconstructed using limited number of modes
error: Pointwise difference between original and reconstructed frames
"""
import os
# Change to directory containing the dataset
os.chdir('ESE-10-Reduced Order Models')

import numpy as np 
import matplotlib.pyplot as plt

# Load the frames array from the file
frames_array = np.load('frames_array.npy')
print("Frames array shape:", frames_array.shape)  # Display dimensions of video data

# Animate the original frames to visualize the video
plt.ion()  # Turn on interactive mode for real-time animation
for frame in frames_array:
    plt.imshow(frame, cmap='gray')  # Display each frame as grayscale image
    plt.axis('off')  # Hide axis for cleaner visualization
    plt.show()
    plt.pause(0.01)  # Short pause between frames controls animation speed
    plt.clf()  # Clear the figure before showing next frame
plt.ioff()  # Turn off interactive mode

# Flatten the frames array: reshape to 2D matrix for SVD
# Transpose to get pixels as rows and time as columns
flattened_frames = frames_array.reshape(frames_array.shape[0], -1).T

# Calculate the mean frame across all time steps
mean_frame = np.mean(flattened_frames, axis=1, keepdims=True)

# Center the data by subtracting the mean (important for SVD analysis)
flattened_frames_centered = flattened_frames - mean_frame

# Perform SVD decomposition on the flattened frames
# U: spatial modes (pixel patterns), S: singular values, Vt: temporal modes
U, S, Vt = np.linalg.svd(flattened_frames, full_matrices=False)
print("Shapes of U, S, Vt:", U.shape, S.shape, Vt.shape)

# Number of dominant spatial modes to visualize
n = 4

# Create subplot to display the first n spatial modes
fig, axs = plt.subplots(1, n)

for i in range(n):
    # Reshape each column of U (spatial mode) back to image dimensions
    reconstructed_frame = U[:, i].reshape(frames_array.shape[1], frames_array.shape[2])
    
    # Display the spatial mode as an image
    axs[i].imshow(reconstructed_frame, cmap='gray')
    axs[i].axis('off')
    axs[i].set_title(f"U[{i}]")  # Label each mode

plt.show()

# Calculate temporal coefficients: a(t) = S * Vt
# These represent how each spatial mode evolves over time
a = np.diag(S) @ Vt  # Scale temporal modes by their importance (singular values)
print(a.shape)

# Plot temporal evolution of the first four modes
plt.plot(a[0, :], label='mode 1')
plt.plot(a[1, :], label='mode 2')
plt.plot(a[2, :], label='mode 3')
plt.plot(a[3, :], label='mode 4')
plt.legend()
plt.show()

# Visualize the variance captured by each mode
# Higher singular values indicate more important modes
plt.scatter(range(len(S)), S, marker="o", color="black")
plt.xlabel('Mode')
plt.ylabel('Variance')
plt.yscale('log')  # Log scale to better visualize large range of values
plt.show()

# Visualize the second mode behavior in isolation
# Outer product of spatial mode and temporal coefficient
mode2 = U[:, 1].reshape(-1, 1) @ a[1, :].reshape(1, -1)
mode2 = mode2.T.reshape(frames_array.shape)  # Reshape to original dimensions

# Animate the isolated mode to see its pattern over time
plt.ion()
for frame in mode2:
    plt.imshow(frame, cmap='gray')
    plt.axis('off')
    plt.show()
    plt.pause(0.01)
    plt.clf()
plt.ioff()

# Reconstruct the video using only the top n modes
n = 35  # Number of modes used for reconstruction

# Perform truncated reconstruction: U[:,:n] @ S[:n] @ Vt[:n,:]
reconstructed_frames = (U[:, :n] @ np.diag(S[:n]) @ Vt[:n, :])
print("Reconstructed frames shape:", reconstructed_frames.shape)

# Calculate absolute error between original and reconstructed frames
error = np.abs(flattened_frames - reconstructed_frames)

# Reshape error to match original frame dimensions
error = error.T.reshape(frames_array.shape)
print("Error shape:", error.shape)
print("Frames array shape:", frames_array.shape)

# Reshape reconstructed frames to match original dimensions
reconstructed_frames = reconstructed_frames.T.reshape(frames_array.shape)

# Create animation comparing original, reconstruction, and error
plt.ion()
fig, axs = plt.subplots(1, 3)  # 3 panels: original, reconstruction, error

# Loop through all frames to create animation
for i in range(reconstructed_frames.shape[0]):

    # Panel 1: Original video frame
    axs[0].imshow(frames_array[i], cmap='gray')
    axs[0].axis('off')
    axs[0].set_title("Original Video")

    # Panel 2: Reconstructed frame using n modes
    axs[1].imshow(reconstructed_frames[i], cmap='gray')
    axs[1].axis('off')
    axs[1].set_title(f"Reconstructed video using {n} modes")
    
    # Panel 3: Pointwise error visualization
    axs[2].imshow(error[i])
    axs[2].set_title("Error")
    
    plt.axis('off')
    plt.show()
    plt.pause(0.01)  # Control animation speed
    
    # Clear all panels for next frame
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

plt.ioff()  # Turn off interactive mode