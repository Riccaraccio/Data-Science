import cv2
import numpy as np

# Path to your video file
video_path = 'output.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize an empty list to store the frames
frames = []

# Loop through the video frames and store them in the list
while cap.isOpened():
    ret, frame = cap.read()

    # If there are no more frames to read, break the loop
    if not ret:
        break

    # Append the frame to the list
    frames.append(frame)

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Convert the list of frames to a NumPy array
frames_array = np.array(frames)
print(f"Frames array shape: {frames_array.shape}")
output_file = 'frames_array_color.npy'
np.save(output_file, frames_array)

#converting the frames to grayscale
frames_array = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames_array])
print(f"Frames array shape: {frames_array.shape}")

# Save the frames array to a file
output_file = 'frames_array.npy'
np.save(output_file, frames_array)
print(f"Frames array saved as {output_file}", frames_array.shape)

