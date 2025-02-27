import cv2

def resize_video(input_file, output_file, width, height):
    """
    Resize a video to the specified dimensions using OpenCV.

    Parameters:
        input_file (str): Path to the input video file.
        output_file (str): Path to save the output video file.
        width (int): The desired width of the output video.
        height (int): The desired height of the output video.
    """
    try:
        cap = cv2.VideoCapture(input_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (width, height))
            out.write(resized_frame)

        cap.release()
        out.release()
        print(f"Video resized and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Example usage
    input_file = "karman.mp4"
    output_file = 'output.mp4'
    
    cap = cv2.VideoCapture(input_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Video Dimensions: {} x {}".format(width, height))
    width = 256
    height = 128

    resize_video(input_file, output_file, width, height)
