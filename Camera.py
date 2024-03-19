import cv2

def capture_camera_feed():
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Preprocess the frame
        # preprocessed_frame = preprocess_frame(frame)

        # Display the captured frame
        cv2.imshow('Camera Feed', preprocess_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


def preprocess_frame(frame):
    # Noise Reduction
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Image Stabilization (Not implemented in this example)

    # Color Correction
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # Convert to LAB color space for better color correction
    lab_planes = cv2.split(frame)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])  # Apply CLAHE to the L channel
    frame = cv2.merge(lab_planes)
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)  # Convert back to BGR color space



    # Resizing and Scaling
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Region of Interest (ROI) Selection (Not implemented in this example)

    return frame

# Example usage
if __name__ == "__main__":
    # Read input frame from a file or camera
    input_frame = cv2.imread("input_frame.jpg")

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(input_frame)

    # Display original and preprocessed frames
    cv2.imshow("Original Frame", input_frame)
    cv2.imshow("Preprocessed Frame", preprocessed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Call the function to start capturing the camera feed
capture_camera_feed()
