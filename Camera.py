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
        preprocessed_frame = preprocess_frame(frame)

        # Display the captured frame
        cv2.imshow('Original Frame', frame)
        cv2.imshow("Preprocessed Frame", preprocessed_frame)
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
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)  # Apply CLAHE to the L channel
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Resizing and Scaling
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Region of Interest (ROI) Selection (Not implemented in this example)

    return frame



# Call the function to start capturing the camera feed
capture_camera_feed()
