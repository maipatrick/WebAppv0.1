import cv2
from sports2d import Sports2D

# Path to the video file
video_path = "C:/Users/adpatrick/OneDrive - nih.no/Desktop/WebApp1080sync/WebAppv0.1/fp22_m505_left 2.MOV"

# Load the video file
cap = cv2.VideoCapture(video_path)

# Initialize Sports2D
sports2d_model = Sports2D()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation using Sports2D
    results = sports2d_model.process(frame)

    # Draw the pose annotation on the frame
    sports2d_model.draw_landmarks(frame, results)

    # Display the frame
    cv2.imshow('Sports2D Pose Estimation', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()