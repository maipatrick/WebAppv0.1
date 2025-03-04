import cv2
import numpy as np
import pypylon.pylon as pylon
import os
import time

# Set image output parameters
output_dir1 = "camera1_images"
output_dir2 = "camera2_images"

# Create directories if they don't exist
os.makedirs(output_dir1, exist_ok=True)
os.makedirs(output_dir2, exist_ok=True)

# Get all connected cameras
tl_factory = pylon.TlFactory.GetInstance()
devices = tl_factory.EnumerateDevices()

if len(devices) < 2:
    print("At least two cameras are required!")
    exit()

# Open cameras
camera1 = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
camera2 = pylon.InstantCamera(tl_factory.CreateDevice(devices[1]))

try:
    # Start grabbing
    camera1.Open()
    camera2.Open()

    camera1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    camera2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    print("Recording...")

    frame_counter = 0
    start_time = time.time()

    while camera1.IsGrabbing() and camera2.IsGrabbing():
        grab1 = camera1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grab2 = camera2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab1.GrabSucceeded() and grab2.GrabSucceeded():
            img1 = grab1.Array
            img2 = grab2.Array

            # Convert grayscale to color
            if len(img1.shape) == 2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            # Save frames as images
            cv2.imwrite(os.path.join(output_dir1, f"frame_{frame_counter:04d}.png"), img1)
            cv2.imwrite(os.path.join(output_dir2, f"frame_{frame_counter:04d}.png"), img2)

            # Display live preview
            cv2.imshow("Camera 1", img1)
            cv2.imshow("Camera 2", img2)

            # Debugging statements
            # print(f"Frame {frame_counter} from Camera 1 saved as image.")
            # print(f"Frame {frame_counter} from Camera 2 saved as image.")

            frame_counter += 1

        grab1.Release()
        grab2.Release()

        # Exit after 10 seconds
        if time.time() - start_time > 10:
            break

finally:
    # Release resources
    camera1.Close()
    camera2.Close()
    cv2.destroyAllWindows()

    print("Recording stopped and images saved.")