import cv2
print(cv2.__version__)
cap = cv2.VideoCapture(0)
print(cap.isOpened())