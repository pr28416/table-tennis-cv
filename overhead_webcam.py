import cv2 as cv
import matplotlib.pyplot as plt

capture = cv.VideoCapture(0)
didFind, frame = capture.read()
if didFind:
    plt.imshow(frame)
    plt.show()
else:
    print("Something went wrong while trying to access the webcam.")