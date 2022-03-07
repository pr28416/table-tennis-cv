import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    if not isTrue: break

    blur = cv.GaussianBlur(frame, (7,7), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    # table_lower_bound = np.array([47,34,40], 'uint8')
    # table_lower_bound = np.array([38,26,29], 'uint8')
    # table_upper_bound = np.array([170,139,133], 'uint8')
    # table_upper_bound = np.array([170,130,130], 'uint8')
    table_lower_bound = np.array([160,160,160], 'uint8')
    table_upper_bound = np.array([255,255,255], 'uint8')

    mask = cv.inRange(blur, table_lower_bound, table_upper_bound)
    result = cv.bitwise_and(frame, frame, mask=mask)

    # cv.imshow('mask', mask)
    cv.imshow('canny', cv.bilateralFilter(frame, ))

    if cv.waitKey(1) == ord('q'): break