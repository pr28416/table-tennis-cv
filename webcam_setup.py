import cv2
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import numpy as np

# Get boundary
# img = cv.imread('table.png')
# plt.imshow(img)
# plt.show()
# Top: (260, 260) --> (1024, 260)
# Bottom: (60, 670) --> (1225, 670)
capture = cv.VideoCapture(0)

# orange_lower_bound = (4, 95, 155)
# orange_upper_bound = (80, 191, 255)
# orange_hsv_lo = (10,100,20)
# orange_hsv_up = (40,255,255)

firstFrame = None
iteration = 0

def ballCenter(x, y, w, h):
    return (x+x+w)//2, (y+y+h)//2

def areaThreePoints(a, b, c): # Uses shoelace theorem
    return abs((a[0]*b[1]+b[0]*c[1]+c[0]*a[1])-(a[1]*b[0]+b[1]*c[0]+c[1]*a[0]))/2
def areaFourPoints(a, b, c, d): # Uses shoelace theorem
    return abs((a[0]*b[1]+b[0]*c[1]+c[0]*d[1]+d[0]*a[0])-(a[1]*b[0]+b[1]*c[0]+c[1]*d[0]+d[1]*a[0]))/2

def ballInBounds(x, y, w, h):
    # Alg: Point is inside convex shape if areas of subshapes with ball center and 2 other points add up to form area of entire shape
    center = ballCenter(x,y,w,h)
    x, y, m, pt = 0, 1, 0, 1
    topLeft, topRight, bottomLeft, bottomRight = (240,260), (1040,260), (0,670), (1280,670)
    topLine = (0, topLeft)
    bottomLine = (0, bottomLeft)
    totalArea = areaFourPoints(topLeft, topRight, bottomRight, bottomLeft)
    atp = areaThreePoints
    a,b,c,d = atp(center, topLeft, topRight), atp(center, topRight, bottomRight), atp(center, bottomRight, bottomLeft), atp(center, bottomLeft, topLeft)
    # val1 = round(sum((atp(center, topLeft, topRight),
    #         atp(center, topRight, bottomRight),
    #         atp(center, bottomRight, bottomLeft),
    #         atp(center, bottomLeft, topLeft)))*10000)/10000
    val1 = round(sum((a,b,c,d))*10000)/10000
    val2 = round(totalArea*10000)/10000
    inBounds = val1 == val2
    print(f"inBounds:{inBounds}, ball:{center}, tL:{topLeft}, tR:{topRight}, bL:{bottomLeft}, bR:{bottomRight}, sum:{a}+{b}+{c}+{d}={val1}, area:{val2}")
    return inBounds

# """
while True:
    ret, frame = capture.read()
    if not ret:
        print('something went wrong with trying to access webcam')
        break

    # Make frame gray
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = imutils.resize(frame, width=500)
    # text = "No motion"
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0) # ksize originally (21,21)
    # gray = cv.Canny(gray, 125, 175)

    # if iteration % 8 == 0 or eighthFrame is None:
    if firstFrame is None:
        firstFrame = gray


    frameDelta = cv.absdiff(firstFrame, gray)
    thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

    thresh = cv.dilate(thresh, None, iterations=2)
    contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    text="Ball NOT in bounds"
    for contour in contours:
        if cv.contourArea(contour) < 250: # 500 is the minimum area size
            continue
        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        if ballInBounds(x,y,w,h):
            text="Motion, ball IN bounds"

    cv.putText(frame, f"Status: {text}", (10,60), cv.FONT_HERSHEY_COMPLEX, 1.5, (0,0,255), 2)

    # cv.imshow('mask output', frameDelta)
    cv.imshow('mask output', frame)
    if cv.waitKey(1) == ord('q'): break
    iteration += 1

capture.release()
cv.destroyAllWindows()
# """