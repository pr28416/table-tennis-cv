import cv2
import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import numpy as np

class TableTennisCV():
    def __init__(self, stream=0):
        self.stream = stream
        print("Launched TableTennisCV")
        self.opponentBounds = {
            # 'topLeft': (240,260),
            # 'topRight': (1040,260),
            # 'bottomRight': (1280,670),
            # 'bottomLeft': (0,670)
            'topLeft': (645,570),
            'topRight': (2030,570),
            'bottomRight': (2410,1400),
            'bottomLeft': (200,1400)
        }
        self.playerBounds = {
            'topLeft': (890, 0),
            'topRight': (1820, 0),
            'bottomRight': (1970, 360),
            'bottomLeft': (720, 360)
        }

    def plot(self, image_path):
        img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    def ballCenter(self, x, y, w, h):
        return (x+x+w)//2, (y+y+h)//2

    # def areaThreePoints(self, a, b, c): # Uses shoelace theorem
    #     return abs((a[0]*b[1]+b[0]*c[1]+c[0]*a[1])-(a[1]*b[0]+b[1]*c[0]+c[1]*a[0]))/2
    #
    # def areaFourPoints(self, a, b, c, d): # Uses shoelace theorem
    #     return abs((a[0]*b[1]+b[0]*c[1]+c[0]*d[1]+d[0]*a[0])-(a[1]*b[0]+b[1]*c[0]+c[1]*d[0]+d[1]*a[0]))/2

    def shoelaceArea(self, *args): # Uses shoelace theorem
        p, q, m = 0, 0, len(args)
        for i in range(len(args)):
            p += args[i][0]*args[(i+1)%m][1]
            q += args[i][1]*args[(i+1)%m][0]
        return abs(p-q)/2

    def ball_in_opponent_bounds(self, x, y, w, h):
        return self._ball_in_bounds(x, y, w, h,
                                    self.opponentBounds['topLeft'],
                                    self.opponentBounds['topRight'],
                                    self.opponentBounds['bottomLeft'],
                                    self.opponentBounds['bottomRight'])

    def ball_in_player_bounds(self, x, y, w, h):
        return self._ball_in_bounds(x, y, w, h,
                                    self.playerBounds['topLeft'],
                                    self.playerBounds['topRight'],
                                    self.playerBounds['bottomLeft'],
                                    self.playerBounds['bottomRight'])

    def _ball_in_bounds(self, x, y, w, h, topLeft, topRight, bottomLeft, bottomRight):
        # Alg: Point is inside convex shape if areas of subshapes with ball center and
        # 2 other points add up to form area of entire shape
        center = self.ballCenter(x, y, w, h)
        area = self.shoelaceArea
        totalArea = area(topLeft, topRight, bottomRight, bottomLeft)
        a, b, c, d = (area(center, topLeft, topRight), area(center, topRight, bottomRight),
                      area(center, bottomRight, bottomLeft), area(center, bottomLeft, topLeft))
        val1 = round(sum((a, b, c, d)) * 10000) / 10000
        val2 = round(totalArea * 10000) / 10000
        inBounds = val1 == val2
        print(
            f"inBounds:{inBounds} sum:{a}+{b}+{c}+{d}={val1}, area:{val2}, ball:{center}, tL:{topLeft}, tR:{topRight}, bL:{bottomLeft}, bR:{bottomRight}")
        return inBounds

    def start(self):
        print("Starting...")
        capture = cv.VideoCapture(self.stream)
        firstFrame = None

        while True:
            ret, frame = self.capture.read()
            if not ret:
                print('Something went wrong while trying to access webcam')
                break

            # Make frame gray
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # frame = imutils.resize(frame, width=500)
            # text = "No motion"
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0) # ksize originally (21,21)
            # gray = cv.Canny(gray, 125, 175)

            if firstFrame is None:
                firstFrame = gray

            frameDelta = cv.absdiff(firstFrame, gray)
            thresh = cv.threshold(frameDelta, 25, 255, cv.THRESH_BINARY)[1]

            thresh = cv.dilate(thresh, None, iterations=2)
            contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            text=("Ball NOT in bounds", (0,0,255))
            for contour in contours:
                if cv.contourArea(contour) < 250: # 500 is the minimum area size
                    continue
                x,y,w,h = cv.boundingRect(contour)
                cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                if self.ball_in_opponent_bounds(x, y, w, h):
                    text=("Ball in OPPONENT bounds", (0,255,0))
                elif self.ball_in_player_bounds(x, y, w, h):
                    text=("Ball in PLAYER bounds", (0,255,0))

            cv.putText(frame, f"Status: {text[0]}", (10,60), cv.FONT_HERSHEY_COMPLEX, 1.5, text[1], 2)

            # cv.imshow('mask output', frameDelta)
            cv.imshow('mask output', frame)
            if cv.waitKey(1) == ord('q'): break

        print("Stream ended")
        self.capture.release()
        cv.destroyAllWindows()

# """
if __name__ == "__main__":
    ttcv = TableTennisCV()
    ttcv.plot("canon_table.png")
    # ttcv.start()

# """