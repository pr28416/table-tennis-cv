import cv2 as cv
import imutils
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TableTennisCV():
    def __init__(self, stream=0):
        self.stream = stream
        print("Launched TableTennisCV")
        self.opponentBounds = {'topLeft': (320, 256),'topRight': (1036, 256),'bottomRight': (1210, 720),'bottomLeft': (100, 720)}
        self.playerBounds = {'topLeft': (450, 0),'topRight': (900, 0),'bottomRight': (920, 60),'bottomLeft': (424, 60)}

    def plot(self, image_path):
        img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.show()

    def ballCenter(self, x, y, w, h):
        return (x+x+w)//2, (y+y+h)//2

    def shoelaceArea(self, *args): # Uses shoelace theorem
        # TODO: Implement function
        p, q, m = 0, 0, len(args)
        for i in range(len(args)):
            p += args[i][0]*args[(i+1)%m][1]
            q += args[i][1]*args[(i+1)%m][0]
        return abs(p-q)/2

    def ball_in_opponent_bounds(self, x, y, w, h):
        return self._ball_in_bounds(x, y, w, h,self.opponentBounds['topLeft'],self.opponentBounds['topRight'],self.opponentBounds['bottomLeft'],self.opponentBounds['bottomRight'])

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
        a, b, c, d = (area(center, topLeft, topRight),
                      area(center, topRight, bottomRight),
                      area(center, bottomRight, bottomLeft),
                      area(center, bottomLeft, topLeft))
        val1 = round(sum((a, b, c, d)) * 10000) / 10000
        val2 = round(totalArea * 10000) / 10000
        inBounds = val1 == val2
        # print(f"inBounds:{inBounds} sum:{a}+{b}+{c}+{d}={val1}, area:{val2}, ball:{center}, tL:{topLeft}, tR:{topRight}, bL:{bottomLeft}, bR:{bottomRight}")
        return inBounds

    def _timestamp(self):
        return datetime.strftime(datetime.today(), "%m-%d-%Y %H.%M.%S")

    def start(self):
        logfile = open(f"{self._timestamp()} ttcv log.txt", "w")
        try:
            self._start(logfile)
        except Exception as err:
            logfile.close()
            raise err
        else:
            logfile.close()

    def _start(self, logfile):
        print("Starting...")
        capture = cv.VideoCapture(self.stream)
        firstFrame = None
        prevSide = 1 # Oscillates between 0 and 1; 0=opponent, 1=player
        tmpCoord = (-1, -1)

        while True:
            ret, frame = capture.read()
            if not ret:
                print('Something went wrong while trying to access webcam')
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (21, 21), 0) # ksize originally (21,21)

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
                    text=["Ball in OPPONENT bounds", (0,255,0)]
                    if prevSide == 1:
                        log = f"{self._timestamp()}, {tmpCoord}, {self.ballCenter(x,y,w,h)}"
                        print("logging:", log)
                        logfile.write(f"{log}\n")
                        prevSide = 0
                        text[0] += " - CORRECT"
                elif self.ball_in_player_bounds(x, y, w, h):
                    text=["Ball in PLAYER bounds", (0,255,0)]
                    if prevSide == 0:
                        tmpCoord = self.ballCenter(x,y,w,h)
                        prevSide = 1
                        text[0] += " - CORRECT"

            cv.putText(frame, f"{text[0]}", (10,60), cv.FONT_HERSHEY_COMPLEX, 1.5, text[1], 2)

            # cv.imshow('mask output', frameDelta)
            cv.imshow('mask output', frame)
            if cv.waitKey(1) == ord('q'): break

        print("Stream ended")
        capture.release()
        cv.destroyAllWindows()

# """
if __name__ == "__main__":
    ttcv = TableTennisCV()
    # ttcv.plot("canon_table.png")
    ttcv.start()

# """