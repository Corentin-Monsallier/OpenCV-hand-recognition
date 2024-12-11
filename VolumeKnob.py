import cv2
import time
import math
import numpy as np
from HandTracking import handDetector

def main():
    fingerCoords = [(8, 6), (12, 10), (16, 14), (20, 18)]
    thumbCoords = (4, 2)

    ####################          CAMERA SETTINGS          ####################
    pTime, cTime = 0, 0
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    ####################          HAND DETECTOR CONFIG          ####################
    detector = handDetector()

    while True:
        ####################          ESC: end the program          ####################
        key = cv2.waitKey(10)
        if key == 27:
            break
        
        ####################          HAND DISPLAY          ####################
        success, img = cap.read()
        if not success:
            break
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        ####################          FPS DISPLAY          ####################
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        ####################          VOLUME KNOB          ####################
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            ####################          VOLUME CONTROL          ####################
            length = math.hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [50, 300], [0, 100])
            

        ####################          DISPLAY IMAGE          ####################
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()