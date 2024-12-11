import cv2
import time
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
    detector = handDetector(maxHands=4) # Adjust maxHands as needed

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
        lmList = detector.findPosition(img, multiple = True)

        ####################          FPS DISPLAY          ####################
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        ####################          FINGER COUNTING          ####################
        totalFingers = 0
        for handLmList in lmList:
            if handLmList:
                wrist_x = handLmList[0][1]
                thumb_tip_x = handLmList[1][1]
                if thumb_tip_x > wrist_x:
                    handType = "Left Hand"
                else:
                    handType = "Right Hand"

                upCount = 0
                for finger in fingerCoords:
                    if handLmList[finger[0]][2] < handLmList[finger[1]][2]:
                        upCount += 1
                if handType == "Left Hand":
                    if handLmList[thumbCoords[0]][1] > handLmList[thumbCoords[1]][1]:
                        upCount += 1
                else:
                    if handLmList[thumbCoords[0]][1] < handLmList[thumbCoords[1]][1]:
                        upCount += 1
                
                totalFingers += upCount

        cv2.putText(img, f'Total Fingers: {totalFingers}', (10, 120), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        ####################          DISPLAY IMAGE          ####################
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()