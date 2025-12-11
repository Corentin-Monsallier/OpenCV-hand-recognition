import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self, mode=False, maxFaces=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.mode,
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils


    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION, self.mpDraw.DrawingSpec(color=(0, 0, 255)), self.mpDraw.DrawingSpec(color=(0, 255, 0)))
        return img

    def findPosition(self, img, faceNo=0, draw=True, multiple=False):
        lmLists = []
        if self.results.multi_face_landmarks:
            for faceIndex, myFace in enumerate(self.results.multi_face_landmarks):
                if not multiple and faceIndex != faceNo:
                    continue
                lmList = []
                for id, lm in enumerate(myFace.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])
                lmLists.append(lmList)
        return lmLists if multiple else (lmLists[0] if lmLists else [])