import cv2

class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier