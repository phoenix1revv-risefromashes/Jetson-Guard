from ultralytics import YOLO

import cv2

from face_detector import FaceDetector


class YOLODetector():
    def __init__(self, model_path, confidence_threshold, min_person_height_ratio):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_person_height_ratio = min_person_height_ratio
        self.model = YOLO(self.model_path)
        self.face_detector = FaceDetector()

    
    def process_frame (self, frame):
        frame_height, frame_width = frame.shape[:2]
        
        results_from_model = self.model(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )

        boxes = results_from_model[0].boxes

        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())

            box_height = y2-y1
            height_ratio = box_height/frame_height

            if height_ratio<self.min_person_height_ratio:
                continue



            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])

            person_crop = frame[y1:y2, x1:x2]
            faces = self.face_detector.detect_faces(person_crop)

            if len(faces)==0:
                continue



            label = f"Face_Detected (confid:) {confidence: .2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),2)

            cv2.putText(
                frame,
                label,
                (x1, y1-10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0,255,0),
                2
            )
        return frame
    
    








