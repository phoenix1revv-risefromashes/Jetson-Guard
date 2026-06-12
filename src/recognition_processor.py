import cv2

class RecognitionProcessor:

    def __init__(self, yolo_person_detector, face_detector, face_recognizer):
        self.yolo_person_detector = yolo_person_detector
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer

    
    def __call__(self, frame):
        return self.process_frame(frame)
    

    
    def get_largest_face(self, faces):

        if len(faces) ==0:
            return None
        
        return max(faces, key=lambda face: face[2] * face[3])
    

    def process_frame(self, frame):

        display_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        person_detections = self.yolo_person_detector.detect_persons(frame)

        for person in person_detections:
            x1 = max(0, person['x1'])
            x2=min(frame_width, person['x2'])
            y1=max(0,person['y1'])
            y2 = min(frame_width, person['y2'])

            person_crop =frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            faces = self.face_detector.detect_faces(person_crop)
            largest_face = self.get_largest_face(faces)

            if largest_face is None:
            
                continue

            face_x, face_y, face_w, face_h = largest_face

            absolute_face_x1 = x1 + face_x
            absolute_face_y1 = y1 + face_y
            absolute_face_x2 = absolute_face_x1 + face_w
            absolute_face_y2 = absolute_face_y1 + face_h


            padding = 20

            crop_x1 = max(0, absolute_face_x1 - padding)
            crop_y1 = max(0, absolute_face_y1 - padding)
            crop_x2 = min(frame_width, absolute_face_x2 + padding)
            crop_y2 = min(frame_height, absolute_face_y2 + padding)

            face_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if face_crop_bgr.size == 0:
                continue
            
            face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)


            name, distance = self.face_recognizer.recognize_face_images(
                face_crop_rgb
            )

            if name == "Unknown":
                label = "Unknown"
                box_color = (0, 0, 255)
            else:
                label = f"Known: {name}"
                box_color = (255, 0, 0)

            if distance is not None:
                label = f"{label} | {distance:.2f}"



            #Person Bounding box
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            #Face Bounding Box
            cv2.rectangle(
                display_frame,
                (absolute_face_x1, absolute_face_y1),
                (absolute_face_x2, absolute_face_y2),
                box_color,
                2,
            )

           #label above person box
            cv2.putText(
                display_frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2,
            )

        return display_frame
    

            


            






















    

        
