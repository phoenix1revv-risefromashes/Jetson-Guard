import cv2


class RecognitionProcessor:
    def __init__(
        self,
        yolo_person_detector,
        face_detector,
        face_recognizer,
        liveness_detector,
    ):
        self.yolo_person_detector = yolo_person_detector
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.liveness_detector = liveness_detector

    def __call__(self, frame):
        return self.process_frame(frame)

    def get_largest_face(self, faces):
        if len(faces) == 0:
            return None

        return max(faces, key=lambda face: face[2] * face[3])

    def draw_center_warning(self, frame, message):
        frame_height, frame_width = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.85
        thickness = 2

        text_size, _ = cv2.getTextSize(message, font, font_scale, thickness)
        text_width, text_height = text_size

        x = max(20, (frame_width - text_width) // 2)
        y = 50

        cv2.rectangle(
            frame,
            (x - 15, y - text_height - 15),
            (x + text_width + 15, y + 15),
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            frame,
            message,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

    def draw_paused_candidate(self, frame, person_box, face_box):
        x1, y1, x2, y2 = person_box
        fx1, fy1, fx2, fy2 = face_box

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

    def process_frame(self, frame):
        display_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        person_detections = self.yolo_person_detector.detect_persons(frame)

        valid_face_candidates = []

        for person in person_detections:
            x1 = max(0, int(person["x1"]))
            y1 = max(0, int(person["y1"]))
            x2 = min(frame_width, int(person["x2"]))
            y2 = min(frame_height, int(person["y2"]))

            person_crop = frame[y1:y2, x1:x2]

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

            valid_face_candidates.append(
                {
                    "person_box": (x1, y1, x2, y2),
                    "face_box": (
                        absolute_face_x1,
                        absolute_face_y1,
                        absolute_face_x2,
                        absolute_face_y2,
                    ),
                    "face_crop_rgb": face_crop_rgb,
                }
            )

        if len(valid_face_candidates) == 0:
            self.liveness_detector.reset()
            return display_frame

        if len(valid_face_candidates) > 1:
            for candidate in valid_face_candidates:
                self.draw_paused_candidate(
                    display_frame,
                    candidate["person_box"],
                    candidate["face_box"],
                )

            self.draw_center_warning(
                display_frame,
                "Multiple faces detected | Recognition paused",
            )

            self.liveness_detector.reset()
            return display_frame

        candidate = valid_face_candidates[0]

        x1, y1, x2, y2 = candidate["person_box"]
        fx1, fy1, fx2, fy2 = candidate["face_box"]
        face_crop_rgb = candidate["face_crop_rgb"]

        is_live, liveness_status = self.liveness_detector.update(face_crop_rgb)

        if not is_live:
            self.draw_paused_candidate(
                display_frame,
                candidate["person_box"],
                candidate["face_box"],
            )

            if liveness_status == "face_not_clear":
                warning_message = "Face not clear | Move closer"
            elif liveness_status == "eyes_closed":
                warning_message = "Blink detected | Open eyes"
            elif liveness_status == "spoof_detected":
                warning_message = "Spoof detected | Liveness failed"
            else:
                warning_message = "Blink to verify liveness"

            self.draw_center_warning(display_frame, warning_message)

            return display_frame

        name, distance = self.face_recognizer.recognize_face_images(face_crop_rgb)

        if name == "Unknown":
            label = "Unknown"
            box_color = (0, 0, 255)
        else:
            label = f"Known: {name}"
            box_color = (0, 255, 0)

        if distance is not None:
            label = f"{label} | {distance:.2f}"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), box_color, 2)

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