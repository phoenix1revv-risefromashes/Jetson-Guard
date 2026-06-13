from pathlib import Path

import cv2


class FaceDetector:
    def __init__(self):
        cascade_candidates = [
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
        ]

        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_candidates.insert(
                0,
                Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml",
            )

        cascade_path = None

        for candidate_path in cascade_candidates:
            if candidate_path.exists():
                cascade_path = candidate_path
                break

        if cascade_path is None:
            raise FileNotFoundError(
                "Could not find haarcascade_frontalface_default.xml. "
                "Install it with: sudo apt install opencv-data"
            )

        self.detector = cv2.CascadeClassifier(str(cascade_path))

        if self.detector.empty():
            raise RuntimeError(f"Could not load face detector from: {cascade_path}")

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80),
        )

        return faces

    def draw_faces(self, frame, faces):
        for (x, y, w, h) in faces:
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return frame

    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        frame = self.draw_faces(frame, faces)

        return frame