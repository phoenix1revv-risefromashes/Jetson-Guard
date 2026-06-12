from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from config import CAMERA_INDEX
from config import YOLO_MODEL_DIR
from config import YOLO_CONFIDENCE_THRESHOLD
from config import MIN_PERSON_HEIGHT_RATIO

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer


WINDOW_NAME = "Live Known/Unknown Face Recognition"


def get_largest_face(faces):
    if len(faces) == 0:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def run_live_face_recognition():
    yolo_model_path = PROJECT_ROOT / YOLO_MODEL_DIR
    yolo_model = YOLO(str(yolo_model_path))

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera with index: {CAMERA_INDEX}")

    try:
        while True:
            ret, frame = camera.read()

            if not ret:
                print("Could not read frame from camera.")
                break

            display_frame = frame.copy()
            frame_height, frame_width = frame.shape[:2]

            results = yolo_model(
                frame,
                conf=YOLO_CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = yolo_model.names[class_id]

                    if class_name != "person":
                        continue

                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_width, x2)
                    y2 = min(frame_height, y2)

                    person_height = y2 - y1
                    person_height_ratio = person_height / frame_height

                    if person_height_ratio < MIN_PERSON_HEIGHT_RATIO:
                        continue

                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue

                    faces = face_detector.detect_faces(person_crop)
                    largest_face = get_largest_face(faces)

                    if largest_face is None:
                        cv2.rectangle(
                            display_frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 0, 255),
                            2,
                        )

                        cv2.putText(
                            display_frame,
                            "Person detected | No face",
                            (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

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

                    face_crop_rgb = cv2.cvtColor(
                        face_crop_bgr,
                        cv2.COLOR_BGR2RGB,
                    )

                    name, distance = face_recognizer.recognize_face_images(
                        face_crop_rgb
                    )

                    if name == "Unknown":
                        label = "Unknown"
                        box_color = (0, 0, 255)
                    else:
                        label = f"Known: {name}"
                        box_color = (0, 255, 0)

                    if distance is not None:
                        label = f"{label} | {distance:.2f}"

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2,
                    )

                    cv2.rectangle(
                        display_frame,
                        (absolute_face_x1, absolute_face_y1),
                        (absolute_face_x2, absolute_face_y2),
                        box_color,
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_face_recognition()