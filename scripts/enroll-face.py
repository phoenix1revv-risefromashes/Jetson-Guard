from pathlib import Path
import sys
import time
import re

import cv2
from ultralytics import YOLO

#Make SRC importable to this script...


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from camera import Camera
from face_detector import FaceDetector

from config import CAMERA_INDEX, KNOW_FACES_DIR, ENROLL_IMAGE_CAPTURE_DELAY, ENROLLMENT_IMAGE_COUNT, YOLO_MODEL_DIR, YOLO_CONFIDENCE_THRESHOLD, MIN_PERSON_HEIGHT_RATIO


def clean_person_name(person_name):
    person_name = person_name.strip().lower()
    person_name = person_name.replace(" ", "_")
    person_name = re.sub(r"[^a-z0-9_]", "", person_name)

    if person_name == "":
        raise ValueError("Person Name can't be empty")

    print(person_name)
    return person_name


def get_largest_face(faces):
    return max(faces, key=lambda face: face[2] * face[3])


def enroll_person():
    raw_name = input("Enter Person's Name: ")
    person_name = clean_person_name(raw_name)

    output_person_dir = PROJECT_ROOT / KNOW_FACES_DIR / person_name
    output_person_dir.mkdir(parents=True, exist_ok=True)

    camera = Camera(CAMERA_INDEX)
    face_detector = FaceDetector()
    load_yolo_model = YOLO(str(PROJECT_ROOT / YOLO_MODEL_DIR))

    saved_count = 0
    last_capture_time = 0
    enrollment_started = False

    try:
        camera.open_camera()

        while saved_count < ENROLLMENT_IMAGE_COUNT:
            frame = camera.read_frames()
            frame_height, frame_width = frame.shape[:2]
            display_frame = frame.copy()

            results = load_yolo_model(
                frame,
                conf=YOLO_CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                box_height = y2 - y1
                height_ratio = box_height / frame_height

                if height_ratio < MIN_PERSON_HEIGHT_RATIO:
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                person_crop = frame[y1:y2, x1:x2]

                if person_crop.size == 0:
                    continue

                faces = face_detector.detect_faces(person_crop)

                if len(faces) == 0:
                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        "Rejected: Full-Face Please",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    continue

                face_x, face_y, face_w, face_h = get_largest_face(faces)

                absolute_face_x1 = x1 + face_x
                absolute_face_y1 = y1 + face_y
                absolute_face_x2 = absolute_face_x1 + face_w
                absolute_face_y2 = absolute_face_y1 + face_h

                cv2.rectangle(
                    display_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.rectangle(
                    display_frame,
                    (absolute_face_x1, absolute_face_y1),
                    (absolute_face_x2, absolute_face_y2),
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    display_frame,
                    f"Accepted: (conf): {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2,
                )

                current_time = time.time()

                if enrollment_started and current_time - last_capture_time >= ENROLL_IMAGE_CAPTURE_DELAY:
                    face_crop = person_crop[
                        face_y:face_y + face_h,
                        face_x:face_x + face_w,
                    ]

                    if face_crop.size == 0:
                        continue

                    image_name = f"{person_name}_{saved_count + 1:03d}.jpg"
                    image_path = output_person_dir / image_name

                    cv2.imwrite(str(image_path), face_crop)

                    saved_count += 1
                    last_capture_time = current_time

                    print(f"saved: {image_path}")

                break

            progress_text = f"Enrollment: {person_name} | {saved_count}/{ENROLLMENT_IMAGE_COUNT}"

            progress_text_size, _ = cv2.getTextSize(
                progress_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1,
            )

            progress_x = (frame_width - progress_text_size[0]) // 2

            cv2.putText(
                display_frame,
                progress_text,
                (progress_x, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1,
            )

            if enrollment_started:
                status_text = "| Enrollment running |"
            else:
                status_text = "| Press S to start enrollment |"

            status_text_size, _ = cv2.getTextSize(
                status_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            status_x = (frame_width - status_text_size[0]) // 2

            cv2.putText(
                display_frame,
                status_text,
                (status_x+20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )

            cv2.imshow("Validated Face Enrollment: Q to exit", display_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and not enrollment_started:
                enrollment_started = True
                last_capture_time = 0
                print("Enrollment started.")

            if key == ord("q"):
                break

    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        camera.release()


if __name__ == "__main__":
    enroll_person()