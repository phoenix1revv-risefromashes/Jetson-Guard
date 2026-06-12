from camera import Camera

from config import CAMERA_INDEX, YOLO_MODEL_DIR, YOLO_CONFIDENCE_THRESHOLD, MIN_PERSON_HEIGHT_RATIO

from face_detector import FaceDetector

from yolo_detector import YOLODetector

from face_recognizer import FaceRecognizer

from recognition_processor import RecognitionProcessor



def main():
    camera = Camera(CAMERA_INDEX)
    yolo_person_detector =YOLODetector(model_path=YOLO_MODEL_DIR,
                                       confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                                       min_person_height_ratio=MIN_PERSON_HEIGHT_RATIO)
    
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()

    recognition_processor = RecognitionProcessor(
        yolo_person_detector=yolo_person_detector,
        face_detector=face_detector,
        face_recognizer=face_recognizer
    )


    try:
        camera.open_camera()
        camera.show_live_view(frame_processor=recognition_processor)


    
    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        camera.release()

if __name__== "__main__":
    main()
    
