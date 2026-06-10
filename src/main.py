from camera import Camera
from config import *
from face_detector import *
from yolo_detector import *


def main():
    camera = Camera(CAMERA_INDEX)
    yolo_person_detector =YOLODetector(model_path=YOLO_MODEL_DIR,
                                       confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                                       min_person_height_ratio=MIN_PERSON_HEIGHT_RATIO)


    try:
        camera.open_camera()
        camera.show_live_view(frame_processor=yolo_person_detector)
    
    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        camera.release()

if __name__== "__main__":
    main()
    
