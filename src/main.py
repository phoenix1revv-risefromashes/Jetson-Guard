from camera import Camera
from config import CAMERA_INDEX
from face_detector import *

def main():
    camera = Camera(CAMERA_INDEX)
    face_detector =FaceDetector()


    try:
        camera.open_camera()
        camera.show_live_view(frame_processor=face_detector)
    
    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        camera.release()

if __name__== "__main__":
    main()
    
