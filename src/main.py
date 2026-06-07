from camera import Camera
from config import CAMERA_INDEX

def main():
    camera = Camera(CAMERA_INDEX)

    try:
        camera.open_camera()
        camera.show_live_view()
    
    except RuntimeError as error:
        print(f"Error: {error}")

    finally:
        camera.release()

if __name__== "__main__":
    main()
    
