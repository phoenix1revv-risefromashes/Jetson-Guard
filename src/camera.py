import cv2
from config import CAMERA_INDEX

class Camera:
    def __init__(self, camera_index):
        self.camera_index= CAMERA_INDEX
        self.capture =None

    def open_camera(self):
        self.capture = cv2.VideoCapture(self.camera_index)

        if not self.capture:
            raise RuntimeError("Could not open Camera: Try different camera index")
        
    
    def read_frames(self):

        if self.capture == None:
            raise RuntimeError ("Camera is not opened yet. Please open the camera first")
        
        success, frames = self.capture.read()

        if not success:
            raise RuntimeError('Could not read the frames')
        
        return frames
    

    def show_live_view(self, frame_processor=None):

        while True:
            frame =self.read_frames()

            if frame_processor is not None:
                frame = frame_processor.process_frame(frame)

                
            cv2.imshow("Camera Test - press Q to exit", frame)

            if cv2.waitKey(1) & 0xFF == ord ("q"):
                break 

       

    def release(self):
        if self.capture is not None:
            self.capture.release()
        cv2.destroyAllWindows()



