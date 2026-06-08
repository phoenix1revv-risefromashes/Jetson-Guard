import cv2 

class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if self.detector.empty():
            raise RuntimeError("could not load face detector")
        

    def detect_faces(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (80,80)
        )

        return faces
    

    def draw_faces(self, frame, faces):
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,
                            (x,y),
                            (x+w, y+h),
                            (0,255,0),
                            2)
            cv2.putText(frame,
                            "Face Detected",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2
                        )
            
        return frame
    
    def process_frame(self,frame):
        face = self.detect_faces(frame)
        frame= self.draw_faces(frame, face)

        return frame
    
    
        

        


