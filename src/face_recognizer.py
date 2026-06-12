from pathlib import Path
import pickle

import numpy as np
import face_recognition

from config import FACE_EMBEDDINGS_FILE, FACE_RECOGNITION_THRESHOLD

PROJECT_ROOT = Path(__file__).resolve().parents[1]

class FaceRecognizer:

    def __init__(self):
        self.embeddings_file = PROJECT_ROOT / FACE_EMBEDDINGS_FILE
        self.recognition_threshold = FACE_RECOGNITION_THRESHOLD

        self.known_face_names = []
        self.known_face_encodings = []

        self.load_known_faces()


    def load_known_faces(self):

        if not self.embeddings_file.exists():
            raise RuntimeError(f"Face embedding file not found: {self.embeddings_file}")
        

        with open(self.embeddings_file, "rb") as file:
            face_database = pickle.load(file)

        self.known_face_names = face_database["names"]
        self.known_face_encodings = face_database["encodings"]

        if len(self.known_face_encodings) != len(self.known_face_names):
            raise ValueError("Known face names and encodings count mismatch")
        




    def recognize_encoding(self, face_encoding):
        if not self.known_face_encodings:
            return "unknown", None
        
        face_distances = face_recognition.face_distance(
            self.known_face_encodings,
            face_encoding
        )

        best_match_index = int(np.argmin(face_distances))
        best_distance = float(face_distances[best_match_index])

        if best_distance<= self.recognition_threshold:
            name = self.known_face_names[best_match_index]

            return name, best_distance
        

        return "Unknown", best_distance
    





    def recognize_face_images(self, rgb_face_image):
        face_encodings = face_recognition.face_encodings(rgb_face_image)

        if len(face_encodings) !=1:
            return "Unknown/Multiple faces", None
        
        face_encoding = face_encodings[0]


        return self.recognize_encoding(face_encoding)
    




        
