from pathlib import Path
import sys
import pickle

import cv2
import face_recognition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT/'src'
sys.path.insert(0,str(SRC_DIR))


from config import KNOWN_FACES_DIR, FACE_EMBEDDINGS_FILE


def build_face_embeddings():
    known_faces_dir = PROJECT_ROOT/KNOWN_FACES_DIR
    embeddings_file = PROJECT_ROOT/FACE_EMBEDDINGS_FILE

    known_face_names = []
    known_face_encodings =[]

    if not known_faces_dir.exists():
        raise FileNotFoundError(f"Known faces dir not found: {known_faces_dir}")
    
    persons_dirs = [path for path in known_faces_dir.iterdir() if path.is_dir()]

    if len(persons_dirs) == 0:
        print("No enrolled people found")
        return
    
    for person_dir in persons_dirs :
        person_name = person_dir.name

        image_paths = (list(person_dir.glob("*.jpg")))

        for image_path in image_paths:

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            face_encodings = face_recognition.face_encodings(rgb_image)
            
            if len(face_encodings) != 1:
                
                continue

            face_encoding = face_encodings[0]

            known_face_names.append(person_name)
            known_face_encodings.append(face_encoding)


    if len(known_face_encodings) == 0:
        print("No valid face embeddings were created: ")
        return
    
    face_database = {
        "names": known_face_names,
        "encodings": known_face_encodings
    }

    embeddings_file.parent.mkdir(parents=True, exist_ok=True)

    with open(embeddings_file, 'wb') as file:
        pickle.dump(face_database, file)
    

    print(f"Saved face embeddings database: {embeddings_file}")
    print(f"Total embeddings: {len(known_face_encodings)}")
    print(f"Total people: {len(set(known_face_names))}")
        



   
            

    




if __name__ == "__main__":
    build_face_embeddings()
