"""
test_face_recognizer.py

Purpose of this script:
-----------------------
This script tests whether src/face_recognizer.py can correctly load the saved
known face embeddings database and recognize one enrolled face image.

This is a small safety test before we connect recognition to the live camera.

It checks:

    data/known_faces/<person_name>/<image>.jpg
        ↓
    FaceRecognizer
        ↓
    recognized name + distance score

If this works, then the recognition engine is ready for live camera integration.
"""

from pathlib import Path
import sys

import cv2


# Get the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add src/ to Python import path so this script can import project modules.
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


from config import KNOWN_FACE_DIR
from face_recognizer import FaceRecognizer


def test_face_recognizer():
    """
    Load one enrolled face image and test if FaceRecognizer can recognize it.
    """

    # This points to:
    #
    #     data/known_faces/
    #
    known_faces_dir = PROJECT_ROOT / KNOWN_FACE_DIR

    # Find enrolled person folders.
    #
    # Example:
    #
    #     data/known_faces/phoenix/
    #
    person_dirs = [path for path in known_faces_dir.iterdir() if path.is_dir()]

    if len(person_dirs) == 0:
        print("No enrolled person folders found.")
        return

    # Use the first enrolled person folder for this test.
    #
    # Example:
    #
    #     person_dir = data/known_faces/phoenix/
    #
    person_dir = person_dirs[0]

    # The folder name is the expected person name.
    #
    # Example:
    #
    #     expected_name = "phoenix"
    #
    expected_name = person_dir.name

    # Collect face images from that folder.
    image_paths = (
        list(person_dir.glob("*.jpg"))
        + list(person_dir.glob("*.jpeg"))
        + list(person_dir.glob("*.png"))
    )

    if len(image_paths) == 0:
        print(f"No face images found for: {expected_name}")
        return

    # Use the first available enrolled face image.
    test_image_path = image_paths[0]

    # Load image using OpenCV.
    image = cv2.imread(str(test_image_path))

    if image is None:
        print(f"Could not read image: {test_image_path}")
        return

    # OpenCV loads BGR.
    # Face recognition expects RGB.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create recognizer.
    #
    # This loads:
    #
    #     data/face_embeddings/known_faces.pkl
    #
    recognizer = FaceRecognizer()

    # Recognize the enrolled face image.
    name, distance = recognizer.recognize_face_images(rgb_image)

    print(f"Test image: {test_image_path}")
    print(f"Expected name: {expected_name}")
    print(f"Recognized name: {name}")
    print(f"Distance: {distance}")


if __name__ == "__main__":
    test_face_recognizer()
