from pathlib import Path

COCO_RAW_DIR = Path('datasets/detection/raw')
COCO_PROCESSED_DIR =Path('datasets/detection/processed')

TRAIN_ANNOTATION_FILE = COCO_RAW_DIR / "annotations" / "instances_train2017.json"
VAL_ANNOTATION_FILE = COCO_RAW_DIR /"annotations"/"instances_val2017.json"

TRAIN_RAW_IMAGES_DIR = COCO_RAW_DIR /"train2017"
VAL_RAW_IMAGES_DIR = COCO_RAW_DIR / "val2017"

TRAIN_OUTPUT_IMAGES_DIR = COCO_PROCESSED_DIR /'images'/'train'
VAL_OUTPUT_IMAGES_DIR = COCO_PROCESSED_DIR /'images'/'val'

TRAIN_OUTPUT_LABELS_DIR = COCO_PROCESSED_DIR/'labels'/'train'
VAL_OUTPUT_LABELS_DIR = COCO_PROCESSED_DIR/'labels'/'val'

def check_required_paths():
    required_paths = [
        TRAIN_ANNOTATION_FILE,
        VAL_ANNOTATION_FILE,
        TRAIN_RAW_IMAGES_DIR,
        VAL_RAW_IMAGES_DIR,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")


def main():
    check_required_paths()
    print("COCO raw dataset paths found successfully.")


if __name__ == "__main__":
    main()