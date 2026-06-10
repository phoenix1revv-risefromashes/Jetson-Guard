from pathlib import Path
import json
import shutil
from collections import defaultdict


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

YOLO_PERSON_CLASS_ID = 0

MAX_TRAIN_IMAGES = 200
MAX_VAL_IMAGES = 50



def create_output_dirs():
    TRAIN_OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VAL_OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    VAL_OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)



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


def coco_bbox_to_ybbox(coco_bbox, image_width,image_height):
    #COCO bbox format: [x_min, y_min, width, height]
    #YOLO bbox format: [x_center, y_center, width, height], normalized from 0 to 1
    
    x_min, y_min, box_width, box_height = coco_bbox

    x_center, y_center = x_min+ box_width/2, y_min+ box_height/2

    x_center_normalized = x_center/image_width
    y_center_normalized = y_center/image_height
    box_width_normalized = box_width/ image_width
    box_height_normalized = box_height /image_height

    return (x_center_normalized,
            y_center_normalized,
            box_width_normalized,
            box_height_normalized)


def load_json_annotation(annotation_file_dir):
    with open(annotation_file_dir) as file:
        return json.load(file)



def get_person_category_id(coco_data):
    for category in coco_data['categories']:
        if category['name'] == 'person':
            return category['id']
    
    raise ValueError ('Could not find "person" category in the COCO annotations')




def group_person_annotations_by_image(coco_data, person_category_id):
    annotations_by_image_id = defaultdict(list)

    for annotation in coco_data['annotations']:
        if annotation['category_id'] == person_category_id:
            image_id = annotation['image_id']
            annotations_by_image_id[image_id].append(annotation)

    return annotations_by_image_id




def convert_split (annotation_file_dir,
                   raw_images_dir,
                   output_images_dir,
                   output_labels_dir,
                   max_images):
    
    coco_data = load_json_annotation(annotation_file_dir)
    person_category_id = get_person_category_id(coco_data)

    person_annotation_by_image = group_person_annotations_by_image(coco_data, person_category_id)

    copied_image_count = 0
    written_label_count = 0

    for image_info in coco_data["images"]:
        if copied_image_count>=max_images:
            break

        image_id = image_info['id']

        if image_id not in person_annotation_by_image:
            continue

        file_name = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        source_image_dir = raw_images_dir / file_name
        destination_image_dir = output_images_dir/file_name

        if not source_image_dir.exists():
            print(f"skipping_mmissing_image: {source_image_dir}")
            continue

        label_file_name = Path(file_name).with_suffix(".txt").name
        label_file_dir = output_labels_dir/label_file_name



        yolo_label_lines = []

        for annotation in person_annotation_by_image[image_id]:
            coco_bbox = annotation['bbox']

            x_center, y_center, width, height = coco_bbox_to_ybbox(coco_bbox, image_width, image_height)

            yolo_label_line= (
                f'{YOLO_PERSON_CLASS_ID} '
                f'{x_center:.6f} '
                f'{y_center:.6f} '
                f'{width:.6f} '
                f'{height:.6f}'
            )

            yolo_label_lines.append(yolo_label_line)

        

        shutil.copy2(source_image_dir,destination_image_dir)

        with open(label_file_dir, 'w') as label_file:
            label_file.write("\n".join(yolo_label_lines))
        
        copied_image_count+=1
        written_label_count+=len(yolo_label_lines)

    print(f"Images copied : {copied_image_count}")
    print(f"person labels written: {written_label_count}")












def main():
    create_output_dirs()
    check_required_paths()
    print("COCO raw dataset paths found successfully.")

    print("Preparing COCO person subset:  ----")

    convert_split(annotation_file_dir=TRAIN_ANNOTATION_FILE,
                  raw_images_dir=TRAIN_RAW_IMAGES_DIR,
                  output_images_dir=TRAIN_OUTPUT_IMAGES_DIR,
                  output_labels_dir=TRAIN_OUTPUT_LABELS_DIR,
                  max_images=MAX_TRAIN_IMAGES) 
    

    convert_split(
        annotation_file_dir=VAL_ANNOTATION_FILE,
        raw_images_dir=VAL_RAW_IMAGES_DIR,
        output_images_dir=VAL_OUTPUT_IMAGES_DIR,
        output_labels_dir=VAL_OUTPUT_LABELS_DIR,
        max_images=MAX_VAL_IMAGES
    ) 


    print("\n COCO person subset preparation complete..")

    
if __name__ == "__main__":
    main()