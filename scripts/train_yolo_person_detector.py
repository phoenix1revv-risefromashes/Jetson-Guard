from ultralytics import YOLO

DATA_YAML = "datasets/detection/data.yaml"
MODEL_CONFIG = "yolo11n.yaml"

EPOCH =5
IMAGE_SIZE =650
BATCH_SIZE = 16

PROJECT_DIR = "runs/person_detector"
RUN_NAME = "small_test_run"

def main():
	model = YOLO(MODEL_CONFIG)

	model.train(data= DATA_YAML,
			 epoch=EPOCH,
			 imgsz=IMAGE_SIZE,
			 batch=BATCH_SIZE,
			 project=PROJECT_DIR,
			 name=RUN_NAME,
			 device=0
			 )
			 


