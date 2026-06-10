from ultralytics import YOLO

DATA_YAML = "datasets/detection/data.yaml"
MODEL_CONFIG = "yolo11s.yaml"

EPOCHS =50
IMAGE_SIZE =672
BATCH_SIZE = 48

PROJECT_DIR = "runs/person_detector"
RUN_NAME = "small_test_run"

def main():
	model = YOLO(MODEL_CONFIG)

	model.train(data= DATA_YAML,
			 epochs=EPOCHS,
			 imgsz=IMAGE_SIZE,
			 batch=BATCH_SIZE,
			 project=PROJECT_DIR,
			 name=RUN_NAME,
			 device=0
			 )
			 


if __name__ == "__main__":
	main()

