# Edge-Based Person Recognition for Safety and Access Monitoring Using Jetson Orin

An edge-AI safety and access monitoring system designed to detect close human subjects, validate detections, reject person-shaped false positives, and support future local identity recognition on NVIDIA Jetson Orin.

<p align="center">
  <a href="assets/">
    <img src="assets/architecture/architecture.png" alt="Project Architecture Flowchart" width="100%">
  </a>
</p>

<p align="center">
  <sub><b>Click the architecture image to browse demos, bug-fix evidence, and project assets.</b></sub>
</p>

---

## Project Overview

This project builds a local computer vision pipeline for safety and access monitoring using a camera and edge inference. The system detects a close human subject, validates that the detected region contains a visible face, and rejects false positives such as hanging clothing, unclear objects, or person-shaped background noise.

The system is not designed as a general surveillance detector. It is designed as a close-range safety-scanning pipeline where a valid detection requires both:

1. A close person-like region detected by YOLO.
2. A visible face confirmed inside that same detected region.

This creates a two-stage validation system:

```text
YOLO person detection
+
Face confirmation inside YOLO box
=
Confirmed close-person detection
```

The long-term goal is to extend this system into local known-person recognition, where the scanner can classify a detected subject as known or unknown without relying on cloud inference.

---

## Current Version

```text
v0.3.1
```

Current functionality:

- Live camera input using OpenCV.
- YOLO-based person detection.
- Close-subject filtering using bounding-box height ratio.
- Face confirmation inside each YOLO-detected person box.
- Rejection of hanging clothing and person-shaped false positives.
- Demo and bug-fix evidence videos.
- Local edge-oriented architecture prepared for Jetson Orin deployment.

---

## Current Runtime Flow

```text
Camera Sensor
↓
OpenCV Camera Module
↓
Jetson Orin Edge Runtime
↓
YOLO Person Detection
↓
Confidence Filtering
↓
Close-Subject Height-Ratio Filtering
↓
Crop Detected Person Region
↓
Face Confirmation
↓
Detection Validation Decision
↓
Accepted / Rejected Detection
↓
Annotated Live Output
↓
Safety / Access Monitoring Output
```

---

## Why YOLO + Face Confirmation?

A face detector alone can only answer:

```text
Is there a frontal face somewhere in the frame?
```

That is not enough for a safety scanner because a random face in the background or a face-like image may create incorrect behavior.

A YOLO person detector alone can answer:

```text
Is there a person-shaped object in the frame?
```

But YOLO can still falsely detect hanging clothing, jackets, mannequins, or other person-shaped objects as a person.

The combined system is stronger because it asks:

```text
Is there a close person-shaped region?
Does that same detected region contain a visible face?
```

This makes the pipeline stricter and more useful for safety and access monitoring.

---

## Current Detection Rule

A detection is accepted only when all conditions are satisfied:

```text
YOLO detects a person-shaped object
AND detection confidence passes threshold
AND detected box is large enough to be close
AND a visible face is found inside that same YOLO box
```

A detection is rejected when:

```text
No face is found inside the YOLO box
OR the object is too small/far away
OR the object is clothing/noise/unclear subject
```

This means the system intentionally rejects unclear body-only detections or side/profile cases where a visible face is not confirmed. This is acceptable for the current safety-scanning workflow because the subject is expected to face the scanner.

---

## Version History

### v0.1.0 — Camera Module

Implemented the base camera pipeline.

Features:

- Open camera feed using OpenCV.
- Read live frames.
- Display camera window.
- Quit safely using `q`.
- Release camera resources properly.

Main files:

```text
src/camera.py
src/main.py
src/config.py
```

---

### v0.2.0 — Face Detection Baseline

Added a simple face detection baseline using OpenCV Haar Cascade.

Purpose:

- Validate live frame processing.
- Test face detection on real camera input.
- Build a reusable `process_frame()` structure.

Main file:

```text
src/face_detector.py
```

This module later became useful as the face-confirmation stage inside the YOLO detection pipeline.

---

### v0.3.0 — YOLO Live Close-Person Detection

Integrated a trained YOLO person detector into the live camera pipeline.

Major work completed:

- Prepared a COCO person subset.
- Converted COCO annotations into YOLO format.
- Trained a YOLO11s person detector.
- Evaluated the training run.
- Integrated the trained `best.pt` model into the live camera pipeline.
- Added close-subject height-ratio filtering.
- Added demo video proof.

Training dataset used:

```text
Training images:   10,000
Validation images: 2,000
```

Final evaluation metrics from the training run:

```text
Precision: 0.782
Recall:    0.611
mAP50:     0.709
mAP50-95:  0.458
```

Known issue discovered:

```text
Hanging clothing or person-shaped objects could be falsely detected as a close person.
```

This issue was moved into the patch version `v0.3.1`.

---

### v0.3.1 — Face Confirmation False-Positive Fix

Added a second-stage validation step to reduce false-positive detections.

Problem fixed:

```text
Hanging clothing was detected as a close person.
```

Fix implemented:

```text
Each YOLO-detected person box must contain a visible face before being accepted.
```

Updated validation logic:

```text
YOLO detects person-like object
↓
Height ratio confirms object is close enough
↓
System crops the YOLO person box
↓
Face detector checks inside that crop
↓
If face is found: accept as close person
↓
If face is not found: reject detection
```

Validated behavior:

- Detects a person only when a visible face is present.
- Rejects hanging clothing and person-shaped objects.
- Rejects non-human visual noise.
- Rejects unclear body-only or partial scan subjects when no face is visible.

Known limitation:

```text
The current face confirmation step is intentionally strict.
It works best when the subject presents a visible frontal face to the scanner.
Side/profile faces or poorly visible faces may be rejected.
```

This behavior is acceptable for the current safety-scanning workflow.

---

## Project Structure

```text
.
├── assets/
│   ├── architecture/
│   │   └── architecture.png
│   │
│   ├── demo_videos/
│   │   └── v0.3.0/
│   │       └── v0.3.0_yolo_live_close_person_detection_demo.mp4
│   │
│   └── bug_evidence/
│       └── v0.3.1/
│           ├── v0.3.1_before_clothing_false_positive_bug.mp4
│           └── v0.3.1_after_face_confirmation_false_positive_fix.mp4
│
├── datasets/
│   └── detection/
│       ├── raw/
│       ├── processed/
│       └── data.yaml
│
├── models/
│   └── person_detector/
│       └── best.pt
│
├── scripts/
│   ├── prepare_coco_person_subset.py
│   └── train_yolo_person_detector.py
│
├── src/
│   ├── camera.py
│   ├── config.py
│   ├── face_detector.py
│   ├── main.py
│   └── yolo_detector.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Main Software Components

### `src/main.py`

Application entry point.

Responsibilities:

- Loads camera configuration.
- Creates the camera object.
- Creates the YOLO detector object.
- Starts the live camera loop.
- Sends each frame into the detection pipeline.

---

### `src/camera.py`

Camera acquisition module.

Responsibilities:

- Open the camera.
- Read frames.
- Send frames to a frame processor.
- Display processed frames.
- Release the camera safely.

The key design is that the camera module can accept any object with:

```python
process_frame(frame)
```

This allows the system to switch between face detection, YOLO detection, and future recognition modules cleanly.

---

### `src/yolo_detector.py`

Main runtime detection and validation module.

Responsibilities:

- Load the trained YOLO model.
- Run person detection on every frame.
- Apply confidence filtering.
- Apply close-subject height-ratio filtering.
- Crop the detected person region.
- Run face confirmation inside the YOLO box.
- Draw accepted detections on the live frame.

Current accepted label:

```text
Close Person
```

---

### `src/face_detector.py`

Face confirmation module.

Responsibilities:

- Load OpenCV Haar Cascade face detector.
- Convert frames or crops into grayscale.
- Detect frontal faces.
- Return face bounding boxes.

In the current pipeline, this module confirms whether the YOLO-detected person region contains a visible face.

---

### `src/config.py`

Central configuration file.

Example values:

```python
CAMERA_INDEX = 0

YOLO_MODEL_PATH = "models/person_detector/best.pt"
YOLO_CONFIDENCE_THRESHOLD = 0.50
MIN_PERSON_HEIGHT_RATIO = 0.35
```

Meaning:

```text
YOLO_CONFIDENCE_THRESHOLD = minimum confidence needed to keep a detection
MIN_PERSON_HEIGHT_RATIO = minimum box height ratio needed to count as close
```

For a 1080p camera frame:

```text
0.35 × 1080 = 378 pixels
```

So the detected person box must be at least 378 pixels tall to pass the close-subject filter.

---

## Model Training Pipeline

The YOLO detector was trained using a COCO person subset.

Training preparation script:

```text
scripts/prepare_coco_person_subset.py
```

Training script:

```text
scripts/train_yolo_person_detector.py
```

Dataset configuration:

```text
datasets/detection/data.yaml
```

Training output:

```text
models/person_detector/best.pt
```

Current model:

```text
YOLO11s person detector
```

Training summary:

```text
Training images:   10,000
Validation images: 2,000
Epochs:            50
Output:            best.pt
```

Final evaluation metrics:

```text
Precision: 0.782
Recall:    0.611
mAP50:     0.709
mAP50-95:  0.458
```

---

## Evidence and Validation

### Architecture

System architecture image:

```text
assets/architecture/architecture.png
```

The architecture image at the top of this README links to the `assets/` folder, where demo and validation evidence can be reviewed.

---

### v0.3.0 Demo Evidence

Feature demo video:

```text
assets/demo_videos/v0.3.0/v0.3.0_yolo_live_close_person_detection_demo.mp4
```

Purpose:

```text
Shows trained YOLO live close-person detection running in the camera pipeline.
```

---

### v0.3.1 Bug-Fix Evidence

Before/after bug evidence videos:

```text
assets/bug_evidence/v0.3.1/v0.3.1_before_clothing_false_positive_bug.mp4
assets/bug_evidence/v0.3.1/v0.3.1_after_face_confirmation_false_positive_fix.mp4
```

Purpose:

```text
Shows the clothing false-positive problem before the fix and validates the face-confirmation fix after implementation.
```

---

## How to Run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the live scanner:

```bash
python src/main.py
```

Quit the camera window:

```text
Press q
```

---

## Current Detection Behavior

The current system accepts a close-person detection only when:

```text
YOLO detects a person-shaped object
AND confidence is high enough
AND detected box is large enough
AND a visible face is found inside the same detected box
```

The system rejects:

- Hanging clothing.
- Person-shaped background objects.
- Non-human visual noise.
- Body-only detections without face confirmation.
- Partial or unclear scan poses where the face is not visible.

This makes the system stricter and better aligned with safety/access scanning.

---

## Design Principles

This project follows these design principles:

```text
Edge-only inference
Privacy-aware local processing
Two-stage validation: person detection + face confirmation
Modular software design
Versioned development workflow
Evidence-based testing with demo and bug-fix videos
```

---

## GitHub Workflow

This project uses a branch-based workflow with issues, pull requests, milestone tags, and releases.

Completed release tags:

```text
v0.3.0
v0.3.1
```

Selected v0.3 milestone tags:

```text
v0.3.0-m1-yolo-data-preparation
v0.3.0-m2-yolo-training
v0.3.0-m3-yolo-training
```

Completed branches included:

```text
v0.3-yolo-face-detection-module
v0.3.1-fix-clothing-false-positive
```

The project also uses pull requests for review before merging into `main`.

---

## Planned Next Stage

### v0.4.0 — Face Enrollment Module

Next goal:

```text
Build a local face enrollment system.
```

Planned behavior:

- User enters a person’s name.
- Camera captures multiple face images.
- Images are saved locally under that person’s folder.
- The saved images become the known-person dataset for future recognition.

Planned folder structure:

```text
data/known_faces/
└── phoenix/
    ├── phoenix_001.jpg
    ├── phoenix_002.jpg
    └── phoenix_003.jpg
```

---

### v0.5.0 — Known-Person Recognition

Planned behavior:

- Load enrolled face images.
- Generate face embeddings.
- Compare live face against stored identities.
- Return known person or unknown person.

Future recognition flow:

```text
Close person detected
↓
Face confirmed
↓
Face embedding generated
↓
Compared with known face database
↓
Known / Unknown result
```

---

## Long-Term Goal

The long-term goal is to build a complete edge-based safety and access monitoring prototype that can:

- Detect a close scan subject.
- Confirm that the subject is human.
- Reject false positives and unclear scan cases.
- Recognize known vs unknown individuals.
- Run locally on Jetson Orin.
- Preserve privacy by avoiding cloud inference.
- Provide reliable real-time feedback for safety and access monitoring workflows.

---

## Current Status Summary

```text
Camera pipeline:              Complete
Face detection baseline:      Complete
YOLO person detector:         Complete
Live YOLO integration:        Complete
False-positive reduction:     Complete
Face enrollment:              Planned
Known-person recognition:     Planned
Jetson deployment hardening:  Planned
```