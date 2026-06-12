# Edge-Based Person Recognition for Safety and Access Monitoring Using Jetson Orin

An edge-AI computer vision system that detects close human subjects, rejects person-shaped false positives, enrolls known faces locally, and recognizes whether a live subject is known or unknown on NVIDIA Jetson Orin.

<p align="center">
  <a href="assets/">
    <img src="assets/architecture/architecture.png" alt="Project Architecture Flowchart" width="100%">
  </a>
</p>

<p align="center">
  <sub><b>Click the architecture image to browse demos, bug-fix evidence, and project assets.</b></sub>
</p>

---

## Overview

This project uses OpenCV, YOLO, face detection, and face recognition to build a local safety and access-monitoring pipeline.

The system first detects a close person using YOLO, confirms that a visible face exists inside the detected person region, then compares that face against a private local known-face database.

The goal is to run identity-aware recognition locally on edge hardware without relying on cloud inference.

---

## Current Version

```text
v0.5.0 — Live Known/Unknown Face Recognition
```

Current features:

- Live camera input
- YOLO-based close-person detection
- Face confirmation inside detected person regions
- False-positive rejection for clothing and person-shaped objects
- Local face enrollment
- Automatic face embedding update after enrollment
- Private known-face database
- Live known/unknown recognition
- Demo and bug-fix evidence

---

## Pipeline

```text
Camera Frame
↓
YOLO Person Detection
↓
Close-Subject Filtering
↓
Face Confirmation
↓
Face Embedding
↓
Known-Face Database Comparison
↓
Known / Unknown Output
```

---

## Recognition Logic

Recognition only happens when:

```text
YOLO detects a person-shaped object
AND the subject is close enough
AND a visible face is found inside the detected person box
AND a valid face embedding is generated
AND the embedding is compared against the known-face database
```

The system rejects:

- Hanging clothing
- Person-shaped background objects
- Non-human visual noise
- Body-only detections without a visible face
- Invalid face crops

---

## Face Enrollment

Enroll a new person:

```bash
python scripts/enroll_face.py
```

The enrollment script captures verified face crops and saves them locally:

```text
data/known_faces/<person_name>/
```

After enrollment, the face embedding database updates automatically:

```text
data/face_embeddings/known_faces.pkl
```

Private face images and embeddings are not committed to Git.

---

## Run the System

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run live recognition:

```bash
python src/main.py
```

Expected output on the camera feed:

```text
Known: <person_name> | <distance>
```

or:

```text
Unknown | <distance>
```

Quit:

```text
Press q
```

---

## Model Training

The YOLO person detector was trained using a COCO person subset.

Training summary:

```text
Model:             YOLO11s person detector
Training images:   10,000
Validation images: 2,000
Epochs:            50
```

Final evaluation metrics:

```text
Precision: 0.782
Recall:    0.611
mAP50:     0.709
mAP50-95:  0.458
```

Model output:

```text
models/person_detector/best.pt
```

---

## Evidence

Demo videos:

```text
assets/demos/v0.3.0/v0.3.0_yolo_live_close_person_detection_demo.mp4
assets/demos/v0.5.0/live_known_unknown_face_recognition_demo.mp4
```

Bug-fix evidence:

```text
assets/bug_evidence/v0.3.1/v0.3.1_before_clothing_false_positive_bug.mp4
assets/bug_evidence/v0.3.1/v0.3.1_after_face_confirmation_false_positive_fix.mp4
```

---

## Private Data

The following data is intentionally ignored by Git:

```text
data/known_faces/*
data/face_embeddings/*
```

Only placeholder `.gitkeep` files are tracked.

Do not commit:

```text
data/known_faces/<person_name>/
data/face_embeddings/known_faces.pkl
```

---

## Version History

| Version | Milestone |
|---|---|
| v0.1.0 | Camera pipeline |
| v0.2.0 | Face detection baseline |
| v0.3.0 | YOLO close-person detection |
| v0.3.1 | Clothing false-positive fix |
| v0.4.0 | Face enrollment |
| v0.5.0 | Live known/unknown recognition |

---

## Known Limitations

- Phone-screen face images can still be recognized.
- Multiple-face handling needs a single-subject safety gate.
- Liveness / anti-spoof detection is not implemented yet.
- Recognition needs more testing under lighting, angle, and occlusion changes.

---

## Next Milestone

```text
v0.5.1 — Single-Subject Recognition Gate
```

Goal:

```text
Recognize only when exactly one valid face is present.
```

Planned behavior:

- No valid face: no recognition
- One valid face: run known/unknown recognition
- Multiple valid faces: pause recognition and show warning