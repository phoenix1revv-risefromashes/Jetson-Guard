# Jetson FaceGuard

## Edge-Based Person Recognition for Safety and Access Monitoring Using Jetson Orin

Jetson FaceGuard is a final deployable edge-AI vision system for real-time safety and access monitoring. It detects a close human subject from a live camera feed, confirms that the detected person region contains a real face, verifies liveness through blink detection, rejects spoof attempts, and recognizes whether the person is known or unknown directly on NVIDIA Jetson Orin.

The goal of this project was to build a complete local recognition pipeline, not just a model demo. The final system includes person detection, face confirmation, multiple-face safety handling, blink-based liveness verification, known/unknown recognition, Jetson deployment support, and demo-tested bug fixes for real-world edge cases.


<p align="center">
  <a href="assets/">
    <img src="assets/architecture/system_architecture_final.png" alt="Project Architecture Flowchart" width="100%">
  </a>
</p>

<p align="center">
  <sub><b>Click the architecture image to browse demos, bug-fix evidence, and project assets.</b></sub>
</p>



```text
v1.0.0-jetson — Final NVIDIA Jetson Orin Edge Deployment Version
```

---

## Final Product 

Jetson FaceGuard is a completed edge-AI product, not just a model demo.

It is ready for immediate deployment in a controlled local environment using the final Jetson release.

The final system includes:

- Live camera input
- YOLO-based close-person detection
- Face confirmation
- Multiple-face safety handling
- Blink-based liveness verification
- Known/unknown recognition
- Spoof-image rejection
- Operator-facing monitoring output
- PC development/testing release
- NVIDIA Jetson Orin deployment release

---



## Demos / Bug Fixes

The project was validated through demos, bug-fix evidence, and Jetson deployment testing.

### Demo Cases

```text
Real face + blink            → recognition allowed
Real face without blink      → recognition blocked
Phone/computer face image    → spoof rejected
Multiple faces               → recognition paused
Clothing/person-like object  → rejected after face confirmation
Jetson deployment            → full pipeline validated on edge hardware
```

### Main Bug Fixes

- Fixed clothing/person-shaped false positives using face confirmation
- Added multiple-face safety gate
- Added blink-based liveness verification
- Added static phone/computer spoof rejection
- Fixed Jetson OpenCV Haar cascade loading issue
- Rebuilt face embeddings on Jetson for runtime compatibility
- Tuned liveness timing for Jetson hardware speed
- Solved Jetson PyTorch/CUDA/CUPTI setup issues

### Evidence Folders

```text
assets/demos/
assets/bug_evidence/
```

---
## Model Training

The YOLO person detector was trained using a COCO person subset.

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

## Other Implications / Uses

Jetson FaceGuard can be extended into several real-world edge-AI applications:

- Safety monitoring in labs, classrooms, or restricted spaces
- Local access-control systems
- Smart door/security camera prototypes
- Identity-aware edge monitoring without cloud inference
- Retail or workspace monitoring
- Human-aware robotics perception
- Multi-camera safety monitoring systems

The key lesson from this project is that real-world AI is more than training a model. A deployable system needs preprocessing, validation logic, safety gates, hardware calibration, debugging, and clear operator feedback.

---

## Final Releases

| Release | Purpose |
|---|---|
| `v1.0.0-pc` | Final PC development and testing version |
| `v1.0.0-jetson` | Final NVIDIA Jetson Orin edge deployment version |

---


```

---

## Jetson Orin Deployment

The final Jetson version was deployed and validated on NVIDIA Jetson Orin.

Jetson-specific deployment work included:

- Jetson-compatible PyTorch installation
- CUDA/CUPTI runtime setup
- OpenCV system package usage
- Haar cascade path fallback for Jetson OpenCV
- Known-face embeddings rebuilt on Jetson
- Jetson-specific blink/liveness timing calibration
- Final deployment testing on edge hardware

Final Jetson release:

```text
v1.0.0-jetson
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
| `v0.1.0` | Camera pipeline |
| `v0.2.0` | Face detection baseline |
| `v0.3.0` | YOLO close-person detection |
| `v0.3.1` | Clothing false-positive fix |
| `v0.4.0` | Face enrollment |
| `v0.5.0` | Live known/unknown recognition |
| `v1.0.0-pc` | Final PC development/testing version |
| `v1.0.0-jetson` | Final NVIDIA Jetson Orin deployment version |

---

## Technologies Used

- Python
- OpenCV
- YOLO / Ultralytics
- face_recognition
- dlib
- NumPy
- PyTorch / CUDA
- NVIDIA Jetson Orin
- Git and GitHub Releases

---

## Known Limitations

- Blink-based liveness improves spoof rejection but is not a full production-grade anti-spoofing system.
- Recognition accuracy can vary under poor lighting, occlusion, or extreme face angles.
- Multi-person scenes are intentionally paused instead of resolved through tracking.
- The current system is designed for a single-camera setup.

---

## Future Improvements

- Stronger anti-spoofing model
- Face tracking across frames
- Event logging dashboard
- Multi-camera support
- Better low-light recognition
- Access-control hardware integration
- Jetson performance optimization