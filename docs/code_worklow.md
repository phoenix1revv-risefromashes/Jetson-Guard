# Code Workflow

## Jetson FaceGuard

This document explains how the Jetson FaceGuard code runs from live camera input to final Known/Unknown output.

The README explains what the project does. This file explains how the code workflow operates internally.

---

## 1. High-Level Runtime Flow

The live system follows this order:

```text
src/main.py
↓
Camera opens live video stream
↓
Each frame is read from the camera
↓
YOLO detects person candidates
↓
Close-person filtering removes weak or distant detections
↓
Person region is cropped
↓
Face confirmation checks if a real face exists
↓
Multiple-face safety gate decides whether recognition is safe
↓
Blink-based liveness verification runs
↓
Face recognition compares embedding with known database
↓
Annotated output is shown to the operator
```

The important rule is that recognition does not happen immediately after YOLO detection.

A person detection must first pass face confirmation, single-subject validation, and liveness verification.

---

## 2. Main Entry Point

The project starts from:

```text
src/main.py
```

`main.py` controls the live runtime loop.

Its responsibilities are:

- initialize the camera
- load the detection and recognition pipeline
- read frames continuously
- send each frame into the recognition processor
- display the processed output frame
- stop safely when the user presses `q`

Simplified workflow:

```text
start program
↓
open camera
↓
load models and processors
↓
while camera is running:
    read frame
    process frame
    display output
↓
release camera
close windows
```

---

## 3. Camera Module

The camera module handles live frame capture.

Typical file:

```text
src/camera.py
```

Its responsibilities are:

- open the webcam or Jetson camera input
- read frames from the camera
- verify that frames are valid
- release the camera when the program ends

The rest of the system does not need to know the camera details. It only needs a valid frame.

---

## 4. Person Detection Stage

The person detector uses YOLO to find person-shaped regions in the frame.

Input:

```text
camera frame
```

Output:

```text
person bounding boxes
confidence scores
```

At this stage, YOLO may detect anything that looks like a person. This is why the system does not trust YOLO alone.

The code applies filtering after detection:

```text
YOLO detection
↓
confidence check
↓
close-subject height-ratio check
↓
accepted person candidate
```

This removes weak detections and subjects that are too far from the camera.

---

## 5. Person Crop Stage

After a person candidate is accepted, the system crops that region from the frame.

Input:

```text
full frame + person bounding box
```

Output:

```text
cropped person region
```

This crop is used for face confirmation.

The system does not run recognition on the full image immediately. It first verifies whether the detected person region contains a usable face.

---

## 6. Face Confirmation Stage

Face confirmation checks whether a real face exists inside the cropped person region.

This prevents false positives such as:

- hanging clothing
- background shapes
- posters
- body-only detections
- person-shaped objects

Workflow:

```text
cropped person region
↓
face detector
↓
face found?
```

If no valid face is found:

```text
reject detection
do not run recognition
```

If a face is found:

```text
continue to safety checks
```

This was one of the most important bug fixes in the project.

---

## 7. Multiple-Face Safety Gate

Before recognition, the system checks how many valid faces are present.

The logic is:

```text
0 valid faces  → no recognition
1 valid face   → continue
2+ valid faces → pause recognition
```

This prevents unsafe identity decisions when multiple people are visible.

The system intentionally pauses instead of guessing which person should be recognized.

---

## 8. Blink-Based Liveness Verification

After the system confirms that exactly one face is present, it runs blink-based liveness verification.

This step helps reject static spoof images from:

- phone screens
- computer screens
- printed face images

Workflow:

```text
valid face crop
↓
eye landmark detection
↓
eye aspect ratio calculation
↓
open eyes → closed eyes → open eyes
↓
blink confirmed
```

Recognition only runs after liveness passes.

If liveness does not pass:

```text
recognition blocked
spoof / blink required message shown
```

---

## 9. Face Recognition Stage

Once the face passes liveness verification, the system runs known/unknown recognition.

The recognizer compares the live face embedding against the local known-face database.

Input:

```text
verified live face crop
```

Local database:

```text
data/face_embeddings/known_faces.pkl
```

Output:

```text
Known: <person_name>
```

or:

```text
Unknown
```

The known-face database is private and local. It is not committed to Git.

---

## 10. Output Display

After processing, the system draws information on the live frame.

Possible outputs include:

```text
Known: <person_name>
Unknown
Blink required
Spoof detected
Multiple faces detected
Face not clear
Recognition paused
```

The final output is shown in a live OpenCV window.

The goal is not only to classify a person, but to clearly show the operator why the system accepted, rejected, or paused recognition.

---

## 11. Face Enrollment Workflow

Face enrollment is handled separately from the live recognition loop.

Typical script:

```text
scripts/enroll_face.py
```

Enrollment workflow:

```text
run enrollment script
↓
capture face images
↓
save images under person name
↓
build/update face embeddings
↓
store embeddings locally
```

Saved face images:

```text
data/known_faces/<person_name>/
```

Saved embedding database:

```text
data/face_embeddings/known_faces.pkl
```

These files are ignored by Git because they contain private identity data.

---

## 12. PC vs Jetson Code Behavior

The PC and Jetson versions share the same main system idea, but the Jetson version required deployment-specific adjustments.

PC version:

```text
v1.0.0-pc
```

Used for:

- faster development
- easier debugging
- initial model testing
- feature validation

Jetson version:

```text
v1.0.0-jetson
```

Used for:

- final edge deployment
- hardware validation
- Jetson-specific runtime calibration

Jetson-specific runtime issues included:

- OpenCV Haar cascade path differences
- PyTorch / CUDA / CUPTI setup
- face embedding compatibility
- slower liveness and landmark processing
- Jetson-specific blink calibration

---

## 13. Why the Workflow Is Layered

The code is intentionally layered because each stage answers a different question.

YOLO detection answers:

```text
Is there a person-shaped object?
```

Face confirmation answers:

```text
Is there a real face inside the detected person region?
```

Multiple-face gate answers:

```text
Is it safe to recognize only one subject?
```

Liveness detection answers:

```text
Is the face live instead of a static image?
```

Known/Unknown recognition answers:

```text
Is this verified live subject in the known-face database?
```

This is why the system is more reliable than running face recognition directly on every frame.

---

## 14. Final Runtime Rule

The final recognition rule is:

```text
Recognize only when:
    close person detected
    AND valid face confirmed
    AND exactly one face is present
    AND blink-based liveness passes
    AND valid face embedding is generated
```

If any condition fails, the system blocks or pauses recognition.

This conservative behavior is important for a safety/access monitoring system.

---

## 15. Summary

Jetson FaceGuard is structured as a full edge-AI workflow:

```text
Detection
→ Validation
→ Liveness
→ Recognition
→ Operator Feedback
```

The code workflow shows how model output becomes a safer deployable system through filtering, validation, and runtime decision logic.