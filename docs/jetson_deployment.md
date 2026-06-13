# Jetson Deployment Notes

This project has a Jetson-specific deployment path because the NVIDIA Jetson Orin runtime differs from the PC development environment.

## Important Jetson Differences

- PyTorch must be installed from the Jetson-compatible wheel index.
- CUDA CUPTI may be required for PyTorch to import correctly.
- OpenCV is installed through the system package manager instead of `opencv-python`.
- The Haar cascade path needs a fallback because Jetson OpenCV may not expose `cv2.data.haarcascades`.
- Face embeddings should be rebuilt on the Jetson instead of copying `known_faces.pkl` from the PC.
- Blink/liveness thresholds were calibrated separately for Jetson because face-landmark processing is slower on edge hardware.

## Required System Packages

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-dev python3-opencv opencv-data build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev cuda-cupti-12-6


