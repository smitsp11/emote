# Setup Guide

Due to dependency conflicts with NumPy 2.x and MediaPipe API changes, this project requires a **virtual environment** with specific package versions.

## Quick Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe==0.10.9 numpy
```

## Why These Steps?

| Issue | Solution |
|-------|----------|
| NumPy 2.x breaks matplotlib/mediapipe | Virtual environment isolates deps |
| MediaPipe 0.10.32+ removed `solutions` API | Pin to `mediapipe==0.10.9` |

## Running the App

```bash
# Always activate venv first
source venv/bin/activate

# Run the engine
python vibe_check.py
```

## Notes

- The Patrick image should be named `patrick.jpg` (or update the code for `.png`)
- Press `q` to quit the webcam window
- Ignore the matplotlib cache warnings — they're harmless
