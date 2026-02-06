# 🎭 Vibe Check Engine

A custom gesture reaction engine using OpenCV and MediaPipe for expressive, meme-based video overlays.

## Features

### The "6,7" (The Hype) 🙌
- **Trigger:** Rapidly move both hands up and down ("spamming")
- **Effect:** Chaotic overlay of "6" and "7" numbers popping randomly across the screen

### The "Patrick" (The Mood) 😛
- **Trigger:** Open mouth wide (simulating tongue out) for > 1 second
- **Effect:** Patrick Star slides horizontally across the video feed

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Add your `patrick.png` image to this folder (with transparency for best effect)
2. Run the engine:

```bash
python vibe_check.py
```

3. Press `q` to quit

## Controls

| Gesture | Description | Effect Duration |
|---------|-------------|-----------------|
| 🙌 Both hands shaking | "6,7" overlay | 2 seconds |
| 😮 Mouth open (held) | Patrick slides | 4 seconds |

## Tuning

- **Hand sensitivity:** Adjust `motion_threshold` and variance checks (`l_var > 0.001`)
- **Mouth sensitivity:** Adjust `mouth_openness > 0.05` threshold
- **Tongue hold time:** Change `TONGUE_HOLD_DURATION` (default: 1.0 sec)
