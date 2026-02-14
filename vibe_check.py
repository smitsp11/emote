import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your Patrick Image (Ensure 'patrick.png' is in the folder)
# We handle the image loading safely
try:
    patrick_img = cv2.imread('patrick.png', -1) # -1 to keep alpha channel (transparency)
    if patrick_img is None:
        raise FileNotFoundError
    # Resize patrick to be a reasonable size (e.g., 200px wide)
    h, w = patrick_img.shape[:2]
    scale = 200 / w
    patrick_img = cv2.resize(patrick_img, (200, int(h * scale)))
except:
    print("Warning: 'patrick.png' not found. Creating a placeholder.")
    patrick_img = np.zeros((100, 100, 4), dtype=np.uint8)
    patrick_img[:] = (0, 0, 255, 255) # Red square

# --- Load the Rock meme and pre-cache at multiple sizes ---
ROCK_MIN_SIZE = 60
ROCK_MAX_SIZE = 400
ROCK_SIZE_STEP = 10

def load_rock_meme_sizes():
    """Load the Rock meme image and cache it at multiple sizes with alpha channel."""
    rock_path = os.path.join(SCRIPT_DIR, 'rock meme.jpg')
    rock_bgr = cv2.imread(rock_path)
    if rock_bgr is None:
        print("Warning: 'rock meme.jpg' not found! Using red placeholder.")
        rock_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        rock_bgr[:] = (0, 0, 255)
    
    # Add alpha channel (fully opaque)
    rock_bgra = cv2.cvtColor(rock_bgr, cv2.COLOR_BGR2BGRA)
    
    sizes = list(range(ROCK_MIN_SIZE, ROCK_MAX_SIZE + 1, ROCK_SIZE_STEP))
    cache = {}
    oh, ow = rock_bgra.shape[:2]
    aspect = oh / ow
    for size in sizes:
        new_w = size
        new_h = int(size * aspect)
        resized = cv2.resize(rock_bgra, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cache[size] = resized
    return cache, sizes

print("Loading Rock meme at multiple sizes...")
rock_cache, rock_sizes = load_rock_meme_sizes()
print(f"  Cached {len(rock_cache)} sizes.")

# --- Logic Variables ---
# For 6,7 Gesture
left_hand_y_history = []
right_hand_y_history = []
HISTORY_LEN = 10 # Frames to track for motion
motion_threshold = 0.05 # Sensitivity for "movement"

# For Tongue Gesture
tongue_start_time = None
TONGUE_HOLD_DURATION = 1.0 # Seconds
MOUTH_OPEN_THRESHOLD = 0.5 # Threshold for mouth open ratio

# For Rock Eyebrow Gesture — self-calibrating baseline approach
eyebrow_start_time = None
CALIBRATION_FRAMES = 30  # Number of frames to calibrate neutral position
calibration_samples_left = []
calibration_samples_right = []
baseline_left = None   # Neutral left brow-to-eye distance (normalized)
baseline_right = None  # Neutral right brow-to-eye distance (normalized)
EYEBROW_RAISE_SENSITIVITY = 1.4  # How much above baseline counts as "raised" (1.4 = 40% higher)

# Reaction States
current_reaction = None
reaction_timer = 0
patrick_x_pos = 0
patrick_direction = 1 # 1 for right, -1 for left

def overlay_image_alpha(img, img_overlay, x, y):
    """Overlays a PNG with transparency onto the video frame."""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]
    alpha_s = img_overlay[y1o:y2o, x1o:x2o, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, channels):
        img[y1:y2, x1:x2, c] = (alpha_s * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_l * img[y1:y2, x1:x2, c])

def calculate_mouth_ratio(landmarks):
    # Indices for upper and lower lip (inner)
    # Upper: 13, Lower: 14
    top = landmarks[13]
    bottom = landmarks[14]
    
    # Distance between lips
    distance = np.sqrt((top.x - bottom.x)**2 + (top.y - bottom.y)**2)
    return distance

def get_brow_distances(landmarks):
    """Get normalized eyebrow-to-eye distances on each side.
    
    Uses multiple landmark points per eyebrow for robustness.
    Distances are normalized by face height (chin to forehead) to be
    scale-invariant.
    
    Returns (left_normalized, right_normalized) or None if landmarks invalid.
    """
    # Face height for normalization: forehead (10) to chin (152)
    forehead = landmarks[10]
    chin = landmarks[152]
    face_height = abs(forehead.y - chin.y)
    if face_height < 0.01:
        return None
    
    # Left eyebrow landmarks (top): 63, 65, 66, 70
    # Left eye top landmarks: 159, 145
    left_brow_pts = [landmarks[i] for i in [63, 65, 66, 70]]
    left_eye_pts = [landmarks[i] for i in [159, 145]]
    left_brow_y = np.mean([p.y for p in left_brow_pts])
    left_eye_y = np.mean([p.y for p in left_eye_pts])
    left_dist = (left_eye_y - left_brow_y) / face_height  # Positive = brow above eye
    
    # Right eyebrow landmarks (top): 293, 295, 296, 300
    # Right eye top landmarks: 386, 374
    right_brow_pts = [landmarks[i] for i in [293, 295, 296, 300]]
    right_eye_pts = [landmarks[i] for i in [386, 374]]
    right_brow_y = np.mean([p.y for p in right_brow_pts])
    right_eye_y = np.mean([p.y for p in right_eye_pts])
    right_dist = (right_eye_y - right_brow_y) / face_height
    
    return (left_dist, right_dist)

def detect_eyebrow_raise(left_dist, right_dist, baseline_l, baseline_r):
    """Detect if one eyebrow is raised relative to calibrated baseline.
    
    Returns ('left', ratio), ('right', ratio), or (None, 0).
    """
    if baseline_l is None or baseline_r is None:
        return (None, 0)
    
    # How much each eyebrow is raised relative to its own baseline
    left_ratio = left_dist / baseline_l if baseline_l > 0.001 else 1.0
    right_ratio = right_dist / baseline_r if baseline_r > 0.001 else 1.0
    
    # One eyebrow must be raised AND the other should be near baseline
    left_raised = left_ratio > EYEBROW_RAISE_SENSITIVITY
    right_raised = right_ratio > EYEBROW_RAISE_SENSITIVITY
    
    if left_raised and not right_raised:
        return ('left', left_ratio)
    elif right_raised and not left_raised:
        return ('right', right_ratio)
    elif left_raised and right_raised:
        # Both raised — still counts (like The Rock tilting head)
        return ('both', max(left_ratio, right_ratio))
    else:
        return (None, 0)

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Detect 6,7 Gesture (Alternating Hands)
        if results.left_hand_landmarks and results.right_hand_landmarks:
            # Get wrist Y positions
            l_wrist_y = results.left_hand_landmarks.landmark[0].y
            r_wrist_y = results.right_hand_landmarks.landmark[0].y
            
            left_hand_y_history.append(l_wrist_y)
            right_hand_y_history.append(r_wrist_y)
            
            if len(left_hand_y_history) > HISTORY_LEN:
                left_hand_y_history.pop(0)
                right_hand_y_history.pop(0)

            # Analyze Motion: Look for variance (shaking)
            l_var = np.var(left_hand_y_history)
            r_var = np.var(right_hand_y_history)
            
            # If both hands are moving significantly
            if l_var > 0.001 and r_var > 0.001:
                # Check for "alternating" phase could be complex, 
                # but usually high variance on both hands = "Spamming"
                current_reaction = "67"
                reaction_timer = time.time() # Reset timer to keep it alive

        # 2. Detect Tongue (Mouth Open)
        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            mouth_openness = calculate_mouth_ratio(landmarks)
            
            # You might need to tune '0.05' based on your camera distance
            if mouth_openness > 0.05: 
                if tongue_start_time is None:
                    tongue_start_time = time.time()
                elif time.time() - tongue_start_time > TONGUE_HOLD_DURATION:
                    current_reaction = "PATRICK"
                    reaction_timer = time.time()
            else:
                tongue_start_time = None

        # 3. Detect Rock Eyebrow (One eyebrow raised) — self-calibrating baseline
        if results.face_landmarks:
            landmarks = results.face_landmarks.landmark
            brow_dists = get_brow_distances(landmarks)
            
            if brow_dists is not None:
                left_dist, right_dist = brow_dists
                
                # Calibration phase: collect baseline samples
                if len(calibration_samples_left) < CALIBRATION_FRAMES:
                    calibration_samples_left.append(left_dist)
                    calibration_samples_right.append(right_dist)
                    
                    if len(calibration_samples_left) == CALIBRATION_FRAMES:
                        baseline_left = np.mean(calibration_samples_left)
                        baseline_right = np.mean(calibration_samples_right)
                        print(f"Eyebrow baseline calibrated! L={baseline_left:.4f} R={baseline_right:.4f}")
                    
                    # Show calibration progress
                    progress = len(calibration_samples_left) / CALIBRATION_FRAMES
                    cv2.putText(image, f"Calibrating eyebrows... {int(progress*100)}%", 
                               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # Detection phase
                    raised_side, raise_ratio = detect_eyebrow_raise(left_dist, right_dist, baseline_left, baseline_right)
                    
                    if raised_side is not None:
                        if eyebrow_start_time is None:
                            eyebrow_start_time = time.time()
                        current_reaction = "ROCK"
                        reaction_timer = time.time()
                    else:
                        eyebrow_start_time = None
                    
                    # Debug: show brow ratios
                    l_ratio = left_dist / baseline_left if baseline_left and baseline_left > 0.001 else 0
                    r_ratio = right_dist / baseline_right if baseline_right and baseline_right > 0.001 else 0
                    debug_color = (0, 255, 0) if raised_side else (128, 128, 128)
                    cv2.putText(image, f"Brow L:{l_ratio:.2f} R:{r_ratio:.2f}", 
                               (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, debug_color, 1)

        # --- Render Reactions ---
        
        # Reaction: 6, 7
        if current_reaction == "67":
            # Show for 2 seconds after last trigger
            if time.time() - reaction_timer < 2.0:
                for _ in range(5): # Draw 5 random numbers
                    rx = random.randint(50, w-50)
                    ry = random.randint(50, h-50)
                    num = random.choice(["6", "7"])
                    color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                    cv2.putText(image, num, (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
            else:
                current_reaction = None

        # Reaction: Patrick
        elif current_reaction == "PATRICK":
            if time.time() - reaction_timer < 4.0: # Show for 4 seconds
                # Move Patrick
                patrick_x_pos += 15 * patrick_direction
                
                # Bounce off edges
                if patrick_x_pos > w - 100 or patrick_x_pos < 0:
                    patrick_direction *= -1
                
                # Overlay Patrick
                overlay_image_alpha(image, patrick_img, patrick_x_pos, h - 250)
            else:
                current_reaction = None
                patrick_x_pos = 0 # Reset

        # Reaction: Rock Eyebrow — growing Rock meme
        elif current_reaction == "ROCK":
            if time.time() - reaction_timer < 2.0:  # Persist 2s after release
                # Calculate how long eyebrow has been raised
                if eyebrow_start_time is not None:
                    hold_duration = time.time() - eyebrow_start_time
                else:
                    hold_duration = 0
                
                # Grow image from ROCK_MIN_SIZE to ROCK_MAX_SIZE over ~3 seconds
                t = min(hold_duration / 3.0, 1.0)  # Normalized 0→1
                target_size = int(ROCK_MIN_SIZE + t * (ROCK_MAX_SIZE - ROCK_MIN_SIZE))
                
                # Snap to nearest cached size
                nearest_size = min(rock_sizes, key=lambda s: abs(s - target_size))
                rock_img = rock_cache[nearest_size]
                rh, rw = rock_img.shape[:2]
                
                # Position on face using Holistic face landmarks
                nose_x, nose_y = w // 2, h // 2  # fallback center
                if results.face_landmarks:
                    nose = results.face_landmarks.landmark[1]
                    nose_x = int(nose.x * w)
                    nose_y = int(nose.y * h)
                cx = nose_x - rw // 2
                cy = nose_y - rh // 2
                overlay_image_alpha(image, rock_img, cx, cy)
            else:
                current_reaction = None

        cv2.imshow('Vibe Check Engine', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
