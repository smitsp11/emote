import cv2
import mediapipe as mp
import numpy as np
import time
import random

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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

        cv2.imshow('Vibe Check Engine', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
