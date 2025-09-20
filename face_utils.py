# face_utils.py
import numpy as np
from scipy.spatial import distance as dist

# --- Constants for challenge thresholds (made more lenient for better detection) ---
EYE_AR_THRESH = 0.25  # Increased from 0.22 for easier blinking detection
EYE_AR_CONSEC_FRAMES = 2
MOUTH_AR_THRESH = 0.50  # Decreased from 0.60 for easier mouth opening detection

def get_eye_aspect_ratio(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_mouth_aspect_ratio(mouth):
    """Calculates the Mouth Aspect Ratio (MAR) for the mouth."""
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (2.0 * D)
    return mar

def get_head_pose(shape):
    """
    A simplified head pose estimation based on the nose's position
    relative to the line connecting the two eyes.
    Returns 'left', 'right', or 'center'.
    """
    # Define points for eyes and nose
    left_eye_pts = shape[36:42]
    right_eye_pts = shape[42:48]
    nose_tip = shape[33]

    # Calculate eye centers
    left_eye_center = left_eye_pts.mean(axis=0)
    right_eye_center = right_eye_pts.mean(axis=0)

    # Calculate the midpoint between the eyes
    eyes_center = (left_eye_center + right_eye_center) / 2.0

    # Calculate deviation of the nose from the center
    # A positive value means the head is turned to the user's left (camera's right)
    deviation = nose_tip[0] - eyes_center[0]
    
    if deviation > 8:  # Reduced from 12 for easier detection
        return "right"
    elif deviation < -8:  # Reduced from -12 for easier detection
        return "left"
    else:
        return "center"

def verify_challenge(challenge_name, landmarks):
    """Verifies if the user is performing the correct action based on landmarks."""
    if challenge_name == "blink":
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = get_eye_aspect_ratio(left_eye)
        right_ear = get_eye_aspect_ratio(right_eye)
        # Check if either eye is closed
        if left_ear < EYE_AR_THRESH or right_ear < EYE_AR_THRESH:
            return True
            
    elif challenge_name == "turn_left":
        pose = get_head_pose(landmarks)
        if pose == "left":
            return True

    elif challenge_name == "turn_right":
        pose = get_head_pose(landmarks)
        if pose == "right":
            return True

    elif challenge_name == "open_mouth":
        mouth = landmarks[48:68]
        mar = get_mouth_aspect_ratio(mouth)
        if mar > MOUTH_AR_THRESH:
            return True

    return False