import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# Drawing utilities (optional)
mp_drawing = mp.solutions.drawing_utils


def detect(frame):
    """
    Detect facial landmarks using MediaPipe
    Returns landmarks if face detected, else None
    """

    # Convert BGR → RGB (important for mediapipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        # Return first face landmarks
        return results.multi_face_landmarks[0]
    
    return None