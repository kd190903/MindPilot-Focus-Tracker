import math

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_focus_score(face_landmarks):
    if face_landmarks is None:
        return 0, "No Face"

    landmarks = face_landmarks.landmark
    score = 0
    status = []

    # 👁️ Eye detection
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear > 0.25:
        score += 40
    else:
        status.append("Eyes Closed")

    # 👀 Gaze detection
    nose_x = landmarks[1].x
    if 0.4 < nose_x < 0.6:
        score += 30
    else:
        status.append("Looking Away")

    # 🧍 Head alignment (simple)
    left_eye_x = landmarks[33].x
    right_eye_x = landmarks[263].x

    if abs(left_eye_x - right_eye_x) < 0.1:
        score += 20
    else:
        status.append("Head Tilt")

    # 🙂 Face present
    score += 10

    # Final status
    if score > 70:
        final_status = "Focused"
    elif score > 40:
        final_status = "Partially Focused"
    else:
        final_status = "Not Focused"

    return score, final_status