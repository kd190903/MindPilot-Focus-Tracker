import streamlit as st
import cv2
import detector
import tracker

st.set_page_config(page_title="Focus Tracker", layout="centered")

st.title("🎯 AI Focus Tracker")
st.write("Real-time attention monitoring using computer vision")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])
score_text = st.empty()
status_text = st.empty()

cap = cv2.VideoCapture(0)

score_history = []

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    # Detect face
    face_data = detector.detect(frame)

    # Get score
    score, status = tracker.get_focus_score(face_data)

    # Smooth score
    score_history.append(score)
    if len(score_history) > 10:
        score_history.pop(0)

    smooth_score = sum(score_history) // len(score_history)

    # Display text
    score_text.markdown(f"### 🔢 Focus Score: {smooth_score}")
    status_text.markdown(f"### 📌 Status: {status}")

    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

cap.release()