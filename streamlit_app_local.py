import streamlit as st
import cv2
import time

import detector
import tracker


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Focus Tracker",
    layout="centered"
)

st.title("🎯 AI Focus Tracker")
st.write("Real-time attention monitoring using computer vision")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

threshold = st.sidebar.slider(
    "Focus Threshold",
    0,
    100,
    70
)


# ============================================================
# SESSION STATE
# ============================================================

if "run" not in st.session_state:
    st.session_state.run = False

if "cap" not in st.session_state:
    st.session_state.cap = None

if "score_history" not in st.session_state:
    st.session_state.score_history = []

if "final_screenshot" not in st.session_state:
    st.session_state.final_screenshot = None

if "final_score" not in st.session_state:
    st.session_state.final_score = None

if "final_status" not in st.session_state:
    st.session_state.final_status = None


# ============================================================
# START / STOP BUTTONS
# ============================================================

col1, col2 = st.columns(2)


# ---------------- START ----------------

with col1:

    if st.button("▶️ Start", use_container_width=True):

        # Start tracking
        st.session_state.run = True

        # Clear previous screenshot
        st.session_state.final_screenshot = None
        st.session_state.final_score = None
        st.session_state.final_status = None

        # Clear previous score history
        st.session_state.score_history = []

        # Open camera
        if st.session_state.cap is None:

            st.session_state.cap = cv2.VideoCapture(0)


# ---------------- STOP ----------------

with col2:

    if st.button("⏹️ Stop", use_container_width=True):

        # Stop tracking
        st.session_state.run = False

        # Release camera
        if st.session_state.cap is not None:

            st.session_state.cap.release()
            st.session_state.cap = None


# ============================================================
# UI PLACEHOLDERS
# ============================================================

FRAME_WINDOW = st.empty()

score_box = st.empty()

status_box = st.empty()

progress_bar = st.progress(0)


# ============================================================
# LIVE CAMERA PROCESSING
# ============================================================

if st.session_state.run:

    cap = st.session_state.cap

    # --------------------------------------------------------
    # CHECK CAMERA
    # --------------------------------------------------------

    if cap is None or not cap.isOpened():

        st.error("❌ Camera could not be opened.")

        st.session_state.run = False

    else:

        # ----------------------------------------------------
        # READ FRAME
        # ----------------------------------------------------

        ret, frame = cap.read()

        if not ret:

            st.error("❌ Unable to read camera frame.")

        else:

            # ------------------------------------------------
            # FACE DETECTION
            # ------------------------------------------------

            face_data = detector.detect(frame)

            # ------------------------------------------------
            # FOCUS SCORE
            # ------------------------------------------------

            score, _ = tracker.get_focus_score(face_data)

            # ------------------------------------------------
            # SCORE HISTORY
            # ------------------------------------------------

            st.session_state.score_history.append(score)

            if len(st.session_state.score_history) > 10:

                st.session_state.score_history.pop(0)

            smooth_score = int(
                sum(st.session_state.score_history)
                / len(st.session_state.score_history)
            )

            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            if smooth_score >= threshold:

                status = "Focused"
                status_color = "green"
                text_color = (0, 255, 0)

            else:

                status = "Not Focused"
                status_color = "red"
                text_color = (0, 0, 255)

            # ------------------------------------------------
            # DISPLAY SCORE
            # ------------------------------------------------

            score_box.metric(
                "🎯 Focus Score",
                f"{smooth_score}%"
            )

            # ------------------------------------------------
            # DISPLAY STATUS
            # ------------------------------------------------

            status_box.markdown(
                f"""
                <h3 style="color:{status_color};">
                Status: {status}
                </h3>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # PROGRESS BAR
            # ------------------------------------------------

            progress_bar.progress(
                min(max(smooth_score, 0), 100)
            )

            # =================================================
            # CREATE FINAL SCREENSHOT
            # =================================================

            final_frame = frame.copy()

            # Focus score
            cv2.putText(
                final_frame,
                f"Focus Score: {smooth_score}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
                cv2.LINE_AA
            )

            # Status
            cv2.putText(
                final_frame,
                f"Status: {status}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # SAVE LATEST FRAME
            # ------------------------------------------------

            st.session_state.final_screenshot = cv2.cvtColor(
                final_frame,
                cv2.COLOR_BGR2RGB
            )

            st.session_state.final_score = smooth_score

            st.session_state.final_status = status

            # ------------------------------------------------
            # DISPLAY LIVE FRAME
            # ------------------------------------------------

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            FRAME_WINDOW.image(
                frame_rgb,
                channels="RGB"
            )

        # ----------------------------------------------------
        # SMALL DELAY
        # ----------------------------------------------------

        time.sleep(0.03)

        # ----------------------------------------------------
        # REFRESH
        # ----------------------------------------------------

        st.rerun()


# ============================================================
# TRACKING STOPPED
# ============================================================

else:

    # --------------------------------------------------------
    # DISPLAY FINAL SCREENSHOT
    # --------------------------------------------------------

    if st.session_state.final_screenshot is not None:

        st.subheader("📸 Focus Session Snapshot")

        st.image(
            st.session_state.final_screenshot,
            caption="Final focus snapshot",
            use_container_width=True
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = st.session_state.final_score

        final_status = st.session_state.final_status

        if final_score is not None:

            st.metric(
                "🎯 Final Focus Score",
                f"{final_score}%"
            )

        if final_status is not None:

            if final_status == "Focused":

                st.success(
                    f"✅ Session Status: {final_status}"
                )

            else:

                st.warning(
                    f"⚠️ Session Status: {final_status}"
                )

    else:

        # ----------------------------------------------------
        # INITIAL SCREEN
        # ----------------------------------------------------

        FRAME_WINDOW.info(
            "📷 Camera is stopped. Click ▶️ Start to begin tracking."
        )

        score_box.metric(
            "🎯 Focus Score",
            "0%"
        )

        status_box.markdown(
            "### Status: Waiting"
        )

        progress_bar.progress(0)