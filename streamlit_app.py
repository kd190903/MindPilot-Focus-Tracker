import streamlit as st
import cv2
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode

import detector
import tracker
import shared_state


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MindPilot AI",
    layout="centered"
)

st.title("🎯 MindPilot AI")

st.write(
    "Real-time attention monitoring using computer vision"
)


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

if "session_active" not in st.session_state:
    st.session_state.session_active = False

if "final_screenshot" not in st.session_state:
    st.session_state.final_screenshot = None

if "final_score" not in st.session_state:
    st.session_state.final_score = None

if "final_status" not in st.session_state:
    st.session_state.final_status = None


# ============================================================
# VIDEO FRAME CALLBACK
# ============================================================

def video_frame_callback(frame):

    # --------------------------------------------------------
    # Convert WebRTC frame to OpenCV frame
    # --------------------------------------------------------

    image = frame.to_ndarray(format="bgr24")

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_data = detector.detect(image)

    # --------------------------------------------------------
    # FOCUS SCORE
    # --------------------------------------------------------

    score, tracker_status = tracker.get_focus_score(
        face_data
    )

    # --------------------------------------------------------
    # SCORE HISTORY
    # --------------------------------------------------------

    with shared_state.lock:

        shared_state.score_history.append(score)

        if len(shared_state.score_history) > 10:

            shared_state.score_history.pop(0)

        smooth_score = int(
            sum(shared_state.score_history)
            / len(shared_state.score_history)
        )

    # --------------------------------------------------------
    # STATUS FROM TRACKER
    # --------------------------------------------------------

    status = tracker_status

    # --------------------------------------------------------
    # COLOR
    # --------------------------------------------------------

    if status == "Focused":

        text_color = (0, 255, 0)

    elif status == "Partially Focused":

        text_color = (0, 255, 255)

    else:

        text_color = (0, 0, 255)

    # ========================================================
    # DRAW SCORE ON VIDEO
    # ========================================================

    cv2.putText(
        image,
        f"Focus Score: {smooth_score}%",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        f"Status: {status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
        cv2.LINE_AA
    )

    # ========================================================
    # SAVE LATEST FRAME
    # ========================================================

    with shared_state.lock:

        shared_state.latest_frame = image.copy()

        shared_state.latest_score = smooth_score

        shared_state.latest_status = status

    # ========================================================
    # DEBUG
    # ========================================================

    print(
        f"Score: {smooth_score} | Status: {status}"
    )

    # ========================================================
    # RETURN FRAME
    # ========================================================

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


# ============================================================
# START / STOP BUTTONS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# START
# ============================================================

with col1:

    if st.button(
        "▶️ Start",
        use_container_width=True
    ):

        # Clear old result

        st.session_state.final_screenshot = None

        st.session_state.final_score = None

        st.session_state.final_status = None

        # Reset tracking data

        with shared_state.lock:

            shared_state.latest_frame = None

            shared_state.latest_score = 0

            shared_state.latest_status = "Waiting"

            shared_state.score_history.clear()

        # Start logical session

        st.session_state.session_active = True

        st.success(
            "Tracking session started."
        )


# ============================================================
# STOP
# ============================================================

with col2:

    if st.button(
        "⏹️ Stop",
        use_container_width=True
    ):

        print("STOP BUTTON PRESSED")

        # ----------------------------------------------------
        # GET LATEST FRAME
        # ----------------------------------------------------

        with shared_state.lock:

            if shared_state.latest_frame is not None:

                final_frame = (
                    shared_state.latest_frame.copy()
                )

                final_score = (
                    shared_state.latest_score
                )

                final_status = (
                    shared_state.latest_status
                )

                # Convert BGR → RGB

                st.session_state.final_screenshot = (
                    cv2.cvtColor(
                        final_frame,
                        cv2.COLOR_BGR2RGB
                    )
                )

                st.session_state.final_score = (
                    final_score
                )

                st.session_state.final_status = (
                    final_status
                )

                print(
                    "SCREENSHOT SAVED"
                )

                print(
                    "FINAL SCORE:",
                    final_score
                )

                print(
                    "FINAL STATUS:",
                    final_status
                )

            else:

                print(
                    "NO FRAME AVAILABLE"
                )

                st.warning(
                    "No processed frame available. "
                    "Start the camera and wait a few seconds."
                )

        st.session_state.session_active = False


# ============================================================
# WEBRTC CAMERA
# ============================================================

st.subheader(
    "📷 Live Camera"
)

ctx = webrtc_streamer(

    key="mindpilot-camera",

    mode=WebRtcMode.SENDRECV,

    video_frame_callback=video_frame_callback,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True
)


# ============================================================
# LIVE ANALYSIS
# ============================================================

st.subheader(
    "📊 Live Focus Analysis"
)

with shared_state.lock:

    current_score = (
        shared_state.latest_score
    )

    current_status = (
        shared_state.latest_status
    )


# ------------------------------------------------------------
# SCORE
# ------------------------------------------------------------

st.metric(
    "🎯 Focus Score",
    f"{current_score}%"
)


# ------------------------------------------------------------
# STATUS
# ------------------------------------------------------------

if current_status == "Focused":

    st.success(
        "🟢 Status: Focused"
    )

elif current_status == "Partially Focused":

    st.warning(
        "🟡 Status: Partially Focused"
    )

elif current_status == "Not Focused":

    st.error(
        "🔴 Status: Not Focused"
    )

else:

    st.info(
        "⏳ Waiting for face detection..."
    )


# ------------------------------------------------------------
# PROGRESS
# ------------------------------------------------------------

st.progress(
    min(
        max(current_score, 0),
        100
    )
)


# ============================================================
# FINAL SESSION RESULT
# ============================================================

if st.session_state.final_screenshot is not None:

    st.divider()

    st.subheader(
        "📸 Focus Session Snapshot"
    )

    # --------------------------------------------------------
    # SCREENSHOT
    # --------------------------------------------------------

    st.image(
        st.session_state.final_screenshot,
        caption="Final focus snapshot",
        use_container_width=True
    )

    # --------------------------------------------------------
    # SCORE
    # --------------------------------------------------------

    st.metric(
        "🎯 Final Focus Score",
        f"{st.session_state.final_score}%"
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if (
        st.session_state.final_status
        == "Focused"
    ):

        st.success(
            "✅ Session Status: Focused"
        )

    elif (
        st.session_state.final_status
        == "Partially Focused"
    ):

        st.warning(
            "🟡 Session Status: Partially Focused"
        )

    else:

        st.error(
            "🔴 Session Status: Not Focused"
        )