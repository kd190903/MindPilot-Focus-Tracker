import streamlit as st
import cv2
import av
import threading

from streamlit_webrtc import webrtc_streamer, WebRtcMode

import detector
import tracker


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MindPilot AI",
    layout="centered"
)

st.title("🎯 MindPilot AI")
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
# SHARED DATA
# ============================================================

if "shared_data" not in st.session_state:

    st.session_state.shared_data = {
        "frame": None,
        "score": 0,
        "status": "Waiting",
        "history": []
    }


# Lock for safely accessing data between
# Streamlit and the video-processing thread
if "lock" not in st.session_state:

    st.session_state.lock = threading.Lock()


# ============================================================
# VIDEO FRAME PROCESSING
# ============================================================

def video_frame_callback(frame):

    image = frame.to_ndarray(format="bgr24")

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    face_data = detector.detect(image)

    # --------------------------------------------------------
    # FOCUS SCORE
    # --------------------------------------------------------

    score, _ = tracker.get_focus_score(face_data)

    # --------------------------------------------------------
    # SCORE HISTORY
    # --------------------------------------------------------

    with st.session_state.lock:

        history = st.session_state.shared_data["history"]

        history.append(score)

        if len(history) > 10:
            history.pop(0)

        smooth_score = int(
            sum(history) / len(history)
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    if smooth_score >= threshold:

        status = "Focused"
        text_color = (0, 255, 0)

    else:

        status = "Not Focused"
        text_color = (0, 0, 255)

    # --------------------------------------------------------
    # ADD INFORMATION TO VIDEO FRAME
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SAVE LATEST PROCESSED FRAME
    # --------------------------------------------------------

    with st.session_state.lock:

        st.session_state.shared_data["frame"] = image.copy()

        st.session_state.shared_data["score"] = smooth_score

        st.session_state.shared_data["status"] = status

    # --------------------------------------------------------
    # RETURN FRAME TO BROWSER
    # --------------------------------------------------------

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


# ============================================================
# START / STOP CONTROLS
# ============================================================

if "tracking" not in st.session_state:
    st.session_state.tracking = False


col1, col2 = st.columns(2)


# ============================================================
# START BUTTON
# ============================================================

with col1:

    if st.button(
        "▶️ Start",
        use_container_width=True
    ):

        st.session_state.tracking = True

        with st.session_state.lock:

            st.session_state.shared_data["frame"] = None
            st.session_state.shared_data["score"] = 0
            st.session_state.shared_data["status"] = "Waiting"
            st.session_state.shared_data["history"] = []

        st.rerun()


# ============================================================
# STOP BUTTON
# ============================================================

with col2:

    if st.button(
        "⏹️ Stop",
        use_container_width=True
    ):

        # Save the latest frame before stopping
        with st.session_state.lock:

            latest_frame = st.session_state.shared_data["frame"]

            latest_score = st.session_state.shared_data["score"]

            latest_status = st.session_state.shared_data["status"]

            if latest_frame is not None:

                st.session_state.final_screenshot = cv2.cvtColor(
                    latest_frame,
                    cv2.COLOR_BGR2RGB
                )

                st.session_state.final_score = latest_score

                st.session_state.final_status = latest_status

        st.session_state.tracking = False

        st.rerun()


# ============================================================
# INITIALIZE FINAL RESULT STATE
# ============================================================

if "final_screenshot" not in st.session_state:

    st.session_state.final_screenshot = None

if "final_score" not in st.session_state:

    st.session_state.final_score = None

if "final_status" not in st.session_state:

    st.session_state.final_status = None


# ============================================================
# LIVE CAMERA
# ============================================================

if st.session_state.tracking:

    st.info(
        "📷 Allow camera access in your browser to start tracking."
    )

    webrtc_streamer(

        key="mindpilot-camera",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=video_frame_callback,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True
    )


    # --------------------------------------------------------
    # LIVE SCORE DISPLAY
    # --------------------------------------------------------

    with st.session_state.lock:

        current_score = st.session_state.shared_data["score"]

        current_status = st.session_state.shared_data["status"]


    st.metric(
        "🎯 Focus Score",
        f"{current_score}%"
    )


    if current_status == "Focused":

        st.success(
            "🟢 Status: Focused"
        )

    elif current_status == "Not Focused":

        st.error(
            "🔴 Status: Not Focused"
        )

    else:

        st.info(
            "Status: Waiting"
        )


    st.progress(
        min(
            max(current_score, 0),
            100
        )
    )


# ============================================================
# TRACKING STOPPED
# ============================================================

else:

    if st.session_state.final_screenshot is not None:

        st.subheader(
            "📸 Focus Session Snapshot"
        )

        st.image(
            st.session_state.final_screenshot,
            caption="Final focus snapshot",
            use_container_width=True
        )


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        if st.session_state.final_score is not None:

            st.metric(
                "🎯 Final Focus Score",
                f"{st.session_state.final_score}%"
            )


        # ----------------------------------------------------
        # FINAL STATUS
        # ----------------------------------------------------

        if st.session_state.final_status == "Focused":

            st.success(
                "✅ Session Status: Focused"
            )

        elif st.session_state.final_status == "Not Focused":

            st.warning(
                "⚠️ Session Status: Not Focused"
            )


    else:

        st.info(
            "📷 Camera is stopped. Click ▶️ Start to begin tracking."
        )

        st.metric(
            "🎯 Focus Score",
            "0%"
        )

        st.write(
            "Status: Waiting"
        )

        st.progress(0)