# =========================================================
# 🏭 Industrial Defect Detection using YOLOv8
# =========================================================
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Industrial Defect Detection", layout="wide")
st.title("🏭 Industrial Defect Detection Dashboard (YOLOv8)")
st.write("Detect cracks, rust, or other surface defects in real time using YOLOv8.")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    model_path = "defect_detector6/weights/best.pt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.stop()

    model = YOLO(model_path)
    return model

model = load_model()
st.success("✅ YOLOv8 Model loaded successfully!")

# =========================================================
# SIDEBAR OPTIONS
# =========================================================
st.sidebar.header("⚙️ Options")
option = st.sidebar.radio(
    "Select Input Mode",
    ["Capture from Camera", "Upload Image", "Upload Video (MP4)", "Live Camera Detection"],
    index=1
)
save_output = st.sidebar.checkbox("💾 Save Output", value=False)

# =========================================================
# 1️⃣ CAPTURE FROM CAMERA
# =========================================================
if option == "Capture from Camera":
    st.info("📸 Capture an image directly from your webcam.")
    camera_image = st.camera_input("Take a photo", key="camera_capture")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        if st.button("🔍 Detect Defects"):
            with st.spinner("Detecting defects..."):
                results = model(image)
                annotated = results[0].plot()
                st.image(annotated, channels="BGR", use_container_width=True)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.subheader("📊 Detection Details:")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"**Class:** {model.names[cls_id]} | **Confidence:** {conf*100:.2f}%")
                else:
                    st.warning("No defects detected.")

                if save_output:
                    cv2.imwrite("output_image.jpg", annotated)
                    st.success("💾 Saved as `output_image.jpg`")

# =========================================================
# 2️⃣ UPLOAD IMAGE
# =========================================================
elif option == "Upload Image":
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Defects"):
            with st.spinner("Detecting defects..."):
                results = model(image)
                annotated = results[0].plot()
                st.image(annotated, channels="BGR", use_container_width=True)

                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.subheader("📊 Detection Details:")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"**Class:** {model.names[cls_id]} | **Confidence:** {conf*100:.2f}%")
                else:
                    st.warning("No defects detected.")

                if save_output:
                    cv2.imwrite("output_image.jpg", annotated)
                    st.success("💾 Saved as `output_image.jpg`")

# =========================================================
# 3️⃣ UPLOAD VIDEO
# =========================================================
elif option == "Upload Video (MP4)":
    uploaded_video = st.file_uploader("📹 Upload a Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        if st.button("▶️ Start Detection"):
            cap = cv2.VideoCapture(video_path)
            stframe = st.empty()

            if save_output:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (
                    int(cap.get(3)), int(cap.get(4))))
                st.info("💾 Saving output video...")

            with st.spinner("Processing video..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame, channels="BGR", use_container_width=True)

                    if save_output:
                        out.write(annotated_frame)

                cap.release()
                if save_output:
                    out.release()
                    st.success("🎬 Output video saved as `output_video.avi`")

            st.success("✅ Video Processing Complete!")

# =========================================================
# 4️⃣ LIVE CAMERA DETECTION (REAL-TIME)
# =========================================================
elif option == "Live Camera Detection":
    st.info("🎥 Starting live defect detection (allow camera access).")

    # Ask user if they want to save the live detection output
    save_live = st.radio(
        "💾 Do you want to save detected live video output?",
        ["No", "Yes"],
        index=0,
        horizontal=True
    )

    class YOLOVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.save_output = (save_live == "Yes")
            self.frames = []
            self.frame_size = None

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = self.model(img)
            annotated = results[0].plot()

            # Store frames if saving is enabled
            if self.save_output:
                self.frames.append(annotated)
                if self.frame_size is None:
                    self.frame_size = (annotated.shape[1], annotated.shape[0])

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        def save_video(self):
            """Save all collected frames into a video file."""
            if self.save_output and self.frames:
                output_path = "live_output.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_path, fourcc, 20.0, self.frame_size)
                for f in self.frames:
                    out.write(f)
                out.release()
                return output_path
            return None

    processor_instance = YOLOVideoProcessor()

    ctx = webrtc_streamer(
        key="live_yolo_defect",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: processor_instance,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Button to stop and save video (if user chose to save)
    if save_live == "Yes" and ctx.video_processor:
        if st.button("🛑 Stop & Save Video"):
            path = ctx.video_processor.save_video()
            if path:
                st.success(f"🎬 Live detection saved as `{path}`")
            else:
                st.warning("⚠️ No frames captured yet.")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("⚙️ Built by Pavan Kalyan | YOLOv8 + Streamlit + WebRTC")

