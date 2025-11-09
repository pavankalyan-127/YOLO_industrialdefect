import os
# Prevent OpenCV GUI issues on Streamlit Cloud
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import numpy as np


# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Industrial Defect Detection", layout="wide")
st.title("🏭 Industrial Defect Detection using YOLOv8")
st.write("Upload an image or video to detect cracks, rusts, or other surface defects.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    # ✅ Correct relative path to your trained model
    model_path = "defect_detector6/weights/best.pt"

    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}. Please check your repository path.")
        st.stop()

    model = YOLO(model_path)
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# -------------------- SIDEBAR OPTIONS --------------------
st.sidebar.header("⚙️ Options")
input_type = st.sidebar.radio("Select Input Type", ["Image", "Video"])
save_output = st.sidebar.checkbox("💾 Save Output", value=False)

# -------------------- IMAGE DETECTION --------------------
if input_type == "Image":
    uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Detect Defects"):
            with st.spinner("Detecting defects..."):
                results = model(image)
                res_plotted = results[0].plot()  # returns numpy array (BGR)

                # Convert to RGB for Streamlit display
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, caption="Detected Defects", use_column_width=True)

                # Show detection info
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.subheader("📊 Detection Details:")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"**Class:** {model.names[cls_id]} | **Confidence:** {conf*100:.2f}%")
                else:
                    st.warning("No defects detected.")

                # Save result
                if save_output:
                    output_path = "output_image.jpg"
                    cv2.imwrite(output_path, res_plotted)
                    st.success(f"💾 Image saved as `{output_path}`")

                st.success("✅ Detection Complete!")

# -------------------- VIDEO DETECTION --------------------
elif input_type == "Video":
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

                    # Stream to Streamlit
                    stframe.image(annotated_frame, channels="BGR", use_column_width=True)

                    if save_output:
                        out.write(annotated_frame)

                cap.release()
                if save_output:
                    out.release()
                    st.success("🎬 Output video saved as `output_video.avi`")

            st.success("✅ Video Processing Complete!")

