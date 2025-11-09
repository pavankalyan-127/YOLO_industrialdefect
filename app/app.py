import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import numpy as np
import os

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="Industrial Defect Detection", layout="wide")
st.title("🏭 Industrial Defect Detection using YOLOv8")
st.write("Upload an image or video to detect cracks, rusts, or other surface defects.")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    # Use relative path for Streamlit Cloud (model file in repo root or models/)
    model_path = "models/best.pt"
    model = YOLO(model_path)
    return model

model = load_model()

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
                res_plotted = results[0].plot()  # returns a numpy array (BGR)

                # Convert BGR to RGB for Streamlit display
                res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                st.image(res_rgb, caption="Detected Defects", use_column_width=True)

                # Display detected boxes
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    st.subheader("📊 Detection Details:")
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.write(f"**Class:** {model.names[cls_id]} | **Confidence:** {conf*100:.2f}%")
                else:
                    st.warning("No defects detected.")

                # Save output image if selected
                if save_output:
                    output_path = "output_image.jpg"
                    cv2.imwrite(output_path, res_plotted)
                    st.success(f"💾 Image saved as `{output_path}`")

                st.success("✅ Detection Complete!")

# -------------------- VIDEO DETECTION --------------------
elif input_type == "Video":
    uploaded_video = st.file_uploader("📹 Upload a video", type=["mp4", "avi", "mov", "mkv"])

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

                    # Stream the frame in Streamlit
                    stframe.image(annotated_frame, channels="BGR", use_column_width=True)

                    if save_output:
                        out.write(annotated_frame)

                cap.release()
                if save_output:
                    out.release()
                    st.success("🎬 Output video saved as `output_video.avi`")

            st.success("✅ Video Processing Complete!")
