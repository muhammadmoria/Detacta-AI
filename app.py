import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detecta AI", page_icon="🚗", layout="wide")

st.markdown("""
    <style>
        body { background: linear-gradient(135deg, #1F1C2C, #928DAB); color: #E0E0E0; font-family: 'Poppins', sans-serif; }
        .main-title { font-size: 3.5em; font-weight: bold; color: #FFD700; text-align: center; padding: 30px; }
        .stButton>button { background-color: #FFD700; color: #2C3E50; font-size: 1.4em; padding: 15px 30px; border-radius: 12px; }
        .stButton>button:hover { background-color: #FFC300; color: #282C34; }
        .stFileUploader>div>div>button { background-color: #FFD700 !important; }
        .stSelectbox>div>div { background-color: #2C3E50; color: #FFD700; font-size: 1.3em; }
        .stImage { border-radius: 15px; }
        .footer { font-size: 1.2em; text-align: center; padding: 20px; color: #FFD700; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🔍 Detecta AI 🚗</div>", unsafe_allow_html=True)

MODEL_OPTIONS = {"YOLOv8 Trained V11": "./yolov8_trained_V11.pt", "YOLOv8 Trained": "./yolov8_trained.pt"}
selected_model = st.selectbox("Select YOLO Model:", list(MODEL_OPTIONS.keys()))
MODEL_PATH = MODEL_OPTIONS[selected_model]
model = YOLO(MODEL_PATH)

def detect_objects(image):
    results = model(image)
    return results

def draw_boxes(image, results):
    image = np.array(image)
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box[:4])
            label = f"{model.names[int(cls)]}: {float(conf):.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 127), 4)
            cv2.putText(image, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 127), 4)
    return Image.fromarray(image)

st.sidebar.header("Upload an Image for Object Detection")
uploaded_file = st.sidebar.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"], help="Upload an image for AI-powered object detection.")

col1, col2 = st.columns([1, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption="📸 Uploaded Image", use_column_width=True)
    if col1.button("🔍 Detect Objects"):
        results = detect_objects(image)
        output_image = draw_boxes(image, results)
        col2.image(output_image, caption="🎯 Detected Objects", use_column_width=True)

st.markdown("<div class='footer'>🚀 AI-Powered Object Detection for Smarter Insights!</div>", unsafe_allow_html=True)
