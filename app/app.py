import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components

# Add parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.visualization.gradcam import get_gradcam_image
except ImportError:
    st.error("Could not import Grad-CAM module. Please ensure the project structure is correct.")

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Diagnostic Center",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #e9ecef; }
    </style>
    """, unsafe_allow_html=True)

# Robust path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classification_models", "efficientnet_lung_model.h5")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES = ["Adenocarcinoma", "Large Cell Carcinoma", "Normal", "Squamous Cell Carcinoma"]

# Treatment logic
TREATMENTS = {
    "Adenocarcinoma": {
        "description": "Adenocarcinoma is the most common form of lung cancer, often occurring in the outer parts of the lungs.",
        "steps": ["Genetic testing for EGFR/ALK mutations", "Surgical resection (early stage)", "Immunotherapy or Target Therapy"]
    },
    "Large Cell Carcinoma": {
        "description": "Large cell carcinoma grows and spreads quickly, making it more difficult to treat than other types.",
        "steps": ["Combination chemotherapy", "Radiation therapy", "Clinical trials for targeted drugs"]
    },
    "Squamous Cell Carcinoma": {
        "description": "Often linked to smoking, this type usually forms in the center of the lungs near the main airways.",
        "steps": ["Surgery (if localized)", "Adjuvant Chemotherapy", "PD-L1 expression testing for Immunotherapy"]
    },
    "Normal": {
        "description": "No evidence of cancerous tissue detected in this CT scan.",
        "steps": ["Regular health screenings", "Avoidance of smoking/pollutants", "Maintain healthy diet and exercise"]
    }
}

# Sidebar Content
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2864/2864333.png", width=100)
    st.title("AI Clinical Assistant")
    st.divider()
    components.html("""
        <canvas id="robot" width="200" height="200"></canvas>
        <script src="https://unpkg.com/@rive-app/canvas@latest"></script>
        <script>
            new rive.Rive({ 
                src: "https://public.rive.app/community/runtime-files/2126-4217-little-robot.riv", 
                canvas: document.getElementById("robot"), 
                autoplay: true, 
                layout: new rive.Layout({ fit: 'contain', alignment: 'center' }),
                onLoad: () => { r.resizeDrawingSurfaceToCanvas(); } 
            });
        </script>
    """, height=220)
    st.info("Upload a scan in the main panel to begin analysis.")

st.title("🫁 Lung Cancer Diagnostic Center")

if model is None:
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Preprocessing
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Save temp file for grad-cam
    temp_path = "temp_scan.png"
    cv2.imwrite(temp_path, image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    resized = cv2.resize(enhanced, (224, 224))
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    input_tensor = np.expand_dims(rgb_image, axis=0)

    # Prediction
    preds = model.predict(input_tensor)[0]
    idx = np.argmax(preds)
    label = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Diagnosis", "🧪 AI Insights (Grad-CAM)", "🩺 Treatment Plan"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(uploaded_file, caption="Input Scan", use_container_width=True)
        with c2:
            st.subheader("Results")
            if label == "Normal":
                st.success(f"**Predicted Category:** {label}")
            else:
                st.error(f"**Predicted Category:** {label}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")
            
            # Chart
            df = pd.DataFrame({"Type": CLASS_NAMES, "Score": [round(float(p)*100, 2) for p in preds]})
            fig = px.bar(df, x="Score", y="Type", orientation='h', color="Score", color_continuous_scale="RdBu_r")
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Model Attention Visualization")
        st.write("The heatmap below indicates the areas the AI analyzed to reach its conclusion.")
        with st.spinner("Generating heatmap..."):
            try:
                heatmap_img = get_gradcam_image(temp_path, model)
                st.image(heatmap_img, caption="AI Heatmap (Warm colors indicate high importance)", use_container_width=True)
            except Exception as e:
                st.error(f"Error generating heatmap: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    with tab3:
        st.subheader(f"Clinical Guidance for {label}")
        data = TREATMENTS[label]
        st.info(data["description"])
        st.write("### Recommended Next Steps:")
        for step in data["steps"]:
            st.markdown(f"- {step}")
        st.divider()
        st.button("📄 Export Clinical Report (PDF)", disabled=True)
        st.caption("Note: This is an AI-assisted tool. All diagnoses must be confirmed by a certified radiologist.")

else:
    # Landing page Rive animation
    components.html("""
    <div style="display: flex; justify-content: center;"><canvas id="scan" width="400" height="300"></canvas></div>
    <script src="https://unpkg.com/@rive-app/canvas@latest"></script>
    <script>
        new rive.Rive({ 
            src: "https://public.rive.app/community/runtime-files/2191-4372-scanning-process.riv", 
            canvas: document.getElementById("scan"), 
            autoplay: true, 
            layout: new rive.Layout({ fit: 'contain', alignment: 'center' }),
            onLoad: () => { r.resizeDrawingSurfaceToCanvas(); } 
        });
    </script>""", height=320)
    st.markdown("<h3 style='text-align: center;'>Please upload a CT Scan image to begin automated analysis.</h3>", unsafe_allow_html=True)
