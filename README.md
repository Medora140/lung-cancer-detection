# 🫁 Lung Cancer Diagnostic Center

An AI-powered clinical assistant that classifies lung CT scans into four categories: **Adenocarcinoma**, **Large Cell Carcinoma**, **Squamous Cell Carcinoma**, and **Normal**.

The system uses a deep learning model (**EfficientNetB0**) with **Explainable AI (Grad-CAM)** to provide radiologists with a detailed analysis and heatmaps indicating potential areas of concern.

---

## ✨ Key Features

- **Automated Diagnosis:** High-accuracy classification across four lung tissue types.
- **Explainable AI (Grad-CAM):** Generates heatmaps to show exactly where the AI is focusing its attention.
- **Interactive Rive UI:** Engaging clinical interface featuring an animated AI assistant and a medical scanner theme.
- **Clinical Guidance:** Provides specialized treatment plans and next-step recommendations for each diagnosis.
- **Probability Distribution:** Visual breakdown of all potential categories using interactive Plotly charts.

---

## 🛠️ Technology Stack

- **Deep Learning:** TensorFlow / Keras (EfficientNetB0)
- **Image Processing:** OpenCV (CLAHE, Gaussian Blur)
- **Frontend:** Streamlit
- **Animations:** Rive (Runtime Canvas API)
- **Data Visualization:** Plotly
- **Language:** Python 3.10+

---

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lung-cancer-ai.git
   cd lung-cancer-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   streamlit run lung-cancer-ai/app/app.py
   ```

---

## 📂 Project Structure

```text
lung-cancer-ai/
├── app/
│   └── app.py              # Streamlit Web Application
├── models/
│   └── classification_models/ # Trained .h5 model weights
├── src/
│   ├── evaluation/         # Model metrics and confusion matrix
│   ├── inference/          # CLI prediction scripts
│   ├── preprocessing/      # Image enhancement (CLAHE, Blur)
│   └── visualization/      # Grad-CAM heatmap generation
├── data/                   # (Excluded from Git) CT scan dataset
└── requirements.txt        # Dependency list
```

---

## 📊 Model Information

The core model is an **EfficientNetB0** architecture trained on CT scan imagery. It includes a custom preprocessing pipeline that applies:
1. **Grayscale conversion**
2. **Gaussian Blurring** (Noise reduction)
3. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)

These steps ensure the highest possible diagnostic sensitivity for subtle pulmonary nodules.

---

## 🩺 Disclaimer
*This tool is intended for educational and research purposes only. All AI-generated diagnoses should be verified by a board-certified radiologist or medical professional.*
