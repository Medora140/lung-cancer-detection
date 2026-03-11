import tensorflow as tf
import numpy as np
import cv2
import sys
import os


# Robust path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classification_models", "efficientnet_lung_model.h5")

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

IMG_SIZE = 224

# Class names
CLASS_NAMES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]


def preprocess_image(image_path):

    # read image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.GaussianBlur(image, (5,5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    # resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # image = image / 255.0  # EfficientNetB0 handles rescaling internally

    # add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def predict(image_path):

    # load model
    model = tf.keras.models.load_model(MODEL_PATH)

    # preprocess image
    
    image = preprocess_image(image_path)

    # prediction
    predictions = model.predict(image)[0]
    for i, prob in enumerate(predictions):
        print(CLASS_NAMES[i], ":", round(prob*100,2), "%")

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    print("\nPrediction Result")
    print("------------------")
    print("Cancer Type:", CLASS_NAMES[predicted_class])
    print("Confidence:", round(float(confidence) * 100, 2), "%")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python detect_cancer.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]

    predict(image_path)
    print(os.path.getmtime("models/classification_models/efficientnet_lung_model.h5"))