import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Robust path handling
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classification_models", "efficientnet_lung_model.h5")
DATA_PATH = os.path.join(BASE_DIR, "data", "augmented", "train")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224,224),
    batch_size=16
)

class_names = val_dataset.class_names

y_true = []
y_pred = []

for images, labels in val_dataset:
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(preds)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()