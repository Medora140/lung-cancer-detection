import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/classification_models/efficientnet_lung_model.h5"

IMG_SIZE = 224

CLASS_NAMES = [
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Normal",
    "Squamous Cell Carcinoma"
]


def preprocess_image(image_path):

    # read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
        
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


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()


def get_gradcam_image(image_path, model):
    """
    Returns the superimposed Grad-CAM image as a numpy array.
    """
    img_array = preprocess_image(image_path)
    
    # EfficientNet last conv layer
    last_conv_layer_name = "top_conv"

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    image_path = input("Enter CT image path: ")
    model = tf.keras.models.load_model(MODEL_PATH)
    result = get_gradcam_image(image_path, model)
    plt.imshow(result)
    plt.show()