import os
import cv2
from tqdm import tqdm

# Input dataset folder
INPUT_DIR = "data/raw/train"

# Output processed dataset folder
OUTPUT_DIR = "data/processed/train"

# Image size
IMG_SIZE = 224


def preprocess_image(image_path):
    """
    Apply preprocessing to CT scan image
    """

    # read image
    image = cv2.imread(image_path)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # resize image
    resized = cv2.resize(enhanced, (IMG_SIZE, IMG_SIZE))

    return resized


def process_dataset():

    classes = os.listdir(INPUT_DIR)

    for cls in classes:

        input_class_path = os.path.join(INPUT_DIR, cls)
        output_class_path = os.path.join(OUTPUT_DIR, cls)

        os.makedirs(output_class_path, exist_ok=True)

        images = os.listdir(input_class_path)

        print(f"Processing {cls} images...")

        for img_name in tqdm(images):

            img_path = os.path.join(input_class_path, img_name)

            processed = preprocess_image(img_path)

            save_path = os.path.join(output_class_path, img_name)

            cv2.imwrite(save_path, processed)


if __name__ == "__main__":
    process_dataset()
    