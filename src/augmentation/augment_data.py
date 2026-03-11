import os
import cv2
import albumentations as A
from tqdm import tqdm

INPUT_DIR = "data/processed/train"
OUTPUT_DIR = "data/augmented/train"

AUGMENTATIONS_PER_IMAGE = 4

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.GaussNoise(p=0.3)
])


def augment_image(image):

    augmented = transform(image=image)
    return augmented["image"]


def augment_dataset():

    classes = os.listdir(INPUT_DIR)

    for cls in classes:

        input_class_path = os.path.join(INPUT_DIR, cls)
        output_class_path = os.path.join(OUTPUT_DIR, cls)

        os.makedirs(output_class_path, exist_ok=True)

        images = os.listdir(input_class_path)

        print(f"Augmenting {cls} images...")

        for img_name in tqdm(images):

            img_path = os.path.join(input_class_path, img_name)

            image = cv2.imread(img_path)

            # Save original
            cv2.imwrite(os.path.join(output_class_path, img_name), image)

            for i in range(AUGMENTATIONS_PER_IMAGE):

                aug_img = augment_image(image)

                new_name = f"{img_name.split('.')[0]}_aug{i}.png"

                save_path = os.path.join(output_class_path, new_name)

                cv2.imwrite(save_path, aug_img)


if __name__ == "__main__":
    augment_dataset()