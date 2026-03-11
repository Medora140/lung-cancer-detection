import tensorflow as tf

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/augmented/train",
    image_size=(224,224),
    batch_size=32
)

print(dataset.class_names)
