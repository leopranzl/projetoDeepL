import numpy as np
import os
import tensorflow as tf
import sys
import tensorflow_datasets as tfds

NUM_CATEGORIES = 12
IMG_WIDTH = 224
IMG_HEIGHT = 224
EPOCHS = 10
BATCH_SIZE = 32
 
def load_data(data_dir):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='int',
        color_mode='rgb',
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset



def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandAugment(value_range=(0, 1), num_ops=2, factor=0.5, interpolation="bilinear", seed=123, data_format=None,),
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(12, activation = "softmax")
    ])
    
    model.compile(
        optimizer="adam",
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    return model

def main():
    
    model = get_model()
    training_set, validation_set, test_set = load_data(sys.argv[1])
    model.fit(
        training_set,
        validation_data=validation_set,
        epochs=EPOCHS
    )
    model.evaluate(test_set)
    
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    
if __name__ == "__main__":
    main()
