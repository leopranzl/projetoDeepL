import numpy as np
import os
import tensorflow as tf
import sys
import tensorflow_datasets as tfds
import pandas as pd

NUM_CATEGORIES = 10
IMG_WIDTH = 224
IMG_HEIGHT = 224
EPOCHS = 50
BATCH_SIZE = 64
SEED = 123

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

    class_names = train_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

def show_classes_info(dataset, class_names):
    labels = []
    for images, batch_labels in dataset:
        labels.extend(batch_labels.numpy())

    label_counts = pd.Series(labels).value_counts().sort_index()

    label_counts.index = [class_names[i] for i in label_counts.index]

    print("\nImagens por Classe:")
    print(label_counts.sort_index())

def get_model():
    # Use apenas a regularização L2 com um valor pequeno
    # Possivel motivo para  abatch normalization estar atrapalhando o treinamento: dados desbalanceados (classes com muitas e poucas imagens)
    my_regularizer = tf.keras.regularizers.l2(0.001)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1),
        
        #1o bloco
        tf.keras.layers.Conv2D(32, (7, 7), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        #2o bloco
        tf.keras.layers.Conv2D(64, (7, 7), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        #3o bloco
        tf.keras.layers.Conv2D(128, (7, 7), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        #4o bloco
        tf.keras.layers.Conv2D(256, (7, 7), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation = "softmax")
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    return model

def main():
    model = get_model()

    training_set, validation_set, test_set, class_names = load_data(sys.argv[1])
    
    print("--- Training Set ---")
    show_classes_info(training_set, class_names)
    print("\n--- Validation Set ---")
    show_classes_info(validation_set, class_names)
    print("\n--- Test Set ---")
    show_classes_info(test_set, class_names)
    
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.01,
        patience=5,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=False,
        start_from_epoch=0
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,     # Reduz para metade
    patience=2,     # Espera 2 épocas sem melhorar
    min_lr=1e-6,
    verbose=1
)
    
    model.fit(
        training_set,
        validation_data=validation_set,
        epochs=EPOCHS,
        callbacks=[callback, reduce_lr]
    )
    model.evaluate(test_set)
    
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

    
if __name__ == "__main__":
    main()
