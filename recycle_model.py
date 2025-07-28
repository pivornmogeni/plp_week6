import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load dataset from tfds or use your own custom data
(train_ds, val_ds), ds_info = tfds.load(
    'rock_paper_scissors',  # Use as a lightweight example (simulate recyclables)
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

IMG_SIZE = (64, 64)

def format_image(image, label):
    image = tf.image.resize(image, IMG_SIZE) / 255.0
    return image, label

train_ds = train_ds.map(format_image).batch(32).prefetch(1)
val_ds = val_ds.map(format_image).batch(32).prefetch(1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes in simulated data
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=5)

val_loss, val_accuracy = model.evaluate(val_ds)
print(f"Validation Accuracy: {val_accuracy:.2f}")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save to .tflite file
with open('recycle_model.tflite', 'wb') as f:
    f.write(tflite_model)

import numpy as np

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="recycle_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Pick one image from validation dataset
for img, label in val_ds.take(1):
    test_img = img[0].numpy().reshape(1, 64, 64, 3).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], test_img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = tf.argmax(output, axis=1).numpy()
    print(f"Predicted class: {pred}, Actual class: {label[0].numpy()}")
