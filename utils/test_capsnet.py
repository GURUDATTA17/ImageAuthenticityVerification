# utils/test_capsnet.py
import tensorflow as tf
import numpy as np


class SquashLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        norm = tf.reduce_sum(tf.square(inputs), -1, keepdims=True)
        scale = norm / (1 + norm) / tf.sqrt(norm + tf.keras.backend.epsilon())
        return scale * inputs


def predict_capsnet(img_path):
    try:
        # Load the pre-trained model
        model_path = 'models/capsnet_model.keras'
        print(f"CapsNet: Loading model from {model_path}")
        model = tf.keras.models.load_model(
            model_path, custom_objects={'SquashLayer': SquashLayer}
        )

        # Preprocess image
        print("CapsNet: Preprocessing image")
        image_size = (128, 128)
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        print("CapsNet: Running prediction")
        predictions = model.predict(img_array, verbose=0)
        confidence = float(predictions[0][0])

        if confidence > 0.5:
            label = "Real"
            confidence *= 100
        else:
            label = "Fake"
            confidence = (1 - confidence) * 100

        print(f"CapsNet Prediction: {label}, Confidence: {confidence}")
        return label, confidence
    except Exception as e:
        print(f"Error in CapsNet prediction: {str(e)}")
        return "Error", 0.0
