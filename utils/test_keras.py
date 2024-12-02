# # utils/test_keras.py
# import tensorflow as tf
# import numpy as np

# def predict_keras(img_path):
#     try:
#         # Load the pre-trained model
#         model_path = 'models/my_model.keras'
#         print(f"Keras Model: Loading model from {model_path}")
#         model = tf.keras.models.load_model(model_path)

#         # Preprocess image
#         print("Keras Model: Preprocessing image")
#         image_size = (128, 128)  # Adjust to match the model's input size
#         img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
#         img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Make prediction
#         print("Keras Model: Running prediction")
#         predictions = model.predict(img_array, verbose=0)
        
#         # Assuming binary classification where predictions[0][0] represents "Real"
#         confidence = float(predictions[0][0])
#         if confidence > 0.5:
#             label = "Real"
#             confidence *= 100
#         else:
#             label = "Fake"
#             confidence = (1 - confidence) * 100

#         print(f"Keras Model Prediction: {label}, Confidence: {confidence}")
#         return label, confidence
#     except Exception as e:
#         print(f"Error in Keras model prediction: {str(e)}")
#         return "Error", 0.0


# utils/test_keras.py
import tensorflow as tf
import numpy as np
import cv2

def preprocess_image(img_path, input_shape):
    print("Preprocessing image")
    # Load and preprocess image based on model input shape
    if input_shape[-1] == 1:  # Grayscale
        img = tf.keras.preprocessing.image.load_img(img_path, color_mode='grayscale')
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = tf.image.resize(img_array, input_shape[:2])  # Resize to match input shape
    else:  # RGB or other formats
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=input_shape[:2])
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_keras(img_path):
    try:
        # Load the pre-trained model
        model_path = 'models/my_model.keras'
        print(f"Keras Model: Loading model from {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Inspect model input shape
        print(f"Model Input Shape: {model.input_shape}")

        # Preprocess image
        img_array = preprocess_image(img_path, model.input_shape[1:])

        # Make prediction
        print("Keras Model: Running prediction")
        predictions = model.predict(img_array, verbose=0)
        
        # Assuming binary classification where predictions[0][0] represents "Real"
        confidence = float(predictions[0][0])
        if confidence > 0.5:
            label = "Real"
            confidence *= 100
        else:
            label = "Fake"
            confidence = (1 - confidence) * 100

        print(f"Keras Model Prediction: {label}, Confidence: {confidence}")
        return label, confidence
    except Exception as e:
        print(f"Error in Keras model prediction: {str(e)}")
        return "Error", 0.0
