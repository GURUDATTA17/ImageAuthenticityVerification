import numpy as np
import cv2
from keras.models import load_model
import joblib


def predict_bihpf(image_path):
    try:
        # Determine model format and load accordingly
        model_path = 'models/bihpf_model.keras'
        if model_path.endswith('.joblib'):
            print(f"Loading Joblib model from {model_path}")
            model = joblib.load(model_path)
        elif model_path.endswith('.keras') or model_path.endswith('.pkl'):
            print(f"Loading Keras model from {model_path}")
            model = load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        # Preprocess the image
        print("BiHPF: Preprocessing image")
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image not found or cannot be loaded.")
        
        img = cv2.resize(img, (64, 64))  # Resize to match model input
        img = np.float32(img) / 255.0  # Normalize to [0, 1]
        
        # Apply DCT (Discrete Cosine Transform)
        img_dct = cv2.dct(img)
        
        # Flatten the image to match the expected input shape of the model
        img_dct_flattened = img_dct.flatten()  # Flatten to a 1D array (64x64 = 4096 features)

        # If the model expects 1152 features, you may need to adjust this based on training preprocessing.
        # Example: if model was trained with specific features (e.g., 1152), adjust preprocessing accordingly.
        img_dct_flattened = img_dct_flattened[:1152]  # Truncate or adjust to match expected feature size (1152)

        # Reshape to match the input shape (1, 1152)
        img_dct_flattened = img_dct_flattened.reshape(1, 1152)

        # Make prediction
        print("BiHPF: Running prediction")
        predictions = model.predict(img_dct_flattened, verbose=0)
        confidence = float(np.max(predictions) * 100)
        label = "Real" if np.argmax(predictions) == 0 else "Fake"

        print(f"BiHPF Prediction: {label}, Confidence: {confidence}")
        return label, confidence

    except Exception as e:
        print(f"Error in BiHPF prediction: {str(e)}")
        return "Error", 0.0


# Example usage:
# label, confidence = predict_bihpf('path_to_image.jpg')
# print(label, confidence)
