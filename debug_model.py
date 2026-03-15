import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_PATH = "models/modelo_agrosmart.h5"
IMG_WIDTH = 128
IMG_HEIGHT = 128

def diagnostic():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    try:
        modelo = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Create a dummy healthy image (green) to test baseline
    dummy_healthy = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    dummy_healthy[:, :] = [0, 200, 0] # Green
    
    # Preprocess
    img_array = dummy_healthy / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = modelo.predict(img_array, verbose=0)[0][0]
    print(f"Prediction for dummy green image: {prediction:.4f}")
    print(f"Classified as: {'SAUDÁVEL' if prediction >= 0.5 else 'DOENTE'}")

if __name__ == "__main__":
    diagnostic()
