import tensorflow as tf
import numpy as np
import cv2
import os
import sys

MODEL_PATH = "models/modelo_agrosmart.h5"
IMG_WIDTH = 128
IMG_HEIGHT = 128

def run_inference(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    try:
        modelo = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # PREPROCESSING (Same as app.py)
    imagem_redimensionada = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    imagem_rgb = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB)
    
    # Global Hist Eq (as currently in app.py)
    img_yuv_global = cv2.cvtColor(cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YUV)
    img_yuv_global[:,:,0] = cv2.equalizeHist(img_yuv_global[:,:,0])
    img_global = cv2.cvtColor(cv2.cvtColor(img_yuv_global, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2RGB)
    
    # CLAHE (Adaptive Hist Eq) - Improvement candidate
    img_yuv_clahe = cv2.cvtColor(cv2.cvtColor(imagem_rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv_clahe[:,:,0] = clahe.apply(img_yuv_clahe[:,:,0])
    img_clahe = cv2.cvtColor(cv2.cvtColor(img_yuv_clahe, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2RGB)

    def get_pred(img):
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, axis=0)
        return modelo.predict(arr, verbose=0)[0][0]

    pred_raw = get_pred(imagem_rgb)
    pred_global = get_pred(img_global)
    pred_clahe = get_pred(img_clahe)

    print(f"Results for {image_path}:")
    print(f"  - Raw RGB: {pred_raw:.4f} ({'SAUDÁVEL' if pred_raw >= 0.5 else 'DOENTE'})")
    print(f"  - Global Hist Eq: {pred_global:.4f} ({'SAUDÁVEL' if pred_global >= 0.5 else 'DOENTE'})")
    print(f"  - CLAHE (Proposed): {pred_clahe:.4f} ({'SAUDÁVEL' if pred_clahe >= 0.5 else 'DOENTE'})")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        # Default to the one we will save
        run_inference("data/debug/folha_usuario.jpg")
