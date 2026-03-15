import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

MODEL_PATH = "models/modelo_agrosmart.h5"
CSV_PATH = "logs/dados_agrosmart.csv"

IMG_WIDTH = 128
IMG_HEIGHT = 128

try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
except Exception:
    exit()

class AgroSmartApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AgroSmart FIAP - Analisador de Folhas")
        self.root.geometry("800x650")
        self.root.configure(bg="#f4f4f4")

        self.camera = None
        self.is_camera_on = False
        self.categoria_atual = ""
        self.confianca_atual = 0.0

        self.lbl_title = tk.Label(root, text="AGROSMART - ANALISADOR DE FOLHAS", font=("Arial", 18, "bold"), bg="#f4f4f4")
        self.lbl_title.pack(pady=10)

        self.canvas = tk.Canvas(root, width=640, height=480, bg="black")
        self.canvas.pack(pady=10)

        self.lbl_resultado = tk.Label(root, text="IA: Aguardando imagem...", font=("Arial", 16, "bold"), bg="#f4f4f4", fg="gray")
        self.lbl_resultado.pack(pady=5)

        self.btn_frame = tk.Frame(root, bg="#f4f4f4")
        self.btn_frame.pack(pady=10)

        self.btn_camera = tk.Button(self.btn_frame, text="Ligar Câmera", width=20, font=("Arial", 11), command=self.toggle_camera)
        self.btn_camera.grid(row=0, column=0, padx=10)

        self.btn_upload = tk.Button(self.btn_frame, text="Enviar Imagem", width=20, font=("Arial", 11), command=self.upload_image)
        self.btn_upload.grid(row=0, column=1, padx=10)

        self.btn_salvar = tk.Button(self.btn_frame, text="Salvar CSV", width=20, font=("Arial", 11), state=tk.DISABLED, command=self.salvar_csv)
        self.btn_salvar.grid(row=0, column=2, padx=10)

        self.delay = 15
        self.update_webcam()

    def analisar_planta(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Máscara adaptada para capturar verdes e tons de marrom/amarelado (doenças)
        lower_verde = np.array([20, 30, 30])
        upper_verde = np.array([100, 255, 255])
        
        mask = cv2.inRange(hsv, lower_verde, upper_verde)
        
        # Operações morfológicas para consolidar a folha como um único objeto
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fecha buracos internos (manchas)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove ruído externo
        
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contornos:
            return None, None
            
        # Pega o maior contorno (provável folha)
        maior_contorno = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(maior_contorno)
        
        if area > 5000: # Threshold de área levemente reduzido
            x, y, w, h = cv2.boundingRect(maior_contorno)
            recorte = frame[y:y+h, x:x+w]
            return (x, y, w, h), recorte
        return None, None

    def classificar_imagem(self, frame):
        # 1. Redimensionar para IA
        imagem_redimensionada = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        imagem_rgb = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB)
        
        # 2. IA - Predição
        imagem_array = np.array(imagem_rgb) / 255.0
        imagem_array = np.expand_dims(imagem_array, axis=0)
        previsao_ia = modelo.predict(imagem_array, verbose=0)[0][0]
        
        # 3. HEURÍSTICA - Detecção de Manchas (Marrom/Amarelo/Preto)
        hsv_recorte = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Ranges para necrose e manchas de doenças
        lower_spot = np.array([0, 30, 20])   # Marrom/Castanho
        upper_spot = np.array([35, 255, 180]) # Amarelo escuro até preto
        mask_spot = cv2.inRange(hsv_recorte, lower_spot, upper_spot)
        
        # Calcular proporção de manchas na folha
        leaf_pixels = frame.shape[0] * frame.shape[1]
        spot_pixels = np.count_nonzero(mask_spot)
        heuristic_score = (spot_pixels / leaf_pixels) * 5.0 # Peso amplificado
        
        # 4. LÓGICA HÍBRIDA
        # Se a heurística detectar muitas manchas, aumenta drasticamente a chance de "Doente"
        # IA 0.5 + Heuristica > Threshold
        score_final = (previsao_ia * 0.4) + (min(heuristic_score, 1.0) * 0.6)
        
        # Threshold de decisão híbrido
        if score_final < 0.35:
            categoria = "SAUDÁVEL"
            confianca = float((1.0 - score_final) * 100)
            cor = "green"
            bgr_cor = (0, 255, 0)
        else:
            categoria = "DOENTE"
            confianca = float(score_final * 100)
            cor = "red"
            bgr_cor = (0, 0, 255)
            
        return categoria, confianca, cor, bgr_cor, previsao_ia, heuristic_score

    def desenhar_frame(self, frame):
        frame_display = frame.copy()
        bbox, recorte = self.analisar_planta(frame_display)

        if recorte is not None:
            self.categoria_atual, self.confianca_atual, cor_tk, bgr_cor, raw_ia, raw_h = self.classificar_imagem(recorte)
            
            x, y, w, h = bbox
            cv2.rectangle(frame_display, (x, y), (x+w, y+h), bgr_cor, 3)
            cv2.rectangle(frame_display, (x, y-30), (x+w, y), bgr_cor, -1)
            
            # HUD Híbrido: IA e Heurística (Spot Ratio)
            texto = f"{self.categoria_atual} ({self.confianca_atual:.1f}%) | AI:{raw_ia:.2f} H:{raw_h:.2f}"
            cv2.putText(frame_display, texto, 
                        (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

            self.lbl_resultado.config(text=f"SISTEMA HÍBRIDO: {texto}", fg=cor_tk)
            self.btn_salvar.config(state=tk.NORMAL)
        else:
            self.categoria_atual = ""
            self.confianca_atual = 0.0
            self.lbl_resultado.config(text="IA: Nenhuma planta detectada", fg="orange")
            self.btn_salvar.config(state=tk.DISABLED)

        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        h_img, w_img = frame_rgb.shape[:2]
        razao = min(640/w_img, 480/h_img)
        novo_w, novo_h = int(w_img * razao), int(h_img * razao)
        
        frame_resized = cv2.resize(frame_rgb, (novo_w, novo_h))
        imagem_pil = Image.fromarray(frame_resized)
        
        frame_fundo = Image.new("RGB", (640, 480), (0, 0, 0))
        offset = ((640 - novo_w) // 2, (480 - novo_h) // 2)
        frame_fundo.paste(imagem_pil, offset)

        self.photo = ImageTk.PhotoImage(image=frame_fundo)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            self.btn_camera.config(text="Ligar Câmera")
            if self.camera:
                self.camera.release()
            self.canvas.delete("all")
            self.lbl_resultado.config(text="IA: Câmera Desligada", fg="gray")
            self.btn_salvar.config(state=tk.DISABLED)
        else:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.is_camera_on = True
                self.btn_camera.config(text="Desligar Câmera")
            else:
                messagebox.showerror("Erro", "Não foi possível acessar a Webcam.")

    def update_webcam(self):
        if self.is_camera_on and self.camera.isOpened():
            sucesso, frame = self.camera.read()
            if sucesso:
                self.desenhar_frame(frame)
        self.root.after(self.delay, self.update_webcam)

    def upload_image(self):
        if self.is_camera_on:
            self.toggle_camera()
            
        caminho_imagem = filedialog.askopenfilename(
            title="Selecione uma Imagem",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if caminho_imagem:
            frame = cv2.imread(caminho_imagem)
            if frame is not None:
                self.desenhar_frame(frame)
            else:
                messagebox.showerror("Erro", "Não foi possível carregar a imagem selecionada.")

    def salvar_csv(self):
        if not self.categoria_atual:
            return
            
        data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        nome_captura = f"captura_{datetime.now().strftime('%H%M%S')}"
        
        novo_dado = pd.DataFrame([{
            "Nome da Imagem": nome_captura,
            "Data/Hora": data_atual,
            "Categoria Detectada": self.categoria_atual,
            "Acurácia (Confiança)": f"{self.confianca_atual:.2f}%"
        }])
        
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        
        if not os.path.exists(CSV_PATH):
            novo_dado.to_csv(CSV_PATH, index=False, sep=";")
        else:
            novo_dado.to_csv(CSV_PATH, mode='a', header=False, index=False, sep=";")
            
        messagebox.showinfo("Sucesso", f"Dados Salvos!\nImagem: {nome_captura}\nClasse: {self.categoria_atual}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AgroSmartApp(root)
    root.mainloop()
