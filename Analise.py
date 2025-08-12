import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math

# ==== CONFIGURAÇÕES ====
MODEL_PATH = "best.pt" 
VIDEO_PATH = "entrada.mp4"     # vídeo de entrada
OUTPUT_PATH = "saida_com_velocidade.mp4"
FRAME_RATE = 30                # FPS do vídeo
PIXELS_PER_METER = 50          # 50px = 1 metro

# Carregar modelo YOLO
model = YOLO(MODEL_PATH)

# Abrir vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or FRAME_RATE

# Criar writer para salvar vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Armazenar posições anteriores para calcular velocidade
prev_positions = defaultdict(lambda: None)

def calcular_velocidade(pos1, pos2, dt):
    """Calcula velocidade em km/h a partir de duas posições em pixels"""
    if pos1 is None or pos2 is None:
        return 0.0
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dist_pixels = math.sqrt(dx**2 + dy**2)
    dist_metros = dist_pixels / PIXELS_PER_METER
    m_s = dist_metros / dt
    km_h = m_s * 3.6
    return km_h

def desenhar_elipse(frame, bbox, color):
    """Desenha elipse nos pés do jogador/juiz"""
    y2 = int(bbox[3])
    x_center = int((bbox[0] + bbox[2]) / 2)
    width_box = int(bbox[2] - bbox[0])
    
    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width_box/2), int(0.35*width_box)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2
    )

def desenhar_triangulo(frame, bbox, color):
    """Desenha triângulo acima da bola"""
    x_center = int((bbox[0] + bbox[2]) / 2)
    y_top = int(bbox[1]) - 10
    pontos = np.array([
        [x_center, y_top],
        [x_center - 10, y_top - 20],
        [x_center + 10, y_top - 20]
    ])
    cv2.drawContours(frame, [pontos], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [pontos], 0, (0,0,0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)
    dt = 1 / fps

    if results[0].boxes.id is not None:
        for box, cls_id, track_id in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.id):
            bbox = box.cpu().numpy()
            cls_id = int(cls_id)
            track_id = int(track_id)
            nome = model.names[cls_id]

            # cores por classe
            if nome == "player":
                color = (0, 255, 0)  # verde
            elif nome == "referee":
                color = (0, 255, 255)  # amarelo
            elif nome == "ball":
                color = (255, 0, 0)  # azul
            else:
                continue

            # calcular velocidade apenas para player/referee
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            velocidade_kmh = calcular_velocidade(prev_positions[track_id], center, dt)
            prev_positions[track_id] = center

            if nome == "player" or nome == "referee":
                desenhar_elipse(frame, bbox, color)
                texto = "Jogador" if nome == "player" else "Juiz"
                cv2.putText(frame, f"{texto} {velocidade_kmh:.1f} km/h",
                            (int(bbox[0]), int(bbox[3]) + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
            elif nome == "ball":
                desenhar_triangulo(frame, bbox, color)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Vídeo salvo em: {OUTPUT_PATH}")
