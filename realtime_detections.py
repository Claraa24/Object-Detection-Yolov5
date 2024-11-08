import torch
import cv2
from yolov5 import YOLOv5
import torch

# Load model YOLOv5 dengan model hasil pelatihan
model_path = 'yolov5/best.pt'  # ganti dengan path ke file best.pt Anda
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, source='local')

# Membuka kamera (gunakan 0 untuk kamera default)
cap = cv2.VideoCapture(0)

while True:
    # Membaca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek di frame
    results = model(frame)

    # Menggambar hasil deteksi pada frame
    for pred in results.xyxy[0]:  # prediksi bounding box
        x1, y1, x2, y2, conf, cls = map(int, pred[:6])  # koordinat bounding box
        label = model.names[int(cls)]  # nama kelas
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame dengan hasil deteksi
    cv2.imshow("Real-Time Detection", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
