from ultralytics import YOLO
import torch
import cv2
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO("yolo11m.pt")
model.to(device)

focused_classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

video_path = "mentahan.mp4"
cap = cv2.VideoCapture(video_path)  # untuk video dari file (misal mp4)
# cap = cv2.VideoCapture(0)         # untuk video langsung dari webcam

# Untuk keperluan menyimpan video output, hanya jika diinginkan
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "output_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Lakukan proses deteksi objek
    results = model(frame)

    # Iterasi tiap hasil deteksi
    for result in results:
        boxes = result.boxes
        names = model.names

        # Hitung jumlah objek per label
        label_counts = {label: 0 for label in focused_classes}

        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = names[cls_id]

            if confidence > 0.3 and label in focused_classes:
                label_counts[label] += 1
                # Gambar bounding box dan label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence*100:.0f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,
                    (0, 255, 0),
                    2,
                )

        # Tampilkan teks ringkasan jumlah objek di pojok kiri atas
        summary = ", ".join([f"{v} {k}" for k, v in label_counts.items() if v > 0])
        box_padding = 12
        (text_width, text_height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (5, 5), (5 + text_width + box_padding, 5 + text_height + box_padding), (0, 255, 0), -1)
        cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    # Simpan frame ke output (jika diperlukan -> Poin 6)
    out.write(frame)

    # Tampilkan frame dengan deteksi objek
    cv2.imshow("YOLOv11 Object Detection", frame)

    # jika diinginkan ukuran layar video dapat diubah
    # resized_frame = cv2.resize(frame, (960, 540))
    # cv2.imshow("YOLOv11 Object Detection", resized_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()