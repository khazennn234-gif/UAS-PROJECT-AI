"""
SMART FRUIT SCANNER (EDISI YOLO26 - RTX 3050 GPU EDITION)
Sistem Deteksi Kematangan, Kesegaran, dan Informasi Nutrisi Buah
================================================================
Menggunakan Arsitektur Terbaru: YOLO26 (NMS-Free Real-Time Vision)
Dioptimalkan untuk GPU NVIDIA RTX 3050 4GB (CUDA Acceleration)
Tidak menggunakan best_fruit.pt lama (murni menggunakan model YOLO26)

Mendukung:
- Deteksi langsung dengan pretrained YOLO26 (yolo26m.pt / yolo26s.pt)
- Deteksi dengan model YOLO26 terlatih (best_yolo26_fruit.pt)
- Fitur Training Mandiri YOLO26 pada dataset langsung dari menu!

Dikembangkan dari Modul Praktikum AI:
1. YOLO.py             : Deteksi objek dan bounding box menggunakan YOLO26
2. ReadImageData.py    : Membaca dan menampilkan citra menggunakan OpenCV
3. ReadDisplayVideo.py : Membaca video dan webcam frame-by-frame
4. FUZZY.py            : Inferensi Fuzzy Logic (Antecedent, Consequent, Rule)
================================================================
"""

import os
import glob
import shutil
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------------------------------------------------------------
# 1. KONFIGURASI HARDWARE GPU DAN LOAD MODEL YOLO26
# (Dioptimalkan khusus untuk GPU RTX 3050 4GB VRAM)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 64)
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[INFO] GPU Terdeteksi : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"[INFO] Mode Akselerasi : CUDA AKTIF (Optimal untuk RTX 3050)")
else:
    print("[INFO] Perangkat: CPU (CUDA tidak terdeteksi)")
print("=" * 64)

# Kandidat model khusus YOLO26 (TIDAK MENGGUNAKAN best_fruit.pt)
# Prioritas:
# 1. best_yolo26_fruit.pt (jika teman sudah melatih YOLO26 pada dataset)
# 2. yolo26m.pt (YOLO26 Medium - sangat akurat untuk RTX 3050 4GB)
# 3. yolo26s.pt / yolo26n.pt (YOLO26 Small / Nano)
model_candidates = [
    "best_yolo26_fruit.pt",
    "runs/fruit_freshness/yolo26/weights/best.pt",
    "yolo26m.pt",
    "yolo26s.pt",
    "yolo26n.pt"
]

model_path = "yolo26m.pt"
for candidate in model_candidates:
    if os.path.exists(candidate):
        model_path = candidate
        break

print(f"[INFO] Memuat arsitektur model YOLO26: {model_path} pada {device.upper()}...")
model = YOLO(model_path)
model.to(device)

# Batas kepercayaan (Confidence Threshold) optimal untuk YOLO26
CONFIDENCE_THRESHOLD = 0.35

# Daftar kelas buah dan sayur yang didukung (Dataset Roboflow + COCO)
buah_classes = [
    # 17 Kelas dari Dataset Roboflow (setelah ditraining)
    'fresh apple', 'fresh banana', 'fresh bellpepper', 'fresh carrot',
    'fresh cucumber', 'fresh mango', 'fresh orange', 'fresh potato',
    'rotten apple', 'rotten banana', 'rotten carrot', 'rotten cucumber',
    'rotten mango', 'rotten orange', 'rotten potato', 'rotten tomato',
    'rottenbellpepper',
    # Kelas buah dari pretrained model YOLO26
    'apple', 'banana', 'orange', 'carrot', 'broccoli'
]

focused_classes = buah_classes

# ----------------------------------------------------------------------
# 2. BASIS PENGETAHUAN NUTRISI (9 Komoditas dari Dataset)
# ----------------------------------------------------------------------
database_nutrisi = {
    "apple": {
        "nama": "Apel",
        "kalori": "52 kkal / 100g",
        "vitamin": "Vitamin C (4.6 mg), Serat Pektin",
        "khasiat": "Menjaga kesehatan jantung dan melancarkan pencernaan.",
        "kematangan_info": "Saat matang kandungan pektin dan rasa manis seimbang. Jika lewat matang tekstur bertepung.",
        "tips": "Pilih yang padat saat ditekan dan kulitnya cerah tidak berkerut."
    },
    "banana": {
        "nama": "Pisang",
        "kalori": "89 kkal / 100g",
        "vitamin": "Kalium (358 mg), Vitamin B6, Vitamin C",
        "khasiat": "Sumber energi cepat dan membantu fungsi kerja otot.",
        "kematangan_info": "Pisang mentah tinggi pati resisten, saat matang pati berubah menjadi gula alami yang mudah dicerna.",
        "tips": "Pilih warna kuning merata. Bintik cokelat kecil menandakan kematangan optimal."
    },
    "bellpepper": {
        "nama": "Paprika",
        "kalori": "31 kkal / 100g",
        "vitamin": "Vitamin C (127 mg), Vitamin A",
        "khasiat": "Meningkatkan daya tahan tubuh dan menjaga kesehatan kulit.",
        "kematangan_info": "Paprika hijau adalah fase mentah, saat matang berubah merah/kuning dengan rasa lebih manis.",
        "tips": "Pilih yang kulitnya mengkilap kencang dan tangkainya segar."
    },
    "carrot": {
        "nama": "Wortel",
        "kalori": "41 kkal / 100g",
        "vitamin": "Beta-karoten (Vitamin A), Vitamin K1",
        "khasiat": "Sangat baik untuk kesehatan mata dan imunitas tubuh.",
        "kematangan_info": "Wortel muda lebih manis dan renyah. Jika terlalu tua bagian tengah akan mengeras.",
        "tips": "Pilih yang berbatang lurus, berwarna oranye terang, dan tidak berongga."
    },
    "cucumber": {
        "nama": "Mentimun",
        "kalori": "15 kkal / 100g",
        "vitamin": "Kadar Air 95%, Vitamin K, Kalium",
        "khasiat": "Menjaga hidrasi tubuh dan membantu detoksifikasi alami.",
        "kematangan_info": "Dipanen saat masih muda agar biji belum keras dan tekstur renyah berair.",
        "tips": "Pilih yang ujungnya masih padat kencang dan warnanya hijau segar merata."
    },
    "mango": {
        "nama": "Mangga",
        "kalori": "60 kkal / 100g",
        "vitamin": "Vitamin C (36.4 mg), Vitamin A, Folat",
        "khasiat": "Meningkatkan sistem imun dan menjaga kesehatan pencernaan.",
        "kematangan_info": "Mangga matang memiliki aroma harum di bagian pangkal dan sedikit lunak saat ditekan lembut.",
        "tips": "Hindari mangga dengan kulit yang berkeriput atau memiliki memar lembek."
    },
    "orange": {
        "nama": "Jeruk",
        "kalori": "47 kkal / 100g",
        "vitamin": "Vitamin C (53.2 mg), Asam Sitrat",
        "khasiat": "Antioksidan kuat untuk imunitas dan penyerapan zat besi.",
        "kematangan_info": "Jeruk tidak mengalami pematangan lanjutan setelah dipetik, pilih yang sudah matang sempurna di pohon.",
        "tips": "Pilih jeruk yang terasa berat untuk ukurannya (tanda kandungan sari buah berlimpah)."
    },
    "potato": {
        "nama": "Kentang",
        "kalori": "77 kkal / 100g",
        "vitamin": "Karbohidrat Kompleks, Vitamin B6, Kalium",
        "khasiat": "Sumber karbohidrat pengenyang dan penstabil tekanan darah.",
        "kematangan_info": "Kentang siap konsumsi saat kulit luar sudah tidak mudah terkelupas dengan usapan jari.",
        "tips": "Hindari kentang yang memiliki tunas bertunas atau bercak kehijauan (solanin)."
    },
    "tomato": {
        "nama": "Tomat",
        "kalori": "18 kkal / 100g",
        "vitamin": "Likopen (Antioksidan), Vitamin C, Kalium",
        "khasiat": "Melindungi sel tubuh dari radikal bebas dan kesehatan prostat.",
        "kematangan_info": "Warna merah merata menandakan akumulasi likopen telah mencapai tingkat tertinggi.",
        "tips": "Pilih yang kulitnya mulus mengkilap dan memiliki bobot yang padat berisi."
    }
}

# ----------------------------------------------------------------------
# 3. SISTEM INFERENSI FUZZY LOGIC (Sesuai FUZZY.py praktikum)
# ----------------------------------------------------------------------
warna_in = ctrl.Antecedent(np.arange(0, 101, 1), 'warna')
kondisi_in = ctrl.Antecedent(np.arange(0, 101, 1), 'kondisi')
kesegaran_out = ctrl.Consequent(np.arange(0, 101, 1), 'kesegaran')

warna_in['hijau_segar'] = fuzz.trimf(warna_in.universe, [0, 0, 45])
warna_in['matang_optimal'] = fuzz.trimf(warna_in.universe, [30, 50, 70])
warna_in['kecokelatan'] = fuzz.trimf(warna_in.universe, [55, 100, 100])

kondisi_in['buruk'] = fuzz.trimf(kondisi_in.universe, [0, 0, 40])
kondisi_in['cukup'] = fuzz.trimf(kondisi_in.universe, [30, 50, 70])
kondisi_in['sempurna'] = fuzz.trimf(kondisi_in.universe, [60, 100, 100])

kesegaran_out['busuk'] = fuzz.trimf(kesegaran_out.universe, [0, 0, 40])
kesegaran_out['sedang'] = fuzz.trimf(kesegaran_out.universe, [30, 50, 70])
kesegaran_out['segar'] = fuzz.trimf(kesegaran_out.universe, [60, 100, 100])

rule1 = ctrl.Rule(kondisi_in['sempurna'] & (warna_in['hijau_segar'] | warna_in['matang_optimal']), kesegaran_out['segar'])
rule2 = ctrl.Rule(kondisi_in['cukup'] & warna_in['matang_optimal'], kesegaran_out['sedang'])
rule3 = ctrl.Rule(kondisi_in['buruk'] | warna_in['kecokelatan'], kesegaran_out['busuk'])
rule4 = ctrl.Rule(kondisi_in['cukup'] & warna_in['hijau_segar'], kesegaran_out['segar'])
rule5 = ctrl.Rule(kondisi_in['sempurna'] & warna_in['kecokelatan'], kesegaran_out['sedang'])

fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])

def hitung_kesegaran_fuzzy(skor_warna, skor_kondisi):
    """Menghitung skor kesegaran (0-100) menggunakan inferensi Fuzzy Logic"""
    sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    sim.input['warna'] = np.clip(skor_warna, 0, 100)
    sim.input['kondisi'] = np.clip(skor_kondisi, 0, 100)

    try:
        sim.compute()
        hasil = sim.output['kesegaran']
    except Exception:
        hasil = (skor_kondisi * 0.7) + ((100.0 - skor_warna) * 0.3)

    hasil = float(np.clip(hasil, 0, 100))

    if skor_warna < 40:
        kematangan = "Mentah / Mengkal"
    elif skor_warna < 70:
        kematangan = "Matang"
    else:
        kematangan = "Terlalu Matang"

    if hasil >= 65:
        status_kesegaran = "Segar Secara Visual"
        tag = "[SEGAR]"
        warna_box = (0, 255, 0)
    elif hasil >= 40:
        status_kesegaran = "Mulai Menurun"
        tag = "[SEDANG]"
        warna_box = (0, 255, 255)
    else:
        status_kesegaran = "Terindikasi Busuk"
        tag = "[BUSUK]"
        warna_box = (0, 0, 255)

    return hasil, kematangan, status_kesegaran, tag, warna_box

# ----------------------------------------------------------------------
# 4. EKSTRAKSI FITUR CITRA MENGGUNAKAN OPENCV (HSV)
# ----------------------------------------------------------------------
def ekstrak_fitur_citra(roi, label_yolo):
    """Mengekstrak rata-rata warna dan kondisi visual buah menggunakan fungsi OpenCV dasar"""
    if roi is None or roi.size == 0:
        return 50.0, 50.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    label_lower = label_yolo.lower()

    if "rotten" in label_lower:
        skor_kondisi = 20.0
        skor_warna = 85.0
        return skor_warna, skor_kondisi
    elif "fresh" in label_lower:
        skor_kondisi = 85.0
        skor_warna = 25.0 if (35 <= h_mean <= 85) else 55.0
        return skor_warna, skor_kondisi

    # Label COCO dasar dari model pretrained YOLO26
    if s_mean < 70 or v_mean < 70:
        skor_kondisi = 25.0
        skor_warna = 85.0
    elif s_mean < 95:
        skor_kondisi = 55.0
        skor_warna = 60.0
    else:
        skor_kondisi = 85.0
        skor_warna = 25.0 if (35 <= h_mean <= 85) else 55.0

    return skor_warna, skor_kondisi

def dapatkan_nama_buah(label):
    """Mengambil nama buah murni dari label deteksi YOLO"""
    nama = label.lower().replace("fresh", "").replace("rotten", "").strip()
    return nama

# ----------------------------------------------------------------------
# 5. PEMROSESAN FRAME DAN DETEKSI YOLO26
# ----------------------------------------------------------------------
def proses_deteksi(frame):
    """Menjalankan deteksi objek YOLO26 dan evaluasi kesegaran Fuzzy Logic"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=640, verbose=False)
    hasil_deteksi = []
    label_counts = {}

    for result in results:
        boxes = result.boxes
        names = model.names

        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = names[cls_id]

            # Filter buah sesuai focused_classes
            if label not in focused_classes or confidence < CONFIDENCE_THRESHOLD:
                continue

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 15 or box_h < 15:
                continue

            label_counts[label] = label_counts.get(label, 0) + 1

            roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            skor_warna, skor_kondisi = ekstrak_fitur_citra(roi, label)

            skor_fuzzy, kematangan, kesegaran, tag, warna_box = hitung_kesegaran_fuzzy(skor_warna, skor_kondisi)
            nama_buah = dapatkan_nama_buah(label)

            hasil_deteksi.append({
                "buah": nama_buah,
                "label": label,
                "confidence": confidence * 100.0,
                "kematangan": kematangan,
                "kesegaran": kesegaran,
                "skor_kesegaran": skor_fuzzy,
                "tag": tag
            })

            cv2.rectangle(frame, (x1, y1), (x2, y2), warna_box, 2)
            teks = f"{nama_buah.upper()} {tag} {confidence*100:.0f}%"
            cv2.putText(
                frame,
                teks,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_PLAIN,
                1.2,
                warna_box,
                2
            )

        if label_counts:
            summary = ", ".join([f"{v} {k}" for k, v in label_counts.items() if v > 0])
            box_padding = 12
            (text_width, text_height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (5, 5), (5 + text_width + box_padding, 5 + text_height + box_padding), (0, 255, 0), -1)
            cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "[YOLO26 SCANNER] Arahkan kamera ke buah...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    return frame, hasil_deteksi

def cetak_laporan(hasil_deteksi):
    """Menampilkan ringkasan hasil deteksi dan info nutrisi ke terminal"""
    print("\n" + "=" * 60)
    print("                 HASIL ANALISIS DETEKSI YOLO26")
    print("=" * 60)

    if not hasil_deteksi:
        print("  Tidak ada objek buah yang terdeteksi.")
        print("=" * 60)
        return

    for i, data in enumerate(hasil_deteksi, 1):
        nama = data["buah"]
        print(f"\n[{i}] Objek: {nama.upper()} (Label YOLO26: {data['label']})")
        print(f"    Confidence YOLO26  : {data['confidence']:.2f}%")
        print(f"    Tingkat Kematangan : {data['kematangan']}")
        print(f"    Kondisi Kesegaran  : {data['kesegaran']} {data['tag']}")
        print(f"    Skor Fuzzy         : {data['skor_kesegaran']:.2f}/100")

        if nama in database_nutrisi:
            info = database_nutrisi[nama]
            print(f"\n    INFORMASI NUTRISI ({info['nama']}):")
            print(f"    - Kalori          : {info['kalori']}")
            print(f"    - Kandungan Utama : {info['vitamin']}")
            print(f"    - Khasiat         : {info['khasiat']}")
            print(f"    - Fase Kematangan : {info['kematangan_info']}")
            print(f"    - Tips Pemilihan  : {info['tips']}")

    print("=" * 60)

# ----------------------------------------------------------------------
# 6. MENU SCAN GAMBAR (HANYA 1 PILIHAN: INPUT PATH KUSTOM)
# ----------------------------------------------------------------------
def mode_gambar():
    """Mode membaca gambar dari input path kustom langsung (sesuai ReadImageData.py)"""
    print("\n----------------------------------------------------------------")
    print("MODE SCAN GAMBAR YOLO26 (INPUT PATH KUSTOM)")
    print("----------------------------------------------------------------")
    print("Masukkan lokasi/path file gambar yang ingin di-scan.")
    print("Contoh: sample_images/freshApple_sample.jpg")
    print("(Tekan Enter langsung tanpa ketik untuk memakai sampel bawaan)")
    
    path = input("\nMasukkan path gambar: ").strip()

    if not path:
        default_samples = [
            "sample_images/freshApple_sample.jpg",
            "sample_images/rottenBanana_sample.jpg",
            "sample_images/freshCucumber_sample.jpg"
        ]
        for ds in default_samples:
            if os.path.exists(ds):
                path = ds
                break
        print(f"[INFO] Menggunakan gambar sampel default: {path}")

    if not path or not os.path.exists(path):
        print(f"[ERROR] File '{path}' tidak ditemukan!")
        return

    img = cv2.imread(path)
    if img is None:
        print("[ERROR] Gagal membaca gambar dengan OpenCV!")
        return

    print(f"\n[INFO] Memproses gambar dengan YOLO26: {path}")
    hasil_frame, hasil_deteksi = proses_deteksi(img)
    cetak_laporan(hasil_deteksi)

    cv2.imshow("Smart Fruit Scanner YOLO26 - Hasil Gambar", hasil_frame)
    print("Tekan tombol apa saja pada jendela gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mode_video(source=0):
    """Mode membaca video atau webcam secara real-time (sesuai ReadDisplayVideo.py & YOLO.py)"""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video atau kamera (Indeks: {source})!")
        return

    sumber_teks = f"Kamera Indeks {source}" if isinstance(source, int) else str(source)
    print(f"\n[INFO] Pemindaian Real-time YOLO26 dimulai pada: {sumber_teks}")
    print("Tekan 'q' pada jendela video untuk keluar.")

    terakhir_deteksi = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hasil_frame, deteksi = proses_deteksi(frame)
        if deteksi:
            terakhir_deteksi = deteksi

        cv2.imshow("Smart Fruit Scanner YOLO26 - Real-time", hasil_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if terakhir_deteksi:
        cetak_laporan(terakhir_deteksi)

# ----------------------------------------------------------------------
# 7. FITUR TRAINING YOLO26 PADA DATASET (MENGGUNAKAN GPU RTX 3050)
# ----------------------------------------------------------------------
def mode_training_yolo26():
    """Pelatihan model YOLO26 pada dataset buah menggunakan akselerasi GPU RTX 3050"""
    print("\n" + "=" * 64)
    print("      PELATIHAN (TRAINING) MODEL YOLO26 PADA DATASET")
    print("=" * 64)
    print("Fitur ini memanfaatkan GPU NVIDIA RTX 3050 untuk melatih model YOLO26")
    print("secara langsung pada dataset buah & sayur (17 kelas).")
    
    yaml_candidates = [
        "freshness  fruits and vegetables.v7i.yolov8/data.yaml",
        "data.yaml"
    ]
    yaml_path = None
    for y in yaml_candidates:
        if os.path.exists(y):
            yaml_path = y
            break

    if not yaml_path:
        custom_yaml = input("Masukkan path file data.yaml dataset: ").strip()
        if os.path.exists(custom_yaml):
            yaml_path = custom_yaml
        else:
            print("[ERROR] File data.yaml tidak ditemukan!")
            return

    print(f"\n[INFO] File Dataset YAML : {yaml_path}")
    base_model = input("Pilih base YOLO26 [Default: yolo26m.pt / yolo26s.pt / yolo26n.pt]: ").strip()
    if not base_model:
        base_model = "yolo26m.pt"

    epochs_in = input("Jumlah Epoch Training [Default 15]: ").strip()
    epochs = int(epochs_in) if epochs_in.isdigit() else 15

    print(f"\n[INFO] Memulai training {base_model} selama {epochs} epoch pada {device.upper()}...")
    train_model = YOLO(base_model)
    train_model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16 if device == "cuda" else 4,
        device=device,
        project="runs/fruit_freshness",
        name="yolo26",
        exist_ok=True
    )

    best_trained = "runs/fruit_freshness/yolo26/weights/best.pt"
    if os.path.exists(best_trained):
        shutil.copy(best_trained, "best_yolo26_fruit.pt")
        print(f"\n[SUKSES] Training selesai! Bobot tersimpan di: best_yolo26_fruit.pt")
        global model
        model = YOLO("best_yolo26_fruit.pt")
        model.to(device)
        print("[INFO] Model aktif diperbarui ke: best_yolo26_fruit.pt!")

# ----------------------------------------------------------------------
# 8. MENU UTAMA
# ----------------------------------------------------------------------
def main():
    while True:
        print(f"""
================================================================
          SMART FRUIT SCANNER (EDISI YOLO26 - RTX 3050)
  Sistem Deteksi Kematangan, Kesegaran, dan Nutrisi Buah
================================================================
Model Aktif: {model.ckpt_path or 'YOLO26'} ({device.upper()})
----------------------------------------------------------------
Pilihan:
[1] Scan dari Gambar (Input Path Kustom Langsung)
[2] Scan dari Video File
[3] Scan dari Webcam Laptop (Kamera 0)
[4] Scan dari Kamera Eksternal / USB (Kamera 1)
[5] Training Model YOLO26 pada Dataset (Akselerasi RTX 3050 GPU)
[0] Keluar
----------------------------------------------------------------""")
        pilihan = input("Pilih menu (0-5): ").strip()

        if pilihan == "1":
            mode_gambar()
        elif pilihan == "2":
            video_file = input("Masukkan path file video: ").strip()
            if video_file:
                mode_video(video_file)
            else:
                print("[ERROR] Path video tidak boleh kosong.")
        elif pilihan == "3":
            mode_video(0)
        elif pilihan == "4":
            idx = input("Masukkan indeks kamera eksternal [Default 1]: ").strip()
            cam_idx = int(idx) if idx.isdigit() else 1
            mode_video(cam_idx)
        elif pilihan == "5":
            mode_training_yolo26()
        elif pilihan == "0":
            print("\nProgram selesai. Terima kasih!")
            break
        else:
            print("\n[PERINGATAN] Pilihan tidak valid!")

if __name__ == "__main__":
    main()

