"""
SMART FRUIT SCANNER
Sistem Deteksi Kematangan, Kesegaran, dan Informasi Nutrisi Buah
================================================================
Dikembangkan dari Modul Praktikum AI:
1. YOLO.py             : Deteksi objek dan bounding box menggunakan YOLO
2. ReadImageData.py    : Membaca dan menampilkan citra menggunakan OpenCV
3. ReadDisplayVideo.py : Membaca video dan webcam frame-by-frame
4. FUZZY.py            : Inferensi Fuzzy Logic (Antecedent, Consequent, Rule)

Dataset: 17 kelas (9 komoditas buah/sayur):
apple, banana, bellpepper, carrot, cucumber, mango, orange, potato, tomato
================================================================
"""

import os
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------------------------------------------------------------
# 1. KONFIGURASI DAN LOAD MODEL
# (Sesuai dengan cara di YOLO.py praktikum)
# ----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Prioritaskan model YOLO paling akurat yang tersedia
model_candidates = [
    "best_fruit.pt",
    "yolo11m.pt",
    "runs/fruit_freshness/yolo11_accurate_fruit/weights/best.pt",
    "runs/fruit_freshness/smart_fruit_scanner/weights/best.pt",
    "yolov8n.pt"
]

model_path = "yolov8n.pt"
for candidate in model_candidates:
    if os.path.exists(candidate):
        model_path = candidate
        break

print(f"[INFO] Memuat model YOLO: {model_path} pada perangkat: {device.upper()}")
model = YOLO(model_path)
model.to(device)

# Batas kepercayaan (Confidence Threshold)
# Menggunakan 0.40 agar sangat akurat dan menyaring deteksi palsu (kursi, meja, dll)
CONFIDENCE_THRESHOLD = 0.40

# Daftar semua kelas buah dan sayur yang didukung
buah_classes = [
    # 17 Kelas dari Dataset Roboflow
    'fresh apple', 'fresh banana', 'fresh bellpepper', 'fresh carrot',
    'fresh cucumber', 'fresh mango', 'fresh orange', 'fresh potato',
    'rotten apple', 'rotten banana', 'rotten carrot', 'rotten cucumber',
    'rotten mango', 'rotten orange', 'rotten potato', 'rotten tomato',
    'rottenbellpepper',
    # Kelas dari model COCO (jika menggunakan yolo11m.pt / yolov8n.pt)
    'apple', 'banana', 'orange', 'carrot', 'broccoli'
]

# HANYA kelas buah yang dimasukkan ke focused_classes
# Objek non-buah (person, chair, meja, hp, dll) diabaikan sepenuhnya
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
        "vitamin": "Beta-karoten (Vitamin A), Serat",
        "khasiat": "Baik untuk kesehatan retina mata dan sistem imunitas.",
        "kematangan_info": "Wortel muda tekstur renyah manis, wortel matang warna oranye cerah merata.",
        "tips": "Pilih yang keras dari pangkal hingga ujung, warna cerah tidak kusam."
    },
    "cucumber": {
        "nama": "Mentimun",
        "kalori": "15 kkal / 100g",
        "vitamin": "Kadar Air 95%, Vitamin K, Mineral Silika",
        "khasiat": "Menghidrasi tubuh dan menjaga kesejukan pencernaan.",
        "kematangan_info": "Mentimun segar berwarna hijau tua padat. Jika menguning menandakan sudah lewat matang.",
        "tips": "Pilih yang lurus, padat, dan kulitnya hijau segar tidak lembek."
    },
    "mango": {
        "nama": "Mangga",
        "kalori": "60 kkal / 100g",
        "vitamin": "Vitamin C (36 mg), Vitamin A, Antioksidan",
        "khasiat": "Mendukung kesehatan mata dan regenerasi sel tubuh.",
        "kematangan_info": "Mangga mentah masam tinggi vitamin C, saat matang menjadi manis harum dan berair.",
        "tips": "Pilih yang pangkalnya harum dan sedikit lunak saat ditekan lembut."
    },
    "orange": {
        "nama": "Jeruk",
        "kalori": "47 kkal / 100g",
        "vitamin": "Vitamin C (53 mg), Asam Folat",
        "khasiat": "Menangkal radikal bebas dan menjaga daya tahan tubuh.",
        "kematangan_info": "Jeruk mentah kulit hijau dan asam, saat matang kulit oranye kekuningan dengan sari manis.",
        "tips": "Pilih buah yang terasa berat dan kulitnya halus berpori rapat."
    },
    "potato": {
        "nama": "Kentang",
        "kalori": "77 kkal / 100g",
        "vitamin": "Karbohidrat Kompleks, Kalium, Vitamin B6",
        "khasiat": "Sumber energi mengenyangkan pengganti nasi.",
        "kematangan_info": "Kentang baik berkulit cokelat mulus. Jika berwarna kehijauan mengandung solanin dan harus dihindari.",
        "tips": "Pilih umbi yang keras, kering, dan tidak memiliki mata tunas."
    },
    "tomato": {
        "nama": "Tomat",
        "kalori": "18 kkal / 100g",
        "vitamin": "Likopen (Antioksidan), Vitamin C, Kalium",
        "khasiat": "Menjaga kesehatan jantung dan pembuluh darah.",
        "kematangan_info": "Tomat hijau adalah fase mentah, saat matang merah menyala dengan kadar likopen tertinggi.",
        "tips": "Pilih warna merah merata, kelopak hijau masih menempel, dan kenyal padat."
    }
}

# ----------------------------------------------------------------------
# 3. SISTEM FUZZY LOGIC
# (Mengikuti struktur penulisan di FUZZY.py praktikum)
# ----------------------------------------------------------------------
def buat_sistem_fuzzy():
    # Variabel Input
    warna = ctrl.Antecedent(np.arange(0, 101, 1), 'warna')
    kondisi = ctrl.Antecedent(np.arange(0, 101, 1), 'kondisi')

    # Variabel Output
    kesegaran = ctrl.Consequent(np.arange(0, 101, 1), 'kesegaran')

    # Fungsi Keanggotaan (trimf seperti di FUZZY.py)
    warna['mentah'] = fuzz.trimf(warna.universe, [0, 0, 45])
    warna['matang'] = fuzz.trimf(warna.universe, [30, 60, 85])
    warna['lewat_matang'] = fuzz.trimf(warna.universe, [70, 100, 100])

    kondisi['buruk'] = fuzz.trimf(kondisi.universe, [0, 0, 50])
    kondisi['cukup'] = fuzz.trimf(kondisi.universe, [30, 50, 80])
    kondisi['baik'] = fuzz.trimf(kondisi.universe, [60, 100, 100])

    kesegaran['busuk'] = fuzz.trimf(kesegaran.universe, [0, 0, 40])
    kesegaran['sedang'] = fuzz.trimf(kesegaran.universe, [30, 50, 75])
    kesegaran['segar'] = fuzz.trimf(kesegaran.universe, [65, 100, 100])

    # Aturan Fuzzy (ctrl.Rule seperti di FUZZY.py)
    aturan1 = ctrl.Rule(kondisi['baik'] & warna['matang'], kesegaran['segar'])
    aturan2 = ctrl.Rule(kondisi['baik'] & warna['mentah'], kesegaran['segar'])
    aturan3 = ctrl.Rule(kondisi['cukup'] & warna['matang'], kesegaran['sedang'])
    aturan4 = ctrl.Rule(kondisi['cukup'] & warna['mentah'], kesegaran['sedang'])
    aturan5 = ctrl.Rule(kondisi['buruk'] | warna['lewat_matang'], kesegaran['busuk'])

    sistem_kontrol = ctrl.ControlSystem([aturan1, aturan2, aturan3, aturan4, aturan5])
    pengendali = ctrl.ControlSystemSimulation(sistem_kontrol)
    return pengendali

pengendali_fuzzy = buat_sistem_fuzzy()

def hitung_kesegaran_fuzzy(skor_warna, skor_kondisi):
    """Menghitung skor kesegaran menggunakan sistem fuzzy"""
    pengendali_fuzzy.input['warna'] = np.clip(skor_warna, 0, 100)
    pengendali_fuzzy.input['kondisi'] = np.clip(skor_kondisi, 0, 100)

    try:
        pengendali_fuzzy.compute()
        hasil = pengendali_fuzzy.output['kesegaran']
    except Exception:
        hasil = (skor_warna + skor_kondisi) / 2.0

    # Menentukan kategori kematangan
    if skor_warna < 35:
        kematangan = "Mentah"
    elif skor_warna < 70:
        kematangan = "Matang"
    else:
        kematangan = "Terlalu Matang"

    # Menentukan kategori kesegaran dan warna bounding box
    if hasil >= 65:
        status_kesegaran = "Segar Secara Visual"
        tag = "[SEGAR]"
        warna_box = (0, 255, 0)      # Hijau
    elif hasil >= 40:
        status_kesegaran = "Mulai Menurun"
        tag = "[SEDANG]"
        warna_box = (0, 255, 255)    # Kuning
    else:
        status_kesegaran = "Terindikasi Busuk"
        tag = "[BUSUK]"
        warna_box = (0, 0, 255)      # Merah

    return hasil, kematangan, status_kesegaran, tag, warna_box

# ----------------------------------------------------------------------
# 4. EKSTRAKSI FITUR CITRA MENGGUNAKAN OPENCV
# (Menggunakan fungsi dasar OpenCV: cvtColor, mean sederhana)
# ----------------------------------------------------------------------
def ekstrak_fitur_citra(roi, label_yolo):
    """Mengekstrak rata-rata warna dan kondisi visual buah menggunakan fungsi dasar OpenCV"""
    if roi is None or roi.size == 0:
        return 50.0, 50.0

    # Konversi ke ruang warna HSV menggunakan cv2.cvtColor dasar
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Rata-rata komponen warna (Hue), saturasi (Saturation), dan kecerahan (Value)
    h_mean = float(np.mean(hsv[:, :, 0]))
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))

    label_lower = label_yolo.lower()

    # Kasus 1: Label hasil deteksi mengandung status 'rotten' atau 'fresh'
    if "rotten" in label_lower:
        skor_kondisi = 20.0
        skor_warna = 85.0
        return skor_warna, skor_kondisi
    elif "fresh" in label_lower:
        skor_kondisi = 85.0
        skor_warna = 25.0 if (35 <= h_mean <= 85) else 55.0
        return skor_warna, skor_kondisi

    # Kasus 2: Label umum (misal 'apple', 'banana', 'orange', 'carrot' dari YOLO)
    # Ditentukan secara objektif menggunakan saturasi dan kecerahan OpenCV sederhana
    if s_mean < 70 or v_mean < 70:
        # Warna kusam, gelap, atau memudar (indikasi penurunan kesegaran / busuk)
        skor_kondisi = 25.0
        skor_warna = 85.0
    elif s_mean < 95:
        # Kualitas sedang / mulai menurun
        skor_kondisi = 55.0
        skor_warna = 60.0
    else:
        # Warna cerah dan segar
        skor_kondisi = 85.0
        skor_warna = 25.0 if (35 <= h_mean <= 85) else 55.0

    return skor_warna, skor_kondisi

def dapatkan_nama_buah(label):
    """Mengambil nama buah murni dari label deteksi YOLO"""
    nama = label.lower().replace("fresh", "").replace("rotten", "").strip()
    return nama

# ----------------------------------------------------------------------
# 5. PEMROSESAN FRAME DAN TAMPILAN
# (Mengikuti gaya visualisasi di YOLO.py praktikum)
# ----------------------------------------------------------------------
def proses_deteksi(frame):
    """Menjalankan deteksi YOLO, ekstraksi citra OpenCV, dan evaluasi Fuzzy Logic"""
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    hasil_deteksi = []
    label_counts = {}

    frame_h, frame_w = frame.shape[:2]
    total_area = frame_h * frame_w

    for result in results:
        boxes = result.boxes
        names = model.names

        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = names[cls_id]

            # 1. Filter: Hanya proses objek buah sesuai focused_classes
            if label not in focused_classes or confidence < CONFIDENCE_THRESHOLD:
                continue

            # Abaikan noise artefak sangat kecil (kurang dari 15x15 piksel)
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 15 or box_h < 15:
                continue

            label_counts[label] = label_counts.get(label, 0) + 1

            # Ambil area ROI buah
            roi = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            skor_warna, skor_kondisi = ekstrak_fitur_citra(roi, label)

            # Evaluasi Fuzzy Logic
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

            # Gambar bounding box berwarna berdasarkan kesegaran (Hijau/Kuning/Merah)
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

        # Tampilkan ringkasan jumlah objek di pojok kiri atas (sesuai YOLO.py praktikum)
        if label_counts:
            summary = ", ".join([f"{v} {k}" for k, v in label_counts.items() if v > 0])
            box_padding = 12
            (text_width, text_height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (5, 5), (5 + text_width + box_padding, 5 + text_height + box_padding), (0, 255, 0), -1)
            cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # Petunjuk jika belum ada buah yang terdeteksi di depan kamera
            cv2.putText(frame, "[SCANNER] Arahkan kamera ke buah...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    return frame, hasil_deteksi

def cetak_laporan(hasil_deteksi):
    """Menampilkan ringkasan hasil deteksi dan info nutrisi ke terminal"""
    print("\n" + "=" * 60)
    print("                 HASIL ANALISIS DETEKSI")
    print("=" * 60)

    if not hasil_deteksi:
        print("  Tidak ada objek buah yang terdeteksi.")
        print("=" * 60)
        return

    for i, data in enumerate(hasil_deteksi, 1):
        nama = data["buah"]
        print(f"\n[{i}] Objek: {nama.upper()} (Label YOLO: {data['label']})")
        print(f"    Confidence YOLO    : {data['confidence']:.2f}%")
        print(f"    Tingkat Kematangan : {data['kematangan']}")
        print(f"    Kondisi Kesegaran  : {data['kesegaran']} {data['tag']}")
        print(f"    Skor Fuzzy         : {data['skor_kesegaran']:.2f}/100")

        # Tampilkan informasi nutrisi dari database
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
# 6. MENU UTAMA PROGRAM
# ----------------------------------------------------------------------
def mode_gambar():
    """Mode membaca gambar dengan pilihan preset atau path kustom (sesuai ReadImageData.py)"""
    search_dirs = [
        "sample_images",
        "freshness  fruits and vegetables.v7i.yolov8/test/images"
    ]

    def cari_contoh(prefix):
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                matches = glob.glob(f"{s_dir}/{prefix}*")
                if matches:
                    return matches[0]
        return None

    print("""
----------------------------------------------------------------
PILIH SUMBER GAMBAR:
[1] Contoh Apel Segar     (freshApple)
[2] Contoh Apel Busuk     (rottenApple)
[3] Contoh Pisang Segar   (freshBanana)
[4] Contoh Pisang Busuk   (rottenBanana)
[5] Contoh Mentimun Segar (freshCucumber)
[6] Contoh Mangga Busuk   (rottenMango)
[7] Masukkan Path Gambar Sendiri / Kustom
----------------------------------------------------------------""")
    pilihan_img = input("Pilih (1-7) [Default 1]: ").strip()

    path = None
    if pilihan_img == "1" or pilihan_img == "":
        path = cari_contoh("freshApple")
    elif pilihan_img == "2":
        path = cari_contoh("rottenApple")
    elif pilihan_img == "3":
        path = cari_contoh("freshBanana")
    elif pilihan_img == "4":
        path = cari_contoh("rottenBanana")
    elif pilihan_img == "5":
        path = cari_contoh("freshCucumber")
    elif pilihan_img == "6":
        path = cari_contoh("rottenMango")
    elif pilihan_img == "7":
        path = input("Masukkan path gambar: ").strip()

    if not path or not os.path.exists(path):
        print(f"[ERROR] File '{path}' tidak ditemukan!")
        return

    img = cv2.imread(path)
    if img is None:
        print("[ERROR] Gagal membaca gambar!")
        return

    print(f"\n[INFO] Memproses gambar: {path}")
    hasil_frame, hasil_deteksi = proses_deteksi(img)
    cetak_laporan(hasil_deteksi)

    # Tampilkan jendela gambar seperti di ReadImageData.py
    cv2.imshow("Smart Fruit Scanner - Hasil Gambar", hasil_frame)
    print("Tekan tombol apa saja pada jendela gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def mode_video(source=0):
    """Mode membaca video atau webcam (sesuai ReadDisplayVideo.py & YOLO.py)"""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video atau kamera (Indeks: {source})!")
        print("Tips: Pastikan kamera eksternal sudah terhubung via USB dan tidak sedang digunakan aplikasi lain.")
        return

    sumber_teks = f"Kamera Indeks {source}" if isinstance(source, int) else str(source)
    print(f"\n[INFO] Pemindaian dimulai pada: {sumber_teks}")
    print("Tips: Dekatkan buah ke depan kamera agar terdeteksi dengan jelas.")
    print("Tekan 'q' pada jendela video untuk keluar.")

    terakhir_deteksi = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hasil_frame, deteksi = proses_deteksi(frame)
        if deteksi:
            terakhir_deteksi = deteksi

        cv2.imshow("Smart Fruit Scanner - Real-time", hasil_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if terakhir_deteksi:
        cetak_laporan(terakhir_deteksi)

def main():
    while True:
        print("""
================================================================
                    SMART FRUIT SCANNER
  Sistem Deteksi Kematangan, Kesegaran, dan Nutrisi Buah
================================================================
Pilihan:
[1] Scan dari Gambar (Preset Dataset / File Sendiri)
[2] Scan dari Video
[3] Scan dari Webcam Laptop (Kamera 0)
[4] Scan dari Kamera Eksternal / USB (Kamera 1)
[0] Keluar
----------------------------------------------------------------""")
        pilihan = input("Pilih menu (0-4): ").strip()

        if pilihan == "1":
            mode_gambar()
        elif pilihan == "2":
            video_file = input("Masukkan path file video (misal: mentahan.mp4): ").strip()
            if video_file:
                mode_video(video_file)
            else:
                print("[ERROR] Path video tidak boleh kosong.")
        elif pilihan == "3":
            # Kamera internal laptop (indeks 0)
            mode_video(0)
        elif pilihan == "4":
            # Kamera eksternal USB (biasanya indeks 1, atau bisa dimasukkan angka lain)
            idx = input("Masukkan indeks kamera eksternal [Default 1]: ").strip()
            cam_idx = int(idx) if idx.isdigit() else 1
            mode_video(cam_idx)
        elif pilihan == "0":
            print("\nProgram selesai. Terima kasih!")
            break
        else:
            print("\n[PERINGATAN] Pilihan tidak valid!")

if __name__ == "__main__":
    main()
