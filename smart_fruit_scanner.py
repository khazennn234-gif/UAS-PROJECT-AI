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

# Path model: jika ada model hasil training gunakan best.pt, jika belum gunakan yolov8n.pt
model_path = "runs/fruit_freshness/smart_fruit_scanner/weights/best.pt"
if not os.path.exists(model_path):
    model_path = "yolov8n.pt"

model = YOLO(model_path)
model.to(device)

# Daftar buah dan sayuran
buah_classes = [
    'fresh apple', 'fresh banana', 'fresh bellpepper', 'fresh carrot',
    'fresh cucumber', 'fresh mango', 'fresh orange', 'fresh potato',
    'rotten apple', 'rotten banana', 'rotten carrot', 'rotten cucumber',
    'rotten mango', 'rotten orange', 'rotten potato', 'rotten tomato',
    'rottenbellpepper',
    'apple', 'banana', 'orange', 'carrot', 'broccoli'
]

# Daftar kelas fokus gabungan (mencakup buah dan objek umum seperti person dari YOLO.py)
focused_classes = buah_classes + ['person', 'cell phone', 'cup', 'bottle', 'bowl', 'chair']

# ----------------------------------------------------------------------
# 2. BASIS PENGETAHUAN NUTRISI (9 Komoditas dari Dataset)
# ----------------------------------------------------------------------
database_nutrisi = {
    "apple": {
        "nama": "Apel",
        "kalori": "52 kkal / 100g",
        "vitamin": "Vitamin C (4.6 mg), Serat Pektin",
        "khasiat": "Menjaga kesehatan jantung dan melancarkan pencernaan.",
        "kematangan_info": "Saat matang kandungan pektin dan rasa manis seimbang. Jika terlalu matang tekstur bertepung.",
        "tips": "Pilih yang padat saat ditekan dan kulitnya tidak keriput."
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
        "kematangan_info": "Wortel muda/mentah tekstur renyah keras, wortel matang warna oranye cerah merata.",
        "tips": "Pilih yang keras dari pangkal hingga ujung, warna cerah tidak kusam."
    },
    "cucumber": {
        "nama": "Mentimun",
        "kalori": "15 kkal / 100g",
        "vitamin": "Kadar Air 95%, Vitamin K, Mineral Silika",
        "khasiat": "Menghidrasi tubuh dan menjaga kesejukan pencernaan.",
        "kematangan_info": "Mentimun segar berwarna hijau tua padat. Jika menguning menandakan sudah lewat matang.",
        "tips": "Pilih yang lurus, padat, dan kulitnya hijau segar tidak keriput."
    },
    "mango": {
        "nama": "Mangga",
        "kalori": "60 kkal / 100g",
        "vitamin": "Vitamin C (36 mg), Vitamin A, Antioksidan",
        "khasiat": "Mendukung kesehatan mata dan regenerasi sel tubuh.",
        "kematangan_info": "Mangga mentah masam tinggi vitamin C, saat matang menjadi manis harum dan berair.",
        "tips": "Pilih yang pangkalnya mengeluarkan aroma manis dan sedikit lunak saat ditekan."
    },
    "orange": {
        "nama": "Jeruk",
        "kalori": "47 kkal / 100g",
        "vitamin": "Vitamin C (53 mg), Asam Folat",
        "khasiat": "Menangkal radikal bebas dan menjaga daya tahan tubuh.",
        "kematangan_info": "Jeruk mentah kulit hijau dan asam, saat matang kulit oranye kekuningan dengan sari manis.",
        "tips": "Pilih buah yang terasa berat dan kulitnya halus tidak keropos."
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
        "tips": "Pilih warna merah merata, kelopak hijau masih menempel, dan kenyal."
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

    # Fungsi Keanggotaan (Membership Function trimf seperti di FUZZY.py)
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

    # Menentukan kategori kematangan dan kesegaran
    if skor_warna < 35:
        kematangan = "Mentah"
    elif skor_warna < 70:
        kematangan = "Matang"
    else:
        kematangan = "Terlalu Matang"

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
# 4. EKSTRAKSI FITUR CITRA SEDERHANA MENGGUNAKAN OPENCV
# (Menggunakan fungsi dasar OpenCV: cvtColor, mean)
# ----------------------------------------------------------------------
def ekstrak_fitur_citra(roi, label_yolo):
    """Mengekstrak rata-rata warna dan kondisi citra buah"""
    if roi is None or roi.size == 0:
        return 50.0, 50.0

    # Konversi ke HSV untuk melihat rata-rata komponen warna
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:, :, 0])  # Komponen warna (Hue)
    v_mean = np.mean(hsv[:, :, 2])  # Komponen kecerahan (Value)

    # Indikator warna kematangan sederhana (0 - 100)
    # Hue hijau berkisar 35-85, kuning/oranye/merah 0-30
    if 35 <= h_mean <= 85:
        skor_warna = 25.0   # Cenderung hijau (mentah)
    elif h_mean < 35 or h_mean > 160:
        skor_warna = 60.0   # Kuning / merah / oranye (matang)
    else:
        skor_warna = 80.0   # Cokelat / gelap (lewat matang)

    # Indikator kondisi visual berdasarkan deteksi YOLO dan kecerahan
    if "fresh" in label_yolo.lower():
        skor_kondisi = 85.0
    else:
        skor_kondisi = 25.0
        skor_warna = 90.0

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
    """Menjalankan deteksi YOLO, ekstraksi citra, dan Fuzzy Logic"""
    results = model(frame)
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

            # Filter confidence (0.20 agar sangat peka terhadap objek di depan kamera)
            if confidence > 0.20:
                label_counts[label] = label_counts.get(label, 0) + 1

                # Jika objek terdeteksi adalah buah/sayuran
                if label in buah_classes:
                    # Ambil area ROI buah
                    roi = frame[y1:y2, x1:x2]
                    skor_warna, skor_kondisi = ekstrak_fitur_citra(roi, label)

                    # Evaluasi Fuzzy
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

                    # Gambar bounding box berwarna berdasarkan kesegaran
                    cv2.rectangle(frame, (x1, y1), (x2, y2), warna_box, 2)
                    teks = f"{nama_buah.upper()} {tag} {confidence*100:.0f}%"
                    cv2.putText(
                        frame,
                        teks,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.2,
                        warna_box,
                        2
                    )
                else:
                    # Jika objek umum seperti manusia (person), botol, dll sesuai YOLO.py praktikum
                    # Tetap gambar bounding box hijau agar pengguna tahu model aktif bekerja!
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {confidence*100:.0f}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.2,
                        (0, 255, 0),
                        2
                    )

        # Tampilkan teks ringkasan jumlah objek di pojok kiri atas (persis seperti di YOLO.py praktikum)
        if label_counts:
            summary = ", ".join([f"{v} {k}" for k, v in label_counts.items() if v > 0])
            box_padding = 12
            (text_width, text_height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (5, 5), (5 + text_width + box_padding, 5 + text_height + box_padding), (0, 255, 0), -1)
            cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

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
    """Mode membaca gambar (sesuai ReadImageData.py)"""
    path = input("\nMasukkan path gambar (atau tekan enter untuk gambar default): ").strip()
    if not path:
        # Gunakan contoh gambar dari dataset test jika kosong
        path = "freshness  fruits and vegetables.v7i.yolov8/test/images/freshApple-1-_jpg.rf.ee1e70df72b18faa032bf3f7b42e12cb.jpg"

    if not os.path.exists(path):
        print(f"[ERROR] File {path} tidak ditemukan!")
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
    # Buka video atau kamera langsung dengan cv2.VideoCapture persis seperti di praktikum
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka sumber video atau kamera (Indeks: {source})!")
        print("Tips: Pastikan kamera eksternal sudah terhubung via USB dan tidak sedang digunakan aplikasi lain.")
        return

    sumber_teks = f"Kamera Indeks {source}" if isinstance(source, int) else str(source)
    print(f"\n[INFO] Pemindaian dimulai pada: {sumber_teks}")
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
[1] Scan dari Gambar
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
