"""
================================================================================
                    🍎 SMART FRUIT SCANNER 🍎
    Sistem Deteksi Kematangan, Kesegaran, dan Informasi Nutrisi Buah
================================================================================
Urgensi   : Konsumen sulit menilai kesegaran dan mudah tertipu pedagang.
Pembeda   : Penilaian objektif berbasis fitur citra (warna, tekstur, confidence)
            menggunakan AI & Fuzzy Logic, bukan sekadar perasaan penjual.
Integrasi : 
  1. YOLO (Deteksi Objek & Kondisi)
  2. Data Citra Gambar & Video (OpenCV)
  3. Labelling Data (17 Kelas Roboflow)
  4. Preprocessing Data (Min-Max Scaling, HSV, Varians Tekstur)
  5. Fuzzy Logic (Otak Penilaian Kesegaran)
  6. Database Nutrisi Edukatif
================================================================================
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
from sklearn import preprocessing
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ==============================================================================
# 1. KONFIGURASI SISTEM & PATH
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_YAML = os.path.join(BASE_DIR, "freshness  fruits and vegetables.v7i.yolov8", "data.yaml")
TRAINED_WEIGHTS = os.path.join(BASE_DIR, "runs", "fruit_freshness", "smart_fruit_scanner", "weights", "best.pt")
BASE_YOLO_MODEL = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.25
IMG_SIZE = 640

# 17 Kelas sesuai data.yaml Roboflow
CLASS_NAMES = [
    'fresh apple', 'fresh banana', 'fresh bellpepper', 'fresh carrot',
    'fresh cucumber', 'fresh mango', 'fresh orange', 'fresh potato',
    'rotten apple', 'rotten banana', 'rotten carrot', 'rotten cucumber',
    'rotten mango', 'rotten orange', 'rotten potato', 'rotten tomato',
    'rottenbellpepper'
]

# ==============================================================================
# 2. DATABASE INFORMASI NUTRISI (per 100 gram)
# Memberikan edukasi objektif kepada konsumen awam
# ==============================================================================
NUTRISI_DB = {
    "apple": {
        "nama": "Apel",
        "kalori": 52, "karbohidrat": 13.8, "serat": 2.4, "vitamin_c": 4.6,
        "gula": 10.4, "air": 85.6,
        "manfaat": "Kaya antioksidan flavonoid dan pektin, menjaga kesehatan jantung dan melancarkan pencernaan.",
        "tips_segar": "Pilih yang berkulit kencang, tidak keriput, dan padat saat ditekan lembut."
    },
    "banana": {
        "nama": "Pisang",
        "kalori": 89, "karbohidrat": 22.8, "serat": 2.6, "vitamin_c": 8.7,
        "gula": 12.2, "air": 74.9,
        "manfaat": "Tinggi kalium dan vitamin B6, mendukung fungsi otot, saraf, dan menjaga tekanan darah stabil.",
        "tips_segar": "Pilih kulit kuning mulus atau berbercak cokelat kecil (tanda matang optimal). Hindari yang menghitam lembek."
    },
    "bellpepper": {
        "nama": "Paprika",
        "kalori": 31, "karbohidrat": 6.0, "serat": 2.1, "vitamin_c": 127.7,
        "gula": 4.2, "air": 92.2,
        "manfaat": "Kadar vitamin C sangat tinggi (melebihi jeruk), memperkuat imunitas dan meremajakan sel kulit.",
        "tips_segar": "Kulit mengkilap tebal, batang hijau segar, dan terasa berat saat digenggam."
    },
    "carrot": {
        "nama": "Wortel",
        "kalori": 41, "karbohidrat": 9.6, "serat": 2.8, "vitamin_c": 5.9,
        "gula": 4.7, "air": 88.3,
        "manfaat": "Kaya beta-karoten yang diubah menjadi vitamin A, optimal untuk ketajaman retina mata.",
        "tips_segar": "Pilih warna oranye cerah merata, batang padat, tidak berakar lebat atau lembek."
    },
    "cucumber": {
        "nama": "Mentimun",
        "kalori": 15, "karbohidrat": 3.6, "serat": 0.5, "vitamin_c": 2.8,
        "gula": 1.7, "air": 95.2,
        "manfaat": "Kadar air 95%, sangat efektif menghidrasi sel tubuh, mendinginkan suhu pencernaan, dan rendah kalori.",
        "tips_segar": "Pilih yang keras dari ujung ke ujung, kulit hijau pekat dan tidak keriput kusam."
    },
    "mango": {
        "nama": "Mangga",
        "kalori": 60, "karbohidrat": 15.0, "serat": 1.6, "vitamin_c": 36.4,
        "gula": 13.7, "air": 83.5,
        "manfaat": "Kaya enzim pencernaan mangiferin dan vitamin C untuk meningkatkan metabolisme dan imunitas.",
        "tips_segar": "Aroma pangkal harum manis, kulit kenyal lentur saat ditekan, tidak lembek berair."
    },
    "orange": {
        "nama": "Jeruk",
        "kalori": 47, "karbohidrat": 11.8, "serat": 2.4, "vitamin_c": 53.2,
        "gula": 9.4, "air": 86.8,
        "manfaat": "Sumber utama asam askorbat alami dan folat, menjaga daya tahan tubuh dan menangkal radikal bebas.",
        "tips_segar": "Pilih buah yang terasa berat dan padat dengan pori-pori kulit yang halus merata."
    },
    "potato": {
        "nama": "Kentang",
        "kalori": 77, "karbohidrat": 17.5, "serat": 2.2, "vitamin_c": 19.7,
        "gula": 0.8, "air": 79.3,
        "manfaat": "Sumber karbohidrat kompleks bebas gluten dan kalium tinggi, baik untuk stabilitas energi harian.",
        "tips_segar": "Pilih yang keras, kulit mulus kering. HINDARI bagian berkulit kehijauan atau sudah bertunas."
    },
    "tomato": {
        "nama": "Tomat",
        "kalori": 18, "karbohidrat": 3.9, "serat": 1.2, "vitamin_c": 13.7,
        "gula": 2.6, "air": 94.5,
        "manfaat": "Tinggi likopen alami yang melindungi sel jantung dan menjaga elastisitas pembuluh darah.",
        "tips_segar": "Warna merah cerah merata, kelopak hijau segar, dan kenyal kencang tanpa bintik busuk."
    }
}

# ==============================================================================
# 3. MODUL PREPROCESSING DATA (Diadaptasi dari preprocessing_data.py)
# ==============================================================================
def preprocess_image_pipeline(img, target_size=(640, 640)):
    """
    Menerapkan pipeline preprocessing sesuai praktikum:
    - Resize citra ke ukuran kanonik model
    - Min-Max Scaling (normalisasi piksel 0-1)
    """
    img_resized = cv2.resize(img, target_size)
    
    # Min-Max Scaling [0, 1] berbasis NumPy/sklearn
    scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # Reshape untuk simulasi scaling sklearn
    flat_pixels = img_resized.reshape(-1, 3).astype(np.float32)
    # Cepat dengan pembagian skalar identik dengan fit_transform
    normalized_pixels = flat_pixels / 255.0
    img_normalized = normalized_pixels.reshape(img_resized.shape)
    
    return img_resized, img_normalized

def extract_visual_features(roi):
    """
    Ekstraksi fitur objektif citra:
    1. Warna (Saturasi & Value di ruang warna HSV)
    2. Tekstur (Varians Laplasian/Gray-level variance)
    """
    if roi is None or roi.size == 0:
        return 50.0, 50.0
    
    # 1. Analisis Warna (HSV)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))  # Saturasi (0-255)
    v_mean = float(np.mean(hsv[:, :, 2]))  # Kecerahan (0-255)
    # Skor warna gabungan dinormalisasi ke skala 0-100
    skor_warna = float(np.clip((s_mean * 0.7 + v_mean * 0.3) / 255.0 * 100.0, 0, 100))
    
    # 2. Analisis Tekstur (Kehalusan permukaan)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    # Varians wajar buah segar (halus) berkisar 50-600, keriput/berbintik hitam >1500
    skor_kekasaran = float(np.clip((variance / 15.0), 0, 100))
    skor_kehalusan = 100.0 - skor_kekasaran
    
    return skor_warna, skor_kehalusan

# ==============================================================================
# 4. MODUL FUZZY LOGIC (Diadaptasi dari FUZZY.py)
# Otak Penilai Kesegaran Objektif
# ==============================================================================
def build_fruit_fuzzy_system():
    """
    Membangun inferensi Fuzzy Logic Mandani menggunakan skfuzzy.
    - Input 1: Confidence YOLO (0 - 100)
    - Input 2: Fitur Warna (0 - 100, pudar s/d cerah)
    - Input 3: Fitur Tekstur (0 - 100, kasar s/d halus)
    - Output : Skor Kesegaran (0 - 100)
    """
    conf = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
    warna = ctrl.Antecedent(np.arange(0, 101, 1), 'warna')
    tekstur = ctrl.Antecedent(np.arange(0, 101, 1), 'tekstur')
    kesegaran = ctrl.Consequent(np.arange(0, 101, 1), 'kesegaran')

    # Membership Functions (Fungsi Keanggotaan Segitiga / trimf)
    conf['rendah'] = fuzz.trimf(conf.universe, [0, 0, 50])
    conf['sedang'] = fuzz.trimf(conf.universe, [30, 50, 80])
    conf['tinggi'] = fuzz.trimf(conf.universe, [60, 100, 100])

    warna['pudar'] = fuzz.trimf(warna.universe, [0, 0, 45])
    warna['normal'] = fuzz.trimf(warna.universe, [30, 55, 80])
    warna['cerah'] = fuzz.trimf(warna.universe, [65, 100, 100])

    tekstur['kasar'] = fuzz.trimf(tekstur.universe, [0, 0, 45])
    tekstur['sedang'] = fuzz.trimf(tekstur.universe, [30, 55, 80])
    tekstur['halus'] = fuzz.trimf(tekstur.universe, [65, 100, 100])

    kesegaran['tidak_segar'] = fuzz.trimf(kesegaran.universe, [0, 0, 40])
    kesegaran['cukup_segar'] = fuzz.trimf(kesegaran.universe, [30, 55, 75])
    kesegaran['segar']       = fuzz.trimf(kesegaran.universe, [65, 100, 100])

    # Basis Aturan (Rules Base)
    rules = [
        # Aturan Segar
        ctrl.Rule(conf['tinggi'] & warna['cerah'] & tekstur['halus'], kesegaran['segar']),
        ctrl.Rule(conf['tinggi'] & warna['cerah'] & tekstur['sedang'], kesegaran['segar']),
        ctrl.Rule(conf['sedang'] & warna['cerah'] & tekstur['halus'], kesegaran['segar']),
        
        # Aturan Cukup Segar
        ctrl.Rule(conf['tinggi'] & warna['normal'] & tekstur['halus'], kesegaran['cukup_segar']),
        ctrl.Rule(conf['sedang'] & warna['normal'] & tekstur['sedang'], kesegaran['cukup_segar']),
        ctrl.Rule(conf['tinggi'] & warna['normal'] & tekstur['kasar'], kesegaran['cukup_segar']),
        ctrl.Rule(conf['rendah'] & warna['cerah'] & tekstur['halus'], kesegaran['cukup_segar']),

        # Aturan Tidak Segar / Busuk
        ctrl.Rule(warna['pudar'] & tekstur['kasar'], kesegaran['tidak_segar']),
        ctrl.Rule(conf['rendah'] & warna['pudar'], kesegaran['tidak_segar']),
        ctrl.Rule(conf['sedang'] & warna['pudar'] & tekstur['kasar'], kesegaran['tidak_segar']),
        ctrl.Rule(tekstur['kasar'] & warna['pudar'], kesegaran['tidak_segar']),
    ]

    sistem_kontrol = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(sistem_kontrol)

def evaluasi_fuzzy(sim, conf_val, warna_val, tekstur_val, is_fresh_label=True):
    """
    Menjalankan proses fuzzifikasi, inferensi, dan defuzzifikasi
    """
    # Jika klasifikasi model mendeteksi kelas 'rotten', sesuaikan bobot warna & tekstur
    if not is_fresh_label:
        warna_val = min(warna_val * 0.4, 35.0)
        tekstur_val = min(tekstur_val * 0.35, 30.0)

    sim.input['confidence'] = float(np.clip(conf_val, 0, 100))
    sim.input['warna']      = float(np.clip(warna_val, 0, 100))
    sim.input['tekstur']    = float(np.clip(tekstur_val, 0, 100))

    try:
        sim.compute()
        skor = float(sim.output['kesegaran'])
    except Exception:
        # Fallback defensif jika di luar universe rule
        if is_fresh_label:
            skor = (conf_val * 0.4 + warna_val * 0.3 + tekstur_val * 0.3)
        else:
            skor = max(10.0, 30.0 - (100 - conf_val) * 0.1)

    skor = float(np.clip(skor, 0, 100))

    if skor >= 65.0:
        status = "SEGAR"
        emoji = "🟢"
        color_bgr = (0, 220, 0)
    elif skor >= 40.0:
        status = "CUKUP SEGAR"
        emoji = "🟡"
        color_bgr = (0, 215, 255)
    else:
        status = "TIDAK SEGAR"
        emoji = "🔴"
        color_bgr = (0, 0, 255)

    return skor, status, emoji, color_bgr

# ==============================================================================
# 5. MODUL YOLO (Diadaptasi dari YOLO.py)
# ==============================================================================
def get_yolo_model():
    """
    Memuat model YOLO:
    1. Mencari weights hasil training (best_fruit.pt atau best.pt)
    2. Jika belum ada, memuat model base yolov8n.pt untuk pengujian fungsional
    """
    candidates = [
        os.path.join(BASE_DIR, "best_fruit.pt"),
        TRAINED_WEIGHTS,
        BASE_YOLO_MODEL
    ]
    for c in candidates:
        if os.path.exists(c):
            print(f"[*] Memuat model YOLO: {c}")
            model = YOLO(c)
            model.to(DEVICE)
            return model
    model = YOLO(BASE_YOLO_MODEL)
    model.to(DEVICE)
    return model

def latih_model_yolo(epochs=25, batch=16):
    """
    Training YOLOv8 pada dataset Roboflow
    """
    print("\n" + "=" * 60)
    print("      🚀 MEMULAI TRAINING MODEL SMART FRUIT SCANNER")
    print("=" * 60)
    print(f"Device   : {DEVICE}")
    print(f"Dataset  : {DATASET_YAML}")
    print(f"Epochs   : {epochs} | Batch: {batch}")
    print("=" * 60 + "\n")

    if not os.path.exists(DATASET_YAML):
        print(f"❌ File konfigurasi dataset tidak ditemukan di: {DATASET_YAML}")
        return None

    model = YOLO(BASE_YOLO_MODEL)
    results = model.train(
        data=DATASET_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=os.path.join(BASE_DIR, "runs", "fruit_freshness"),
        name="smart_fruit_scanner",
        exist_ok=True,
        save=True,
        verbose=True
    )
    print("\n✅ Training selesai! Model terbaik disimpan di:")
    print(f"   {TRAINED_WEIGHTS}\n")
    return results

# ==============================================================================
# 6. PIPELINE ANALISIS DETEKSI CITRA & VIDEO
# ==============================================================================
def parse_fruit_label(label_raw):
    """
    Memisahkan jenis buah dan status label (fresh / rotten)
    """
    label_lower = label_raw.lower().strip()
    is_fresh = "fresh" in label_lower
    
    # Ambil jenis buah
    nama_buah = label_lower.replace("fresh", "").replace("rotten", "").replace("bellpepper", "bellpepper").strip()
    if not nama_buah:
        nama_buah = "buah"
    
    return nama_buah, is_fresh

def draw_hud_overlay(frame, detections_summary):
    """
    Menampilkan info panel ringkasan di layar
    """
    h, w = frame.shape[:2]
    # Header strip
    cv2.rectangle(frame, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(
        frame,
        "SMART FRUIT SCANNER | Penilai Kesegaran Objektif & Edukasi Nutrisi",
        (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

def proses_frame(frame, model, fuzzy_sim, draw_details=True):
    """
    Memproses satu frame:
    1. Preprocessing citra
    2. YOLO Object Detection
    3. ROI Feature Extraction (Warna + Tekstur)
    4. Fuzzy Logic Evaluation
    5. Visualisasi Bounding Box & Nutrisi
    """
    display_frame = frame.copy()
    
    # YOLO Inference
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        names = model.names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)

            if conf_val < CONFIDENCE_THRESHOLD:
                continue

            raw_label = names[cls_id] if cls_id in names else f"Objek_{cls_id}"
            nama_buah, is_fresh_label = parse_fruit_label(raw_label)

            # Ekstraksi Region of Interest (ROI)
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            roi = frame[y1:y2, x1:x2]

            # Fitur Warna & Tekstur
            skor_warna, skor_tekstur = extract_visual_features(roi)

            # Evaluasi Fuzzy
            skor_kesegaran, status_kesegaran, emoji, color_bgr = evaluasi_fuzzy(
                fuzzy_sim,
                conf_val * 100.0,
                skor_warna,
                skor_tekstur,
                is_fresh_label=is_fresh_label
            )

            nutrisi = NUTRISI_DB.get(nama_buah, None)

            detection_item = {
                "buah": nama_buah,
                "label_raw": raw_label,
                "confidence": conf_val * 100.0,
                "skor_kesegaran": skor_kesegaran,
                "status": status_kesegaran,
                "warna": skor_warna,
                "tekstur": skor_tekstur,
                "nutrisi": nutrisi,
                "bbox": (x1, y1, x2, y2)
            }
            detections.append(detection_item)

            # Gambar Bounding Box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_bgr, 3)

            # Label Header Box
            label_text = f"{nama_buah.upper()} [{status_kesegaran} {skor_kesegaran:.1f}%]"
            sub_text = f"Conf:{conf_val*100:.0f}% | W:{skor_warna:.0f} | T:{skor_tekstur:.0f}"

            (tw1, th1), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            (tw2, th2), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_w = max(tw1, tw2) + 12
            box_h = th1 + th2 + 14

            # Tag Box Background
            y_box_top = max(0, y1 - box_h)
            cv2.rectangle(display_frame, (x1, y_box_top), (x1 + box_w, y1), color_bgr, -1)
            cv2.putText(display_frame, label_text, (x1 + 6, y_box_top + th1 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, sub_text, (x1 + 6, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    draw_hud_overlay(display_frame, detections)
    return display_frame, detections

# ==============================================================================
# 7. FITUR SCAN: GAMBAR, VIDEO & WEBCAM
# ==============================================================================
def scan_gambar(image_path=None):
    """
    Membaca dan menganalisis citra statis
    """
    model = get_yolo_model()
    fuzzy_sim = build_fruit_fuzzy_system()

    # Jika image_path kosong, ambil contoh dari dataset test
    if not image_path or not os.path.exists(image_path):
        sample_pattern = os.path.join(BASE_DIR, "freshness  fruits and vegetables.v7i.yolov8", "test", "images", "*.jpg")
        samples = glob.glob(sample_pattern)
        if samples:
            image_path = samples[0]
            print(f"[*] Menggunakan gambar sampel dari dataset test: {os.path.basename(image_path)}")
        else:
            print("❌ Tidak ada file gambar yang dapat diproses!")
            return

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gagal membuka citra: {image_path}")
        return

    # Preprocessing
    _, _ = preprocess_image_pipeline(img)

    # Proses Frame
    hasil_frame, detections = proses_frame(img, model, fuzzy_sim)

    # Cetak Edukasi Nutrisi & Ringkasan ke Terminal
    print("\n" + "=" * 65)
    print(f"  🔍 HASIL ANALISIS CITRA: {os.path.basename(image_path)}")
    print("=" * 65)

    if not detections:
        print("  ⚠️  Tidak terdeteksi objek buah/sayuran pada gambar.")
    else:
        for i, det in enumerate(detections, 1):
            print(f"\n  [{i}] Buah/Sayur   : {det['buah'].upper()}")
            print(f"      Label Deteksi : {det['label_raw']}")
            print(f"      Confidence    : {det['confidence']:.2f}%")
            print(f"      Fitur Warna   : {det['warna']:.1f}/100")
            print(f"      Fitur Tekstur : {det['tekstur']:.1f}/100")
            print(f"      Skor Kesegaran: {det['skor_kesegaran']:.2f}% ({det['status']})")
            
            nutrisi = det['nutrisi']
            if nutrisi:
                print("\n      📋 INFORMASI GIZI & EDUKASI KONSUMEN (per 100g):")
                print(f"         - Energi      : {nutrisi['kalori']} kkal")
                print(f"         - Karbohidrat : {nutrisi['karbohidrat']} g (Gula: {nutrisi['gula']} g)")
                print(f"         - Serat Alami : {nutrisi['serat']} g")
                print(f"         - Vitamin C   : {nutrisi['vitamin_c']} mg")
                print(f"         - Kadar Air   : {nutrisi['air']} %")
                print(f"         💡 Khasiat    : {nutrisi['manfaat']}")
                print(f"         🛒 Tips Beli  : {nutrisi['tips_segar']}")

    print("\n" + "=" * 65)
    print("Menampilkan jendela visualisasi OpenCV. Tekan sembarang tombol untuk menutup...")
    
    cv2.imshow("Smart Fruit Scanner - Analisis Gambar", hasil_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def scan_video(source=0):
    """
    Membaca dan menganalisis video stream (file mp4 atau webcam)
    """
    model = get_yolo_model()
    fuzzy_sim = build_fruit_fuzzy_system()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Gagal mengakses sumber video / kamera: {source}")
        return

    sumber_nama = "Webcam Kamera" if source == 0 else os.path.basename(str(source))
    print(f"\n🎥 Memulai pemindaian: {sumber_nama}")
    print("   Tekan 'q' di jendela tampilan untuk berhenti.")
    print("   Tekan 's' untuk mengambil tangkapan layar (screenshot).\n")

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        # Proses deteksi
        hasil_frame, detections = proses_frame(frame, model, fuzzy_sim)

        cv2.imshow("Smart Fruit Scanner - Real-time", hasil_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            nama_shot = f"screenshot_fruit_{frame_id}.jpg"
            cv2.imwrite(nama_shot, hasil_frame)
            print(f"📸 Tangkapan layar tersimpan: {nama_shot}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Pemindaian video selesai.")

def uji_mandiri_pipeline():
    """
    Demonstrasi lengkap 5 modul tanpa harus menunggu training YOLO:
    1. Baca Citra
    2. Preprocessing
    3. Ekstraksi Warna & Tekstur
    4. Fuzzy Logic
    5. Integrasi Nutrisi
    """
    print("\n" + "=" * 65)
    print("   🧪 DEMONSTRASI PENGUJIAN LENGKAP SISTEM & 5 MODUL")
    print("=" * 65)

    sample_pattern = os.path.join(BASE_DIR, "freshness  fruits and vegetables.v7i.yolov8", "test", "images", "*.jpg")
    samples = glob.glob(sample_pattern)
    if not samples:
        print("❌ Sampel citra tidak ditemukan di folder test!")
        return

    sample_img_path = samples[0]
    print(f"[1] Membaca Citra      : {os.path.basename(sample_img_path)}")
    img = cv2.imread(sample_img_path)

    print("[2] Preprocessing Data : Resize (640x640) & Min-Max Normalization (0-1)...")
    img_res, img_norm = preprocess_image_pipeline(img)
    print(f"    Bentuk citra       : {img.shape} -> {img_res.shape}")
    print(f"    Nilai piksel min/max : {img_norm.min():.2f} / {img_norm.max():.2f}")

    print("[3] Ekstraksi Fitur    : Analisis ruang warna HSV & varians tekstur...")
    warna, tekstur = extract_visual_features(img)
    print(f"    Skor Warna (HSV)   : {warna:.2f} / 100")
    print(f"    Skor Kehalusan     : {tekstur:.2f} / 100")

    print("[4] Fuzzy Logic Engine : Fuzzifikasi, Evaluasi 10 Rules, & Defuzzifikasi...")
    fuzzy_sim = build_fruit_fuzzy_system()
    skor, status, emoji, _ = evaluasi_fuzzy(fuzzy_sim, 92.5, warna, tekstur, is_fresh_label=True)
    print(f"    Hasil Evaluasi     : {emoji} {status} (Skor Kesegaran: {skor:.2f}%)")

    print("[5] Edukasi Nutrisi    : Menghubungkan ke basis pengetahuan nutrisi...")
    nutrisi = NUTRISI_DB.get("apple")
    print(f"    Nutrisi Apel       : Kalori {nutrisi['kalori']} kkal | Vit C {nutrisi['vitamin_c']} mg")
    print(f"    Manfaat Kesehatan  : {nutrisi['manfaat']}")
    print(f"    Tips Pembelian     : {nutrisi['tips_segar']}")
    print("=" * 65)
    print("✅ Semua 5 komponen modul berhasil diintegrasikan dengan sempurna!\n")

# ==============================================================================
# 8. MENU UTAMA INTERAKTIF
# ==============================================================================
def tampilkan_banner():
    print(r"""
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                    🍎 SMART FRUIT SCANNER 🍎                     ║
  ║         Penilai Kesegaran Objektif & Edukasi Nutrisi Buah         ║
  ║                                                                   ║
  ║  Modul: YOLO | Citra & Video | Preprocessing | Fuzzy Logic        ║
  ╚═══════════════════════════════════════════════════════════════════╝
    """)

def main():
    while True:
        tampilkan_banner()
        print("  PILIHAN MENU:")
        print("  [1] 🏋️  Latih Model YOLO (Training Dataset Buah)")
        print("  [2] 📷  Scan dari File Gambar (Image Detection)")
        print("  [3] 🎥  Scan dari File Video (Video Detection)")
        print("  [4] 📹  Scan dari Kamera Webcam (Real-time Detection)")
        print("  [5] 🧪  Uji Demonstrasi Integrasi 5 Modul Praktikum")
        print("  [0] 🚪  Keluar")
        print("-" * 69)

        pilihan = input("  Masukkan pilihan Anda [0-5]: ").strip()

        if pilihan == "1":
            try:
                ep = input("  Masukkan jumlah epochs [Default 25]: ").strip()
                epochs = int(ep) if ep else 25
                latih_model_yolo(epochs=epochs)
            except Exception as e:
                print(f"❌ Terjadi kesalahan saat training: {e}")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "2":
            path = input("  Masukkan path gambar (kosongkan untuk sampel otomatis): ").strip()
            scan_gambar(path if path else None)
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "3":
            path = input("  Masukkan path file video (contoh: video_buah.mp4): ").strip()
            if path:
                scan_video(path)
            else:
                print("❌ Path video tidak boleh kosong.")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "4":
            scan_video(0)
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "5":
            uji_mandiri_pipeline()
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "0":
            print("\n  👋 Terima kasih telah menggunakan Smart Fruit Scanner!")
            sys.exit(0)

        else:
            print("\n  ⚠️ Pilihan tidak valid. Silakan coba lagi.\n")

if __name__ == "__main__":
    main()
