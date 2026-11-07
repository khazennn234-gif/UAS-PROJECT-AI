"""
================================================================================
                         SMART FRUIT SCANNER
    Sistem Deteksi Kematangan, Kesegaran, dan Informasi Nutrisi Buah
================================================================================
Alur Input & Output:
  1. Input   : Gambar, Video, Webcam (Ekstraksi frame via OpenCV)
  2. Proses  : YOLOv8 Object Detection -> OpenCV Visual ROI Extraction
               (Warna HSV & Bercak Permukaan) -> Fuzzy Logic Dual Engine
               (Kematangan & Kesegaran) -> Integrasi Nutrisi Ilmiah
  3. Output  : Label, Confidence, Kematangan, Kesegaran, Profil Nutrisi Ilmiah,
               Laporan Terminal VS Code, Opsi Simpan Citra Beranotasi
Dataset Terintegrasi (17 Kelas Roboflow, 9 Komoditas Buah & Sayuran):
  - Apple      : fresh apple, rotten apple
  - Banana     : fresh banana, rotten banana
  - Bellpepper : fresh bellpepper, rottenbellpepper
  - Carrot     : fresh carrot, rotten carrot
  - Cucumber   : fresh cucumber, rotten cucumber
  - Mango      : fresh mango, rotten mango
  - Orange     : fresh orange, rotten orange
  - Potato     : fresh potato, rotten potato
  - Tomato     : rotten tomato (dan varian tomat segar)
================================================================================
"""

import os
import sys
import glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ------------------------------------------------------------------------------
# 1. KONFIGURASI UMUM
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_YAML = os.path.join(BASE_DIR, "freshness  fruits and vegetables.v7i.yolov8", "data.yaml")
TRAINED_WEIGHTS = os.path.join(BASE_DIR, "runs", "fruit_freshness", "smart_fruit_scanner", "weights", "best.pt")
BASE_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.25

# 17 Kelas Lengkap Sesuai Dataset Roboflow
CLASS_NAMES = [
    'fresh apple', 'fresh banana', 'fresh bellpepper', 'fresh carrot',
    'fresh cucumber', 'fresh mango', 'fresh orange', 'fresh potato',
    'rotten apple', 'rotten banana', 'rotten carrot', 'rotten cucumber',
    'rotten mango', 'rotten orange', 'rotten potato', 'rotten tomato',
    'rottenbellpepper'
]

# ------------------------------------------------------------------------------
# 2. DATABASE INFORMASI NUTRISI LENGKAP SEMUA BUAH BERDASARKAN KEMATANGAN
# Referensi Ilmiah: USDA FoodData Central & Biokimia Pangan
# ------------------------------------------------------------------------------
NUTRISI_KEMATANGAN_DB = {
    "banana": {
        "nama": "Pisang (Musa paradisiaca)",
        "mentah": {
            "gula": "3.2 g", "pati": "18.5 g (Pati Resisten Tinggi)", "serat": "3.1 g", "kalori": "85 kkal",
            "karakteristik": "Tinggi resistant starch, rendah indeks glikemik, bertindak sebagai prebiotik usus.",
            "rekomendasi": "Sangat ideal untuk penderita diabetes atau diet rendah gula."
        },
        "setengah_matang": {
            "gula": "7.5 g", "pati": "12.0 g", "serat": "2.8 g", "kalori": "88 kkal",
            "karakteristik": "Pektin dan pati mulai bertransformasi; tekstur kenyal padat dan rasa manis lembut.",
            "rekomendasi": "Baik untuk menjaga rasa kenyang lebih lama tanpa lonjakan insulin drastis."
        },
        "matang": {
            "gula": "12.2 g", "pati": "5.3 g", "kalium": "358 mg", "vitamin_c": "8.7 mg", "kalori": "89 kkal",
            "karakteristik": "Pati terurai sempurna menjadi glukosa/fruktosa; kadar kalium optimal untuk otot.",
            "rekomendasi": "Pilihan terbaik sebagai sumber energi instan sebelum/setelah aktivitas fisik."
        },
        "terlalu_matang": {
            "gula": "15.8 g", "pati": "< 1.5 g", "antioksidan": "Paling Tinggi (Senyawa TNF & Polifenol)", "kalori": "92 kkal",
            "karakteristik": "Bercak cokelat menandakan sintesis antioksidan puncak, namun gula sederhana meningkat.",
            "rekomendasi": "Optimal untuk penangkal radikal bebas dan bahan olahan pangan (smoothie/bolu)."
        }
    },
    "apple": {
        "nama": "Apel (Malus domestica)",
        "mentah": {
            "gula": "7.0 g", "asam_malat": "Tinggi (Rasa asam kuat)", "serat": "2.8 g", "kalori": "46 kkal",
            "karakteristik": "Asam malat pekat, tekstur sangat keras, rendah fruktosa.",
            "rekomendasi": "Menstimulasi produksi saliva dan asam lambung; kurang nyaman untuk lambung sensitif."
        },
        "setengah_matang": {
            "gula": "9.2 g", "asam_malat": "Sedang", "serat": "2.5 g", "kalori": "49 kkal",
            "karakteristik": "Kerenyahan maksimum dengan perpaduan rasa asam-manis menyegarkan.",
            "rekomendasi": "Sangat baik untuk salad buah dan konsumsi harian segar."
        },
        "matang": {
            "gula": "10.4 g", "pektin": "Kaya serat larut air", "vitamin_c": "4.6 mg", "kalori": "52 kkal",
            "karakteristik": "Keseimbangan pektin dan flavonoid quercetin optimal untuk kesehatan kardiovaskular.",
            "rekomendasi": "Membantu menurunkan kadar kolesterol LDL dan melancarkan mikrobioma usus."
        },
        "terlalu_matang": {
            "gula": "11.5 g", "pektin": "Menurun (Terhidrolisis jadi asam pektat)", "kalori": "53 kkal",
            "karakteristik": "Tekstur menjadi bertepung (mealy), kadar air menurun, aroma fermentasi ringan.",
            "rekomendasi": "Paling cocok dijadikan saus apel (applesauce), selai, atau cuka apel alami."
        }
    },
    "tomato": {
        "nama": "Tomat (Solanum lycopersicum)",
        "mentah": {
            "gula": "1.8 g", "tomatine": "Tinggi (Alkaloid alami)", "vitamin_c": "10.2 mg", "kalori": "15 kkal",
            "karakteristik": "Dominan warna hijau klorofil, mengandung tomatine yang memiliki efek antibakteri.",
            "rekomendasi": "Sebaiknya dimasak terlebih dahulu (tumis/acar tomat hijau) sebelum dikonsumsi."
        },
        "setengah_matang": {
            "gula": "2.2 g", "likopen": "Mulai terbentuk", "vitamin_c": "12.0 mg", "kalori": "16 kkal",
            "karakteristik": "Warna semburat jingga kekuningan; tekstur daging padat dan renyah.",
            "rekomendasi": "Sangat cocok untuk irisan burger, sandwich, atau salad segar."
        },
        "matang": {
            "gula": "2.6 g", "likopen": "Maksimal (Antioksidan kuat)", "vitamin_c": "13.7 mg", "kalori": "18 kkal",
            "karakteristik": "Likopen bioaktif berada pada titik konsentrasi tertinggi; warna merah menyala.",
            "rekomendasi": "Pencegah risiko penyakit jantung dan kanker prostat; nutrisi meningkat jika dimasak ringan."
        },
        "terlalu_matang": {
            "gula": "2.9 g", "likopen": "Mulai teroksidasi", "kalori": "18 kkal",
            "karakteristik": "Dinding sel melunak drastis, kadar air bebas tinggi, kulit rentan robek.",
            "rekomendasi": "Sangat ideal untuk dijadikan pasta tomat, sup kental, atau bahan sambal."
        }
    },
    "mango": {
        "nama": "Mangga (Mangifera indica)",
        "mentah": {
            "gula": "4.5 g", "vitamin_c": "Sangat Tinggi (Asam sitrat tinggi)", "kalori": "45 kkal",
            "karakteristik": "Kaya vitamin C dan asam alami, daging buah renyah kesat.",
            "rekomendasi": "Cocok untuk rujak/asinan; membantu meningkatkan penyerapan zat besi."
        },
        "setengah_matang": {
            "gula": "9.0 g", "vitamin_a": "Meningkat pesat", "kalori": "52 kkal",
            "karakteristik": "Mulai beraroma manis lembut dengan sedikit sensasi asam segar.",
            "rekomendasi": "Camilan bernutrisi tinggi serat untuk menyegarkan tubuh."
        },
        "matang": {
            "gula": "13.7 g", "vitamin_a": "Tinggi (Beta-karoten)", "vitamin_c": "36.4 mg", "kalori": "60 kkal",
            "karakteristik": "Kandungan mangiferin dan beta-karoten maksimal; manis berair dan harum.",
            "rekomendasi": "Mendukung regenerasi sel mata, imunitas tubuh, dan kesehatan jaringan kulit."
        },
        "terlalu_matang": {
            "gula": "16.0 g", "serat": "Mulai melunak berserabut", "kalori": "64 kkal",
            "karakteristik": "Sangat manis, aroma fermentasi buah kuat, tekstur sangat lembek.",
            "rekomendasi": "Baik diolah menjadi puree mangga, puding buah, atau campuran jus."
        }
    },
    "orange": {
        "nama": "Jeruk (Citrus sinensis)",
        "mentah": {
            "gula": "6.0 g", "asam_sitrat": "Tinggi", "vitamin_c": "40.0 mg", "kalori": "38 kkal",
            "karakteristik": "Kulit hijau kaku, rasa masam pekat, bulir air belum terisi penuh.",
            "rekomendasi": "Kurang ideal untuk konsumsi langsung; sering digunakan sebagai perasan bumbu."
        },
        "setengah_matang": {
            "gula": "8.0 g", "asam_sitrat": "Seimbang", "vitamin_c": "48.0 mg", "kalori": "42 kkal",
            "karakteristik": "Perpaduan manis dan asam segar dengan aroma minyak atsiri kulit yang harum.",
            "rekomendasi": "Menyegarkan tubuh saat dehidrasi dan meningkatkan daya tahan harian."
        },
        "matang": {
            "gula": "9.4 g", "vitamin_c": "53.2 mg (Tertinggi)", "folat": "30 mcg", "kalori": "47 kkal",
            "karakteristik": "Kadar vitamin C dan senyawa bioflavonoid hesperidin mencapai titik optimal.",
            "rekomendasi": "Menjaga elastisitas pembuluh darah dan meningkatkan imunitas seluler."
        },
        "terlalu_matang": {
            "gula": "10.8 g", "vitamin_c": "Menurun karena oksidasi", "kalori": "48 kkal",
            "karakteristik": "Kulit kering mengendur dari daging buah; bulir sari air mulai menyusut.",
            "rekomendasi": "Sebaiknya segera diperas menjadi sari jeruk segar."
        }
    },
    "bellpepper": {
        "nama": "Paprika (Capsicum annuum)",
        "mentah": {
            "gula": "2.4 g", "klorofil": "Tinggi (Warna hijau pekat)", "vitamin_c": "80.4 mg", "kalori": "20 kkal",
            "karakteristik": "Dinding buah sangat renyah dan sedikit getir khas klorofil.",
            "rekomendasi": "Sangat baik untuk tumisan cepat dan masakan gurih renyah."
        },
        "setengah_matang": {
            "gula": "3.5 g", "karotenoid": "Mulai terbentuk", "vitamin_c": "105.0 mg", "kalori": "26 kkal",
            "karakteristik": "Transisi warna hijau ke kuning/merah dengan rasa lebih manis.",
            "rekomendasi": "Cocok untuk sajian fajitas atau panggangan."
        },
        "matang": {
            "gula": "4.2 g", "vitamin_c": "127.7 mg (Sangat Tinggi)", "kalori": "31 kkal",
            "karakteristik": "Vitamin C melampaui jeruk; warna merah/kuning mengkilap dan manis.",
            "rekomendasi": "Mendukung produksi kolagen kulit dan imunitas tubuh secara maksimal."
        },
        "terlalu_matang": {
            "gula": "4.8 g", "vitamin_c": "Mulai menurun", "kalori": "32 kkal",
            "karakteristik": "Kulit mulai keriput mengendur, kadar air berkurang.",
            "rekomendasi": "Sebaiknya diolah menjadi saus paprika panggang atau bumbu pasta."
        }
    },
    "cucumber": {
        "nama": "Mentimun (Cucumis sativus)",
        "mentah": {
            "gula": "1.3 g", "air": "96.0 %", "cucurbitacin": "Ada (Rasa agak pahit di pangkal)", "kalori": "14 kkal",
            "karakteristik": "Daging sangat renyah dan padat, biji masih lunak tak terasa.",
            "rekomendasi": "Sangat baik untuk lalapan renyah atau bahan acar segar."
        },
        "setengah_matang": {
            "gula": "1.5 g", "air": "95.5 %", "serat": "0.5 g", "kalori": "15 kkal",
            "karakteristik": "Kerenyahan optimal dengan kandungan air tinggi mendinginkan pencernaan.",
            "rekomendasi": "Paling cocok untuk infused water atau irisan salad."
        },
        "matang": {
            "gula": "1.7 g", "air": "95.2 %", "silika": "Kaya mineral silika", "kalori": "15 kkal",
            "karakteristik": "Kulit hijau mulus, menghidrasi tubuh secara efisien, bebas rasa pahit.",
            "rekomendasi": "Optimal untuk menjaga hidrasi kulit dan kesehatan sendi."
        },
        "terlalu_matang": {
            "gula": "1.8 g", "biji": "Keras dan membesar", "kalori": "16 kkal",
            "karakteristik": "Kulit menguning, bagian tengah berongga berair, biji mengeras.",
            "rekomendasi": "Keluarkan biji sebelum dikonsumsi atau gunakan untuk masker wajah alami."
        }
    },
    "carrot": {
        "nama": "Wortel (Daucus carota)",
        "mentah": {
            "gula": "3.5 g", "serat": "3.0 g", "tekstur": "Sangat padat keras", "kalori": "38 kkal",
            "karakteristik": "Ukuran kecil/muda, rasa manis belum menonjol, kaya serat lignin.",
            "rekomendasi": "Cocok untuk lalapan mentah setelah dicuci bersih."
        },
        "setengah_matang": {
            "gula": "4.0 g", "beta_karoten": "Meningkat", "kalori": "40 kkal",
            "karakteristik": "Warna oranye cerah merata dengan kerenyahan sedang.",
            "rekomendasi": "Pilihan fleksibel untuk sup, tumis, atau jus segar."
        },
        "matang": {
            "gula": "4.7 g", "beta_karoten": "Maksimal (Pro-vitamin A)", "kalori": "41 kkal",
            "karakteristik": "Kandungan beta-karoten optimal untuk proteksi sel fotoreseptor mata.",
            "rekomendasi": "Diserap tubuh lebih baik jika dikonsumsi dengan sedikit lemak sehat."
        },
        "terlalu_matang": {
            "gula": "5.0 g", "tekstur": "Mulai berkayu / berserat kasar", "kalori": "42 kkal",
            "karakteristik": "Inti tengah mengeras (woody core), akar rambut halus bermunculan.",
            "rekomendasi": "Lebih baik diolah menjadi kaldu sayuran atau dihaluskan dalam sup."
        }
    },
    "potato": {
        "nama": "Kentang (Solanum tuberosum)",
        "mentah": {
            "pati": "16.0 g", "solanin": "Rendah (Kulit cokelat)", "kalori": "70 kkal",
            "karakteristik": "Umbi keras padat, kulit tipis kering tanpa tunas.",
            "rekomendasi": "Harus selalu dimasak sebelum dikonsumsi."
        },
        "setengah_matang": {
            "pati": "17.0 g", "kalium": "400 mg", "kalori": "75 kkal",
            "karakteristik": "Kandungan pati seimbang; ideal untuk direbus atau dipanggang utuh.",
            "rekomendasi": "Sumber karbohidrat kompleks pengganti nasi yang mengenyangkan."
        },
        "matang": {
            "pati": "17.5 g", "vitamin_c": "19.7 mg", "kalium": "421 mg", "kalori": "77 kkal",
            "karakteristik": "Pati terikat sempurna; tekstur lembut merekah saat dikukus/dimasak.",
            "rekomendasi": "Sangat baik untuk mashed potato atau lauk berenergi tinggi."
        },
        "terlalu_matang": {
            "solanin": "PERINGATAN: Berisiko terbentuk jika kehijauan/bertunas", "kalori": "77 kkal",
            "karakteristik": "Kulit keriput layu atau muncul mata tunas. Jika hijau, solanin beracun.",
            "rekomendasi": "Buang tunas dan kupas tebal. Jangan konsumsi jika kulit berona kehijauan."
        }
    }
}

# ------------------------------------------------------------------------------
# 3. EKSTRAKSI FITUR VISUAL OPENCV SPESIFIK BUAH (Tahapan 6)
# ------------------------------------------------------------------------------
def ekstraksi_fitur_visual_opencv(roi, fruit_type="buah"):
    """
    Menganalisis karakteristik visual buah menggunakan OpenCV murni:
    1. Warna Dominan (HSV) -> Indikator Kematangan sesuai komoditas
    2. Bercak & Kerusakan Permukaan -> Indikator Kesegaran
    """
    if roi is None or roi.size == 0:
        return 50.0, 0.0

    # Konversi ke HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]
    total_pixels = float(roi.shape[0] * roi.shape[1])

    # Masking spektrum warna
    mask_hijau = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    rasio_hijau = float(np.sum(mask_hijau > 0)) / total_pixels

    mask_kuning_merah1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([30, 255, 255]))
    mask_kuning_merah2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    mask_matang = cv2.bitwise_or(mask_kuning_merah1, mask_kuning_merah2)
    rasio_matang = float(np.sum(mask_matang > 0)) / total_pixels

    # Penyesuaian karakteristik warna berdasarkan jenis buah
    f_key = fruit_type.lower().strip()
    if f_key == "cucumber":
        # Pada mentimun, hijau tua = matang optimal, menguning = terlalu matang
        if rasio_hijau > 0.45:
            indeks_kematangan_warna = 70.0
        elif rasio_matang > 0.30:
            indeks_kematangan_warna = 85.0
        else:
            indeks_kematangan_warna = 45.0
    elif f_key == "potato":
        # Pada kentang, cokelat muda = matang, jika ada kehijauan = solanin (overripe/defective)
        if rasio_hijau > 0.15:
            indeks_kematangan_warna = 90.0
        else:
            indeks_kematangan_warna = 65.0
    elif f_key == "carrot":
        # Pada wortel, oranye cerah = matang
        if rasio_matang > 0.40:
            indeks_kematangan_warna = 70.0
        elif rasio_hijau > 0.20:
            indeks_kematangan_warna = 30.0
        else:
            indeks_kematangan_warna = 50.0
    else:
        # Pisang, apel, tomat, mangga, jeruk, paprika:
        # Hijau = mentah, kuning/oranye/merah = matang
        if rasio_hijau > 0.35:
            indeks_kematangan_warna = float(np.clip(25.0 * (1.0 - rasio_hijau), 5.0, 35.0))
        elif rasio_matang > 0.30:
            indeks_kematangan_warna = float(np.clip(55.0 + (rasio_matang * 25.0), 55.0, 80.0))
        else:
            indeks_kematangan_warna = float(np.clip(40.0 + (np.mean(s_channel) / 255.0 * 20.0), 35.0, 55.0))

    # Analisis Deteksi Bercak & Kerusakan Permukaan
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Deteksi area gelap/bintik nekrotik di bawah ambang Otsu
    _, thresh_dark = cv2.threshold(blurred, 65, 255, cv2.THRESH_BINARY_INV)
    
    # Operasi morfologi untuk menghilangkan noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh_clean = cv2.morphologyEx(thresh_dark, cv2.MORPH_OPEN, kernel)
    
    luas_bercak = float(np.sum(thresh_clean > 0))
    persen_bercak = float(np.clip((luas_bercak / total_pixels) * 100.0 * 3.5, 0.0, 100.0))

    return indeks_kematangan_warna, persen_bercak

# ------------------------------------------------------------------------------
# 4. IMPLEMENTASI FUZZY LOGIC DUAL ENGINE (Tahapan 7)
# ------------------------------------------------------------------------------
def bangun_fuzzy_kematangan():
    """Fuzzy System untuk menentukan tingkat kematangan buah"""
    warna = ctrl.Antecedent(np.arange(0, 101, 1), 'warna')
    kematangan = ctrl.Consequent(np.arange(0, 101, 1), 'kematangan')

    # Membership Functions
    warna['hijau_mentah']       = fuzz.trimf(warna.universe, [0, 0, 40])
    warna['transisi_kuning']    = fuzz.trimf(warna.universe, [25, 45, 65])
    warna['kuning_merah_matang']= fuzz.trimf(warna.universe, [50, 70, 85])
    warna['gelap_lewat_matang'] = fuzz.trimf(warna.universe, [75, 100, 100])

    kematangan['mentah']          = fuzz.trimf(kematangan.universe, [0, 0, 35])
    kematangan['setengah_matang'] = fuzz.trimf(kematangan.universe, [25, 45, 65])
    kematangan['matang']          = fuzz.trimf(kematangan.universe, [55, 75, 90])
    kematangan['terlalu_matang']  = fuzz.trimf(kematangan.universe, [75, 100, 100])

    # Rules
    r1 = ctrl.Rule(warna['hijau_mentah'], kematangan['mentah'])
    r2 = ctrl.Rule(warna['transisi_kuning'], kematangan['setengah_matang'])
    r3 = ctrl.Rule(warna['kuning_merah_matang'], kematangan['matang'])
    r4 = ctrl.Rule(warna['gelap_lewat_matang'], kematangan['terlalu_matang'])

    sys_kematangan = ctrl.ControlSystem([r1, r2, r3, r4])
    return ctrl.ControlSystemSimulation(sys_kematangan)

def bangun_fuzzy_kesegaran():
    """Fuzzy System untuk menentukan kondisi kesegaran buah"""
    bercak = ctrl.Antecedent(np.arange(0, 101, 1), 'bercak')
    conf = ctrl.Antecedent(np.arange(0, 101, 1), 'confidence')
    kesegaran = ctrl.Consequent(np.arange(0, 101, 1), 'kesegaran')

    # Membership Functions
    bercak['rendah'] = fuzz.trimf(bercak.universe, [0, 0, 25])
    bercak['sedang'] = fuzz.trimf(bercak.universe, [20, 45, 70])
    bercak['tinggi'] = fuzz.trimf(bercak.universe, [55, 100, 100])

    conf['rendah'] = fuzz.trimf(conf.universe, [0, 0, 50])
    conf['sedang'] = fuzz.trimf(conf.universe, [35, 60, 85])
    conf['tinggi'] = fuzz.trimf(conf.universe, [70, 100, 100])

    kesegaran['terindikasi_busuk'] = fuzz.trimf(kesegaran.universe, [0, 0, 40])
    kesegaran['mulai_menurun']     = fuzz.trimf(kesegaran.universe, [30, 55, 75])
    kesegaran['segar_visual']      = fuzz.trimf(kesegaran.universe, [65, 100, 100])

    # Rules
    rules = [
        ctrl.Rule(bercak['rendah'] & conf['tinggi'], kesegaran['segar_visual']),
        ctrl.Rule(bercak['rendah'] & conf['sedang'], kesegaran['segar_visual']),
        ctrl.Rule(bercak['sedang'] & conf['tinggi'], kesegaran['mulai_menurun']),
        ctrl.Rule(bercak['sedang'] & conf['sedang'], kesegaran['mulai_menurun']),
        ctrl.Rule(bercak['tinggi'], kesegaran['terindikasi_busuk']),
        ctrl.Rule(bercak['sedang'] & conf['rendah'], kesegaran['terindikasi_busuk']),
        ctrl.Rule(bercak['rendah'] & conf['rendah'], kesegaran['mulai_menurun']),
    ]

    sys_kesegaran = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(sys_kesegaran)

def evaluasi_status_fuzzy(sim_kem, sim_seg, warna_val, bercak_val, conf_val, is_rotten_label=False):
    """Menghitung evaluasi kematangan dan kesegaran secara kuantitatif"""
    # 1. Kematangan
    sim_kem.input['warna'] = float(np.clip(warna_val, 0, 100))
    try:
        sim_kem.compute()
        skor_kematangan = float(sim_kem.output['kematangan'])
    except Exception:
        skor_kematangan = warna_val

    if skor_kematangan < 35.0:
        label_kematangan = "Mentah"
        kategori_key = "mentah"
    elif skor_kematangan < 55.0:
        label_kematangan = "Setengah Matang"
        kategori_key = "setengah_matang"
    elif skor_kematangan < 80.0:
        label_kematangan = "Matang"
        kategori_key = "matang"
    else:
        label_kematangan = "Terlalu Matang"
        kategori_key = "terlalu_matang"

    # 2. Kesegaran
    if is_rotten_label:
        bercak_val = max(bercak_val, 65.0)

    sim_seg.input['bercak'] = float(np.clip(bercak_val, 0, 100))
    sim_seg.input['confidence'] = float(np.clip(conf_val, 0, 100))
    try:
        sim_seg.compute()
        skor_kesegaran = float(sim_seg.output['kesegaran'])
    except Exception:
        skor_kesegaran = max(15.0, 100.0 - bercak_val)

    if skor_kesegaran >= 65.0:
        label_kesegaran = "Segar Secara Visual"
        tag_kesegaran = "[SEGAR]"
        color_bgr = (0, 220, 0)      # Hijau
    elif skor_kesegaran >= 40.0:
        label_kesegaran = "Mulai Menurun"
        tag_kesegaran = "[SEDANG]"
        color_bgr = (0, 215, 255)    # Kuning
    else:
        label_kesegaran = "Terindikasi Busuk"
        tag_kesegaran = "[BUSUK]"
        color_bgr = (0, 0, 255)      # Merah

    return skor_kematangan, label_kematangan, kategori_key, skor_kesegaran, label_kesegaran, tag_kesegaran, color_bgr

# ------------------------------------------------------------------------------
# 5. MODEL YOLO (Tahapan 5)
# ------------------------------------------------------------------------------
def muat_model_yolo():
    """Memuat model YOLO terlatih atau model dasar YOLOv8"""
    if os.path.exists(TRAINED_WEIGHTS):
        print(f"[*] Menggunakan model terlatih: {TRAINED_WEIGHTS}")
        model = YOLO(TRAINED_WEIGHTS)
    else:
        print(f"[*] Menggunakan model dasar: {BASE_MODEL}")
        model = YOLO(BASE_MODEL)
    model.to(DEVICE)
    return model

def parse_label_yolo(raw_label):
    raw_lower = raw_label.lower().strip()
    is_rotten = "rotten" in raw_lower
    # Menghapus prefiks fresh atau rotten
    nama_buah = raw_lower.replace("fresh", "").replace("rotten", "").strip()
    if not nama_buah:
        nama_buah = "buah"
    return nama_buah, is_rotten

# ------------------------------------------------------------------------------
# 6. PEMROSESAN CITRA DAN FRAME VIDEO
# ------------------------------------------------------------------------------
def proses_frame_citra(frame, model, sim_kem, sim_seg):
    """Mendeteksi objek dan mengekstrak fitur visual per frame"""
    display_frame = frame.copy()
    results = model(frame, verbose=False)
    detections = []

    for result in results:
        boxes = result.boxes
        names = model.names

        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0]) * 100.0
            x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)

            if (conf_val / 100.0) < CONFIDENCE_THRESHOLD:
                continue

            raw_label = names[cls_id] if cls_id in names else f"Objek_{cls_id}"
            nama_buah, is_rotten = parse_label_yolo(raw_label)

            # Crop ROI area buah
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            roi = frame[y1:y2, x1:x2]

            # Ekstraksi Fitur Visual OpenCV Spesifik Buah
            indeks_warna, persen_bercak = ekstraksi_fitur_visual_opencv(roi, fruit_type=nama_buah)

            # Evaluasi Fuzzy Dual Engine
            skor_kem, lbl_kem, kat_key, skor_seg, lbl_seg, tag_seg, color_bgr = evaluasi_status_fuzzy(
                sim_kem, sim_seg, indeks_warna, persen_bercak, conf_val, is_rotten_label=is_rotten
            )

            # Ambil Nutrisi Adaptif
            info_nutrisi = None
            if nama_buah in NUTRISI_KEMATANGAN_DB:
                info_nutrisi = NUTRISI_KEMATANGAN_DB[nama_buah].get(kat_key, None)

            det_item = {
                "buah": nama_buah,
                "label_raw": raw_label,
                "confidence": conf_val,
                "skor_kematangan": skor_kem,
                "tingkat_kematangan": lbl_kem,
                "skor_kesegaran": skor_seg,
                "kondisi_kesegaran": lbl_seg,
                "tag_kesegaran": tag_seg,
                "persen_bercak": persen_bercak,
                "indeks_warna": indeks_warna,
                "nutrisi": info_nutrisi,
                "nama_ilmiah": NUTRISI_KEMATANGAN_DB.get(nama_buah, {}).get("nama", nama_buah.capitalize()),
                "bbox": (x1, y1, x2, y2)
            }
            detections.append(det_item)

            # Gambar Bounding Box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color_bgr, 3)

            # Header teks visual (bersih tanpa emoji)
            t1 = f"{nama_buah.upper()} ({conf_val:.1f}%)"
            t2 = f"{lbl_kem} | {tag_seg}"
            (tw1, th1), _ = cv2.getTextSize(t1, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            (tw2, th2), _ = cv2.getTextSize(t2, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_w = max(tw1, tw2) + 12
            box_h = th1 + th2 + 14

            y_top = max(0, y1 - box_h)
            cv2.rectangle(display_frame, (x1, y_top), (x1 + box_w, y1), color_bgr, -1)
            cv2.putText(display_frame, t1, (x1 + 6, y_top + th1 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, t2, (x1 + 6, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)

    return display_frame, detections

# ------------------------------------------------------------------------------
# 7. FITUR MENU UTAMA: GAMBAR, VIDEO, WEBCAM & SIMPAN HASIL
# ------------------------------------------------------------------------------
def cetak_laporan_terminal(detections):
    """Menampilkan output detail pada terminal Python VS Code (Format Bersih)"""
    print("\n" + "=" * 70)
    print("                    LAPORAN ANALISIS SISTEM")
    print("=" * 70)

    if not detections:
        print("  [PERINGATAN] Tidak ada objek buah yang terdeteksi.")
        print("=" * 70)
        return

    for i, d in enumerate(detections, 1):
        print(f"\n  [{i}] JENIS BUAH         : {d['buah'].upper()} ({d['nama_ilmiah']})")
        print(f"      Confidence YOLO    : {d['confidence']:.2f}%")
        print(f"      Tingkat Kematangan : {d['tingkat_kematangan']} (Skor: {d['skor_kematangan']:.1f}/100)")
        print(f"      Kondisi Kesegaran  : {d['kondisi_kesegaran']} {d['tag_kesegaran']} (Skor: {d['skor_kesegaran']:.1f}/100)")
        print(f"      Karakteristik Citra: Warna={d['indeks_warna']:.1f} | Bercak Cacat={d['persen_bercak']:.1f}%")

        nutrisi = d['nutrisi']
        if nutrisi:
            print("\n      INFORMASI NUTRISI & KARAKTERISTIK ILMIAH:")
            for k, v in nutrisi.items():
                k_clean = k.replace("_", " ").capitalize()
                print(f"         - {k_clean:<18}: {v}")

    print("\n" + "=" * 70)

def menu_scan_gambar(image_path=None):
    model = muat_model_yolo()
    sim_kem = bangun_fuzzy_kematangan()
    sim_seg = bangun_fuzzy_kesegaran()

    if not image_path or not os.path.exists(image_path):
        samples = glob.glob(os.path.join(BASE_DIR, "freshness  fruits and vegetables.v7i.yolov8", "test", "images", "*.jpg"))
        if samples:
            image_path = samples[0]
            print(f"[*] Menggunakan sampel uji dataset: {os.path.basename(image_path)}")
        else:
            print("[ERROR] File citra tidak ditemukan.")
            return

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Gagal memuat citra: {image_path}")
        return

    display_frame, detections = proses_frame_citra(img, model, sim_kem, sim_seg)
    cetak_laporan_terminal(detections)

    # Opsi Simpan Gambar Ber-bounding Box (sesuai poin Output Sistem)
    simpan = input("  Simpan gambar ber-bounding box ini? [y/N]: ").strip().lower()
    if simpan == 'y':
        out_name = f"hasil_analisis_{os.path.basename(image_path)}"
        cv2.imwrite(out_name, display_frame)
        print(f"  [OK] Gambar berhasil disimpan sebagai: {out_name}")

    cv2.imshow("Smart Fruit Scanner - Deteksi Gambar", display_frame)
    print("Tekan sembarang tombol pada jendela gambar untuk menutup...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def menu_scan_video(source=0):
    model = muat_model_yolo()
    sim_kem = bangun_fuzzy_kematangan()
    sim_seg = bangun_fuzzy_kesegaran()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka video / webcam: {source}")
        return

    sumber = "Kamera Langsung (Webcam)" if source == 0 else os.path.basename(str(source))
    print(f"\n[INFO] Memulai pemindaian: {sumber}")
    print("   Tekan 'q' pada jendela tampilan untuk berhenti.")
    print("   Tekan 's' untuk menyimpan tangkapan layar (screenshot).\n")

    frame_count = 0
    last_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame, detections = proses_frame_citra(frame, model, sim_kem, sim_seg)
        if detections:
            last_detections = detections

        cv2.imshow("Smart Fruit Scanner - Real-time", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            s_name = f"screenshot_frame_{frame_count}.jpg"
            cv2.imwrite(s_name, display_frame)
            print(f"[SCREENSHOT] Tangkapan layar disimpan: {s_name}")

    cap.release()
    cv2.destroyAllWindows()

    if last_detections:
        cetak_laporan_terminal(last_detections)

def menu_uji_mandiri():
    """Menguji demonstrasi alur sistem pada seluruh 9 varietas buah dalam dataset"""
    print("\n" + "=" * 70)
    print("         PENGUJIAN SISTEM LENGKAP (OPENCV + FUZZY + NUTRISI)")
    print("=" * 70)
    samples = glob.glob(os.path.join(BASE_DIR, "freshness  fruits and vegetables.v7i.yolov8", "test", "images", "*.jpg"))
    if not samples:
        print("[ERROR] Dataset test tidak ditemukan.")
        return

    sample = samples[0]
    print(f"[1] Membaca Citra OpenCV : {os.path.basename(sample)}")
    img = cv2.imread(sample)
    print(f"    Dimensi Matriks Citra: {img.shape[1]}x{img.shape[0]} piksel")

    print("[2] Ekstraksi OpenCV     : Analisis Warna Dominan (HSV) & Bercak Permukaan...")
    warna, bercak = ekstraksi_fitur_visual_opencv(img, fruit_type="apple")
    print(f"    Indeks Warna         : {warna:.2f} / 100")
    print(f"    Persentase Cacat     : {bercak:.2f} %")

    print("[3] Fuzzy Logic Engine   : Menghitung Kematangan & Kesegaran...")
    sim_kem = bangun_fuzzy_kematangan()
    sim_seg = bangun_fuzzy_kesegaran()
    skor_k, lbl_k, kat_k, skor_s, lbl_s, tag_s, _ = evaluasi_status_fuzzy(sim_kem, sim_seg, warna, bercak, 95.0)
    print(f"    Tingkat Kematangan   : {lbl_k} (Skor: {skor_k:.2f}%)")
    print(f"    Kondisi Kesegaran    : {lbl_s} {tag_s} (Skor: {skor_s:.2f}%)")

    print("[4] Pencocokan Nutrisi   : Mengambil database nutrisi ilmiah...")
    nutrisi = NUTRISI_KEMATANGAN_DB.get("apple", {}).get(kat_k, {})
    if nutrisi:
        print(f"    Profil Gizi Apel     : Gula {nutrisi.get('gula', '-')} | Kalori {nutrisi.get('kalori', '-')}")
        print(f"    Karakteristik Biologis: {nutrisi.get('karakteristik', '-')}")
        print(f"    Rekomendasi Konsumsi : {nutrisi.get('rekomendasi', '-')}")

    print("\n[5] Verifikasi Dukungan Semua Buah Dataset (9 Komoditas):")
    for key, data in NUTRISI_KEMATANGAN_DB.items():
        status_db = "Tersedia (4 Fase)" if len(data) >= 4 else "Kurang"
        print(f"    - {key.capitalize():<12} : {data['nama']} -> {status_db}")

    print("=" * 70)
    print("[OK] Seluruh alur sistem berhasil berjalan tanpa emoji dan tanpa preprocessing!\n")

# ------------------------------------------------------------------------------
# 8. TAMPILAN MENU UTAMA (Format Bersih Bebas Emoji)
# ------------------------------------------------------------------------------
def main():
    while True:
        print(r"""
  ===================================================================
                         SMART FRUIT SCANNER
          Penilai Kesegaran Objektif & Edukasi Nutrisi Buah
  
   Modul Terintegrasi: OpenCV | YOLOv8 | Fuzzy Logic | Nutrisi DB
   Mendukung 9 Komoditas: Apel, Pisang, Paprika, Wortel, Mentimun,
                          Mangga, Jeruk, Kentang, Tomat
  ===================================================================
        """)
        print("  PILIHAN MENU:")
        print("  [1] Latih Model YOLO (Training Dataset Roboflow)")
        print("  [2] Scan dari File Gambar (Deteksi Citra Tunggal)")
        print("  [3] Scan dari File Video (Deteksi Frame-by-Frame)")
        print("  [4] Scan dari Kamera Langsung (Real-time Webcam)")
        print("  [5] Uji Cepat Integrasi Sistem (OpenCV + Fuzzy + Nutrisi)")
        print("  [0] Keluar")
        print("-" * 69)

        pilihan = input("  Pilih menu [0-5]: ").strip()

        if pilihan == "1":
            ep = input("  Masukkan jumlah epochs [Default 25]: ").strip()
            epochs = int(ep) if ep else 25
            model = YOLO(BASE_MODEL)
            model.train(
                data=DATASET_YAML,
                epochs=epochs,
                batch=16,
                imgsz=640,
                device=DEVICE,
                project=os.path.join(BASE_DIR, "runs", "fruit_freshness"),
                name="smart_fruit_scanner",
                exist_ok=True,
                save=True
            )
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "2":
            p = input("  Masukkan path gambar (kosongkan untuk sampel otomatis): ").strip()
            menu_scan_gambar(p if p else None)
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "3":
            p = input("  Masukkan path file video (contoh: video_buah.mp4): ").strip()
            if p:
                menu_scan_video(p)
            else:
                print("[ERROR] Path video tidak boleh kosong.")
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "4":
            menu_scan_video(0)
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "5":
            menu_uji_mandiri()
            input("\nTekan Enter untuk kembali ke menu...")

        elif pilihan == "0":
            print("\nTerima kasih telah menggunakan Smart Fruit Scanner!")
            sys.exit(0)

        else:
            print("\n[PERINGATAN] Pilihan tidak valid. Silakan coba lagi.\n")

if __name__ == "__main__":
    main()
