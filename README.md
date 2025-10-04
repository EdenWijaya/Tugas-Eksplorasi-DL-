# Benchmark Plain-34 vs ResNet-34 (IndonesianFood – 5 Kelas)

Repositori ini membandingkan performa:
- **Plain-34** (implementasi kustom, tanpa residual/skip connection), dan
- **ResNet-34** (implementasi `torchvision`),

serta dua **modifikasi Tahap 3** sesuai ketentuan tugas:
- **(F)** Dropout pada kepala klasifikasi, dan
- **(G)** penggantian optimizer ke **SGD + Nesterov**.

Dataset: **makanan Indonesia – 5 kelas** (`train.csv` + folder `train/`).

---

## 🔧 Konfigurasi Eksperimen

- **Jumlah kelas**: 5  
- **Transformasi**
  - Train: `Resize(224) → RandomHorizontalFlip(0.5) → RandomRotation(15) → ToTensor → Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`
  - Val  : `Resize(224) → ToTensor → Normalize(sama)`
- **Model**
  - Plain-34: `PlainBlock` + `PlainNet([3,4,6,3])` *(tanpa residual)*  
  - ResNet-34: `torchvision.models.resnet34(weights=None, num_classes=5)` *(residual bawaan)*
- **Optimizer**
  - Tahap 1–2 (baseline): **Adam** (`lr=1e-3`)
  - Tahap 3F: ResNet-34 + **Dropout(p=0.5)** di kepala (optimizer **Adam**)
  - Tahap 3G: ResNet-34 + **SGD+Nesterov** (`lr=1e-2`, `momentum=0.9`, `nesterov=True`)
- **Loss**: `CrossEntropyLoss`  
- **Batch size**: 32  
- **Epoch**: 10  
- **Device**: otomatis `cuda` bila tersedia, selain itu `cpu`

---

## 🧪 Hasil Utama (Val Acc Terbaik)

| Model                          | Val Acc Terbaik | Catatan                      |
|-------------------------------|----------------:|------------------------------|
| Plain-34                      | **0.5766**      | degradasi performa           |
| ResNet-34 (Adam)              | **0.7387**      | residual learning            |
| ResNet-34 + Dropout (F, Adam) | **0.6802**      | regularisasi kepala          |
| ResNet-34 (SGD+Nesterov) (G)  | **0.7523**      | terbaik pada 10 epoch        |

**Ringkasan:** Residual connection meningkatkan performa vs Plain-34; Dropout belum mengungguli baseline pada horizon 10 epoch; SGD+Nesterov memberi puncak akurasi sedikit di atas Adam.

> Opsional: simpan kurva akurasi validasi sebagai `Figure/compare_mods.png` untuk dimasukkan ke laporan.

---

## ▶️ Cara Menjalankan

### A) Google Colab
1. Upload `IF25-4041-dataset.zip` yang berisi `train.csv` dan folder `train/`.
2. Di sel awal jalankan:
   ```python
   !unzip IF25-4041-dataset.zip
3. Jalankan skrip per_6.py (Colab-friendly).

B) Lokal (Python)
1. Instal dependensi:
   ```python
   !pip install torch torchvision pandas matplotlib tqdm scikit-learn pillow
3. Jalankan:
   ```python
   python per_6.py


### Anggota Kelompok

- Eden Wijaya — 122140187
- Bayu Ega Ferdana — 122140129
- Intan Permata Sari — 122140207

### Eksperimen Arsitektur ResNet-34 — Deep Learning (IF25-40401) ###
