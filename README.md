# Benchmark Plain-34 vs ResNet-34 (IndonesianFood – 5 Kelas)

Repositori ini membandingkan performa **Plain-34** (tanpa residual/skip connection, implementasi kustom) dan **ResNet-34** (implementasi `torchvision`) pada dataset makanan Indonesia dengan **5 kelas**. Kode diambil dari skrip `per_6.py` (format Colab-friendly) yang:

- Membaca `train.csv` dan memuat gambar dari folder `train/`
- Melatih **Plain-34** dan **ResNet-34**
- Mencetak metrik tiap epoch (train/val loss & accuracy)
- Menampilkan plot **perbandingan akurasi validasi** (Plain vs ResNet)

---

Konfigurasi Eksperimen

- Jumlah kelas: 5
- Transformasi
Train: Resize(224), RandomHorizontalFlip(0.5), RandomRotation(15), Normalize([0.485,0.456,0.406]/[0.229,0.224,0.225])
Val : Resize(224), Normalize sama
- Model
Plain-34: PlainBlock + PlainNet([3,4,6,3]) (tanpa residual)
ResNet-34: torchvision.models.resnet34(weights=None, num_classes=5) (residual bawaan)
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Batch size: 32
- Epoch: 10
- Device: otomatis cuda bila tersedia, selain itu cpu

Cara Menjalankan 

A). Google Colab
1. Upload IF25-4041-dataset.zip yang berisi train.csv dan folder train/.
2. Jalankan sel di awal skrip:
        "!unzip IF25-4041-dataset.zip"
   
B). Lokal (Python)

- `pip install torch torchvision pandas matplotlib tqdm scikit-learn pillow`

- `python per_6.py`


Anggota Kelompok
- Eden Wijaya - 122140187
- Bayu Ega Ferdana - 122140129
- Intan Permata Sari - 122140207

Eksperimen Arsitektur ResNet-3
