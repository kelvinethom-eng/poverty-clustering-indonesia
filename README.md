# 🗺️ Analisis Klaster Kesejahteraan Wilayah Indonesia

Mengelompokkan 38 provinsi Indonesia berdasarkan tingkat kemiskinan dan kualitas
hidup (IPM) menggunakan K-Means Clustering, untuk mengidentifikasi wilayah yang
perlu diprioritaskan dalam kebijakan pengentasan kemiskinan.

🔗 **[Coba Dashboard Interaktif](link-streamlit-kamu-setelah-deploy)**

## Latar Belakang
Kesenjangan pembangunan antarwilayah adalah tantangan besar di Indonesia.
Proyek ini menjawab tiga pertanyaan: apakah provinsi bisa dikelompokkan
berdasarkan kesejahteraan, wilayah mana yang paling butuh intervensi, dan
indikator apa yang paling membedakan antar wilayah.

## Data
- **Sumber:** [Badan Pusat Statistik (BPS)](https://www.bps.go.id), 2025
- **Indikator:** Persentase kemiskinan, garis kemiskinan, Indeks Pembangunan Manusia
- **Cakupan:** 38 provinsi

## Metodologi
1. Data cleaning & feature selection
2. Standardisasi fitur (StandardScaler)
3. Penentuan k optimal (Elbow Method + Silhouette Score)
4. K-Means Clustering (k=4)
5. Interpretasi & pelabelan klaster

## Temuan Utama

![Peta Klaster](/choropeth_cluster.png)

Empat klaster teridentifikasi: **Kritis**, **Perlu Prioritas**, **Berkembang**, dan
**Sejahtera & Maju**. Klaster "Kritis" — Papua Tengah dan Papua Pegunungan — memiliki
tingkat kemiskinan 28,3% (3x rata-rata nasional) dan IPM 20 poin di bawah rata-rata
nasional.

![Perbandingan Klaster](/cluster_comparison.png)

## Rekomendasi Kebijakan
Intervensi untuk klaster "Kritis" perlu berfokus pada infrastruktur dasar dan akses
pendidikan, sementara klaster "Perlu Prioritas" (NTT, Maluku, sebagian Papua) lebih
cocok dengan program bantuan sosial dan ekonomi jangka menengah.

## Keterbatasan
- Hanya menggunakan 3 indikator; belum memasukkan akses kesehatan/sanitasi
- 4 provinsi hasil pemekaran Papua (2022) digabung ke provinsi induk pada
  visualisasi peta karena keterbatasan data geografis

## Cara Menjalankan
```bash
git clone <repo-url>
cd poverty-clustering-indonesia
pip install -r requirements.txt

# Jalankan notebook
jupyter notebook notebooks/clustering.ipynb

# Atau jalankan dashboard interaktif
streamlit run app.py
```

## Tech Stack
Python · pandas · scikit-learn · GeoPandas · Streamlit · Plotly

## Struktur Repo
```
[isi dengan tree folder di atas]
```
