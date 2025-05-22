# Laporan Proyek Machine Learning - M Dai Bahtiar

## 1. Domain Proyek
Masalah dropout mahasiswa di perguruan tinggi merupakan isu penting yang berdampak luas, baik bagi institusi pendidikan maupun mahasiswa itu sendiri. Tingginya tingkat dropout dapat menyebabkan kerugian finansial bagi kampus, memengaruhi akreditasi dan reputasi institusi, serta menghambat perkembangan akademik dan karier mahasiswa di masa depan.

Menurut Nurmalitasari et al. (2023), keputusan mahasiswa untuk berhenti kuliah dipengaruhi oleh berbagai faktor seperti kondisi akademik, stres, kebiasaan belajar, keterlibatan dalam aktivitas sosial, kondisi keluarga, hingga kondisi mental. Dengan memanfaatkan pendekatan machine learning, institusi pendidikan dapat membangun sistem prediksi yang mampu mengidentifikasi mahasiswa dengan risiko dropout lebih awal, sehingga intervensi pencegahan dapat dilakukan secara tepat waktu.

Referensi:
Nurmalitasari, Awang Long, Z., & Mohd Noor, M. F. (2023). Factors Influencing Dropout Students in Higher Education. Education Research International, 2023, Article ID 7704142. https://doi.org/10.1155/2023/7704142

## 2. Business Understanding
### Problem Statements
Bagaimana cara memprediksi mahasiswa yang berisiko dropout berdasarkan data kebiasaan, performa akademik, kondisi sosial, dan faktor personal lainnya?

### Goals
Membangun model machine learning yang mampu memprediksi risiko dropout mahasiswa dengan performa optimal, khususnya dalam hal recall dan F1-score.

### Solution Statements
- Mengimplementasikan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan XGBoost untuk membandingkan performa model.
- Melakukan hyperparameter tuning untuk meningkatkan performa model baseline.
- Memilih model terbaik berdasarkan metrik F1-Score dan Recall, karena isu dropout biasanya menghadapi masalah data imbalance.

## 3. Data Understanding
Informasi Dataset

Jumlah Data: 80.000 data mahasiswa

Kondisi Data: Terdapat missing value dan distribusi label dropout_risk yang imbalance.

Sumber Dataset:
https://www.kaggle.com/datasets/nishantbhujel/enhanced-student-habits-performance-dataset

Variabel / Fitur Data:
| Nama Fitur                      | Deskripsi                                                 |
| :------------------------------ | :-------------------------------------------------------- |
| `student_id`                    | ID unik untuk setiap mahasiswa.                           |
| `age`                           | Usia mahasiswa (dalam tahun).                             |
| `gender`                        | Jenis kelamin mahasiswa (Male/Female/Other).              |
| `major`                         | Program studi atau jurusan mahasiswa.                     |
| `study_hours_per_day`           | Rata-rata jam belajar mahasiswa per hari.                 |
| `social_media_hours`            | Rata-rata jam penggunaan media sosial per hari.           |
| `netflix_hours`                 | Rata-rata jam menonton Netflix per hari.                  |
| `part_time_job`                 | Status mahasiswa memiliki pekerjaan paruh waktu (Yes/No). |
| `attendance_percentage`         | Persentase kehadiran mahasiswa di kelas.                  |
| `sleep_hours`                   | Rata-rata jam tidur per hari.                             |
| `diet_quality`                  | Kualitas pola makan mahasiswa (Poor/Fair/Good).           |
| `exercise_frequency`            | Frekuensi olahraga per minggu (0-7 kali).                 |
| `parental_education_level`      | Tingkat pendidikan tertinggi orang tua mahasiswa.         |
| `internet_quality`              | Kualitas koneksi internet mahasiswa (Poor/Fair/Good).     |
| `mental_health_rating`          | Self-assessment kondisi mental (1-10).                    |
| `extracurricular_participation` | Keikutsertaan dalam kegiatan ekstrakurikuler (Yes/No).    |
| `previous_gpa`                  | Nilai IPK semester sebelumnya.                            |
| `semester`                      | Semester saat data diambil (1-8).                         |
| `stress_level`                  | Tingkat stres mahasiswa (Low/Medium/High).                |
| `dropout_risk`                  | Label target, risiko dropout (Yes/No).                    |
| `social_activity`               | Frekuensi kegiatan sosial per minggu.                     |
| `screen_time`                   | Rata-rata total waktu menatap layar per hari (jam).       |
| `study_environment`             | Kondisi lingkungan belajar (Poor/Average/Good).           |
| `access_to_tutoring`            | Akses ke bimbingan belajar (Yes/No).                      |
| `family_income_range`           | Rentang pendapatan keluarga (Low/Medium/High).            |
| `parental_support_level`        | Tingkat dukungan orang tua (Low/Medium/High).             |
| `motivation_level`              | Tingkat motivasi belajar (1-10).                          |
| `exam_anxiety_score`            | Tingkat kecemasan saat ujian (1-10).                      |
| `learning_style`                | Gaya belajar mahasiswa (Visual/Auditory/Kinesthetic).     |
| `time_management_score`         | Nilai kemampuan manajemen waktu (1-10).                   |
| `exam_score`                    | Nilai rata-rata ujian terkini.                            |


## 4. Exploratory Data Analysis (EDA):

### Visualisasi distribusi fitur numerik
![image](https://github.com/user-attachments/assets/940dc060-fcdd-446f-80e7-3708377bb276)
![image](https://github.com/user-attachments/assets/7f3180fb-6c71-4359-abb0-e21afe818f1d)

### visualisasi pairplot dari semua fitur numerik dalam dataset
![image](https://github.com/user-attachments/assets/f96c60c9-6b07-4c6c-b1af-13b3c29db697)
![image](https://github.com/user-attachments/assets/f4583b1f-647b-45ce-81e1-b5dd301cecf8)

### Visualisasi distribusi fitur kategorikal
![image](https://github.com/user-attachments/assets/f8e60100-ab08-41f5-ada1-dded545f5a22)
![image](https://github.com/user-attachments/assets/c03b46c4-5d8c-4442-b94e-accb41480e29)
![image](https://github.com/user-attachments/assets/20cc1889-7691-4be4-b6a8-f80767074421)
![image](https://github.com/user-attachments/assets/7035b5bb-3591-4ea8-a5ff-a0b56eee526b)
![image](https://github.com/user-attachments/assets/c62cce16-9353-4d50-a1ff-14bb186a27e0)
![image](https://github.com/user-attachments/assets/3b06bae3-dca3-40d7-a919-63245cf532b7)


Analisis korelasi antar fitur numerik.
![image](https://github.com/user-attachments/assets/bda834ee-5c90-4207-a5e7-8e82417b588b)

## 5. Data Preprocessing
Setelah menilai data dari tahap Data Understanding. Maka beberapa tahapan Data Preparation sebagai berikut:

1. Menangani Outliers - Melakukan filtering atau menggunakan metode IQR (Interquartile Range) untuk menghapus data di luar batas wajar. Outliers dapat menyebabkan model salah belajar karena mereka mendominasi pola data
2. Encoding Data Kategorikal - Mengubah fitur kategorikal menjadi angka dengan LabelEncoder agar bisa diproses model. Disimpan juga encoder-nya di label_encoders untuk dipakai saat prediksi data baru.
Hasil: Data kategorikal → angka. Data siap lanjut ke tahap scaling.
3. Standarisasi - Melakukan standarisasi menggunakan StandarScaler pada fitur numeric. Standarisasi penting untuk menghindari fitur dengan skala besar mendominasi perhitungan dalam algritma machine learnin
4. Splitting Dataset - Mengambil beberapa fitur penting yang dilihat dari correlation matrix, kemudian membagi dataset menjadi data latih (training) dan data uji (testing) dengan rasio umum seperti 80:20. SPlitting penting agar model bisa diuji keandalannya dalam memprediksi data baru yang belum dilihat

## 6. Modelling

Tiga algoritma digunakan dalam pemodelan:
- Logistic Regression: Model baseline yang cepat dan sederhana, cocok untuk interpretasi awal.
- Random Forest: Model ensemble yang tangguh terhadap overfitting dan menangani fitur non-linear.
- SVM: Cocok untuk data dengan margin yang jelas dan kelas yang tidak seimbang.

Setiap model dilatih menggunakan data training, kemudian dievaluasi pada data testing. Parameter default digunakan untuk baseline, tanpa hyperparameter tuning.

### Kelebihan & Kekurangan:
- Logistic Regression: Cepat, mudah diinterpretasi, tapi kurang bagus jika hubungan fitur tidak linear.
- Random Forest: Akurat, robust, tapi lebih kompleks dan butuh lebih banyak sumber daya.
- SVM: Akurat pada dataset kecil, tapi lambat pada dataset besar.

#### Evaluasi

Model dievaluasi menggunakan metrik:
- Accuracy: Persentase prediksi benar dari total prediksi.
- Precision: Seberapa akurat model dalam memprediksi dropout.
- Recall: Seberapa baik model menangkap semua kasus dropout.
- F1-score: Harmonik antara precision dan recall.

#### Hasil evaluasi:
![image](https://github.com/user-attachments/assets/6789c274-d953-4037-840c-9e649d65ae03)
#### Model Random Forest menunjukkan performa terbaik secara keseluruhan, khususnya pada recall dan F1-score, yang menjadi fokus utama proyek ini karena penting untuk mendeteksi sebanyak mungkin mahasiswa yang berisiko dropout.


#### Mengubah hasil evaluasi dari semua model menjadi DataFrame, lalu menampilkannya dalam bentuk tabel agar mudah dibandingkan.
![image](https://github.com/user-attachments/assets/a02e5879-7cc1-40f0-8c9c-ff25f8e361df)

#### Visualisasi Perbandingan Metrik
Menampilkan grafik batang untuk membandingkan nilai metrik evaluasi (Accuracy, Precision, Recall, F1-Score) dari setiap model secara visual.
![image](https://github.com/user-attachments/assets/5bb44245-bc78-4899-ac4c-1ded351f9d65)

#### Hasil Evaluasi Model
Setelah melatih ketiga model — Logistic Regression, Random Forest, dan Support Vector Machine (SVM) — kami melakukan evaluasi terhadap kinerja masing-masing menggunakan empat metrik utama: Accuracy, Precision, Recall, dan F1-Score. Berikut adalah hasil evaluasi lengkap:

- Model Logistic Regression memperoleh accuracy sebesar 99.30%, precision sebesar 99.28%, recall sebesar 99.30%, dan F1-score sebesar 99.29%.
- Model Random Forest menunjukkan performa sempurna dengan nilai 100% untuk keempat metrik: accuracy, precision, recall, dan F1-score.
- Model SVM juga menghasilkan performa sangat tinggi dengan accuracy sebesar 99.25%, precision sebesar 99.21%, recall sebesar 99.25%, dan F1-score sebesar 99.20%.

#### Interpretasi Hasil
Dari hasil evaluasi di atas, Random Forest menunjukkan performa terbaik di antara ketiga model pada semua metrik evaluasi utama. Model ini mencapai akurasi sempurna sebesar 100%, serta memiliki precision, recall, dan F1-score yang maksimal tanpa kesalahan klasifikasi sama sekali.

Model ini memiliki 100% precision, yang berarti seluruh kasus yang diprediksi sebagai dropout benar-benar kasus dropout. Selain itu, dengan 100% recall, Random Forest berhasil mendeteksi seluruh mahasiswa yang benar-benar berisiko dropout tanpa ada yang terlewat. F1-score sebesar 100% mengindikasikan keseimbangan sempurna antara precision dan recall.

Sementara itu, Logistic Regression dan SVM juga menunjukkan performa sangat baik, meskipun masih terdapat sedikit kesalahan prediksi. Logistic Regression menempati posisi kedua dengan F1-score sebesar 99.29%, diikuti oleh SVM dengan F1-score 99.20%.

### Confusion Matrix
#### Model Logistic Regression menghasilkan confusion matrix sebagai berikut:
- Benar Negatif (True Negative): 15.310 kasus mahasiswa yang tidak dropout diklasifikasikan dengan benar.
- Benar Positif (True Positive): 228 kasus mahasiswa dropout diklasifikasikan dengan benar.
- False Positive: 44 kasus mahasiswa yang tidak dropout namun salah diprediksi sebagai dropout.
- False Negative: 65 kasus mahasiswa dropout yang salah diprediksi sebagai tidak dropout.

#### Model Random Forest menghasilkan hasil yang sempurna:
- Benar Negatif (True Negative): 15.354 kasus benar.
- Benar Positif (True Positive): 293 kasus benar.
- False Positive: 0.
- False Negative: 0.

#### Model SVM menghasilkan:
- Benar Negatif (True Negative): 15.334 kasus benar.
- Benar Positif (True Positive): 196 kasus benar.
- False Positive: 20 kasus salah prediksi dropout.
- False Negative: 97 kasus dropout yang tidak terdeteksi.

## Kesimpulan
Berdasarkan hasil evaluasi, Random Forest dipilih sebagai model terbaik untuk kasus ini karena memiliki performa paling tinggi dan tanpa kesalahan klasifikasi sama sekali. Meskipun Logistic Regression dan SVM juga memberikan hasil yang sangat baik, Random Forest lebih unggul secara mutlak.

Model ini sangat direkomendasikan untuk digunakan dalam sistem prediksi risiko dropout mahasiswa karena mampu mendeteksi kasus dropout secara akurat dan konsisten. Ke depannya, model ini dapat tetap ditingkatkan melalui uji coba pada data baru dan penerapan teknik ensemble atau hyperparameter tuning guna memastikan performanya tetap optimal di berbagai kondisi.








