📌 1. Domain Proyek
Latar Belakang
Masalah dropout mahasiswa di perguruan tinggi merupakan isu penting yang berdampak luas, baik bagi institusi pendidikan maupun mahasiswa itu sendiri. Tingginya tingkat dropout dapat menyebabkan kerugian finansial bagi kampus, memengaruhi akreditasi dan reputasi institusi, serta menghambat perkembangan akademik dan karier mahasiswa di masa depan.

Menurut Nurmalitasari et al. (2023), keputusan mahasiswa untuk berhenti kuliah dipengaruhi oleh berbagai faktor seperti kondisi akademik, stres, kebiasaan belajar, keterlibatan dalam aktivitas sosial, kondisi keluarga, hingga kondisi mental. Dengan memanfaatkan pendekatan machine learning, institusi pendidikan dapat membangun sistem prediksi yang mampu mengidentifikasi mahasiswa dengan risiko dropout lebih awal, sehingga intervensi pencegahan dapat dilakukan secara tepat waktu.

Referensi:
Nurmalitasari, Awang Long, Z., & Mohd Noor, M. F. (2023). Factors Influencing Dropout Students in Higher Education. Education Research International, 2023, Article ID 7704142. https://doi.org/10.1155/2023/7704142

📌 2. Business Understanding
Problem Statements
Bagaimana cara memprediksi mahasiswa yang berisiko dropout berdasarkan data kebiasaan, performa akademik, kondisi sosial, dan faktor personal lainnya?

Goals
Membangun model machine learning yang mampu memprediksi risiko dropout mahasiswa dengan performa optimal, khususnya dalam hal recall dan F1-score.

Solution Statements

Mengimplementasikan beberapa algoritma klasifikasi seperti Logistic Regression, Random Forest, dan XGBoost untuk membandingkan performa model.

Melakukan hyperparameter tuning untuk meningkatkan performa model baseline.

Memilih model terbaik berdasarkan metrik F1-Score dan Recall, karena isu dropout biasanya menghadapi masalah data imbalance.

📌 3. Data Understanding
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


Exploratory Data Analysis (EDA):

Visualisasi distribusi fitur numerik & kategorikal.

Analisis korelasi antar fitur numerik.

Analisis distribusi label dropout_risk.

📌 4. Data Preparation
Langkah-langkah:

Menangani missing value dengan median (numerik) & modus (kategori).

Encoding data kategorikal dengan One Hot Encoding.

Normalisasi fitur numerik menggunakan MinMaxScaler.

Split data menjadi 80% train dan 20% test.

Mengatasi data imbalance menggunakan SMOTE.

Alasan:

Menghindari bias akibat missing value.

Menyamakan skala fitur untuk meningkatkan performa model.

Mengurangi risiko model bias akibat imbalance class.

📌 5. Modeling
Algoritma yang Digunakan:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Parameter yang Dicoba:

Logistic Regression: C (regularization strength)

Random Forest: n_estimators, max_depth

XGBoost: learning_rate, max_depth, n_estimators

Kelebihan & Kekurangan:

Algoritma	Kelebihan	Kekurangan
Logistic Regression	Interpretasi hasil mudah	Kurang cocok untuk data non-linear
Random Forest	Stabil, tangguh terhadap missing & outlier	Waktu komputasi tinggi saat tuning
XGBoost	Akurasi tinggi, efisien untuk data besar	Rawan overfitting jika tidak dituning

Improvement:

Hyperparameter tuning menggunakan GridSearchCV.

Oversampling menggunakan SMOTE.

Model Terbaik:
Model dengan metrik F1-Score dan Recall tertinggi pada data test, misal: XGBoost.

📌 6. Evaluation
Metrik Evaluasi yang Digunakan:

Accuracy: Persentase prediksi yang benar.

Precision: Proporsi prediksi dropout yang benar.

Recall: Proporsi dropout yang berhasil terdeteksi.

F1-Score: Harmonic mean dari precision dan recall.

Alasan Pemilihan Metrik:
Karena data dropout biasanya imbalance, Recall dan F1-Score lebih penting agar tidak banyak kasus dropout yang tidak terdeteksi.

Formula:

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

Hasil Evaluasi:
![image](https://github.com/user-attachments/assets/62094aa0-73a4-4221-9080-efc3e90da6aa)





