# Laporan Proyek Machine Learning - M Dai Bahtiar

## 1. Domain Proyek
Masalah dropout mahasiswa di perguruan tinggi merupakan isu penting yang berdampak luas, baik bagi institusi pendidikan maupun mahasiswa itu sendiri. Tingginya tingkat dropout dapat menyebabkan kerugian finansial bagi kampus, memengaruhi akreditasi dan reputasi institusi, serta menghambat perkembangan akademik dan karier mahasiswa di masa depan.

Menurut Nurmalitasari et al. (2023), keputusan mahasiswa untuk berhenti kuliah dipengaruhi oleh berbagai faktor seperti kondisi akademik, stres, kebiasaan belajar, keterlibatan dalam aktivitas sosial, kondisi keluarga, hingga kondisi mental. Dengan memanfaatkan pendekatan machine learning, institusi pendidikan dapat membangun sistem prediksi yang mampu mengidentifikasi mahasiswa dengan risiko dropout lebih awal, sehingga intervensi pencegahan dapat dilakukan secara tepat waktu.

Referensi:
Nurmalitasari, Awang Long, Z., & Mohd Noor, M. F. (2023). Factors Influencing Dropout Students in Higher Education. Education Research International, 2023, Article ID 7704142. https://doi.org/10.1155/2023/7704142

## 2. Business Understanding
## Problem Statement
Tingkat dropout mahasiswa menjadi salah satu permasalahan serius di institusi pendidikan tinggi. Dropout dapat disebabkan oleh berbagai faktor seperti kebiasaan belajar, kondisi ekonomi, motivasi, dan performa akademik mahasiswa. Tingginya angka dropout dapat berdampak pada citra institusi, efektivitas program pendidikan, serta kerugian finansial.

Institusi pendidikan perlu memiliki sistem prediksi risiko dropout untuk mengetahui mahasiswa yang berpotensi mengalami dropout lebih awal, sehingga dapat dilakukan intervensi preventif.

## Goals
Tujuan dari proyek ini adalah membangun model machine learning untuk mengklasifikasikan risiko dropout mahasiswa berdasarkan kebiasaan, kondisi, dan performa akademik mereka. Model ini diharapkan dapat:
- Mengidentifikasi mahasiswa yang berisiko tinggi dropout.
- Memberikan insight kepada pihak kampus untuk melakukan tindakan preventif.
- Meningkatkan retensi mahasiswa.

Target yang diukur:
- Akurasi model minimal 80%
- Model memiliki precision dan recall seimbang di atas 75% pada kelas risiko tinggi.

Solution Statement (Opsional)

Model klasifikasi risiko dropout ini nantinya dapat diimplementasikan dalam sistem monitoring kampus, sehingga mahasiswa dengan risiko tinggi dapat diberikan program pembinaan atau konseling khusus.



## 3. Data Understanding
Informasi Dataset
Jumlah Data: Dataset ini terdiri dari 80.000 baris dan 31 kolom sebelum dilakukan penghapusan kolom student_id.

Kondisi Data:

Missing Values: Setelah dilakukan pengecekan menggunakan df.isnull().sum(), ditemukan bahwa tidak terdapat missing value di dataset.

Data Duplikat: Hasil pengecekan df.duplicated().sum() menunjukkan tidak ada data duplikat.

Sumber Data: [Enhanced Student Habits Performance Dataset - Kaggle](https://www.kaggle.com/datasets/aryan208/student-habits-and-academic-performance-dataset)

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
1. Struktur Data

Menggunakan df.info() diketahui bahwa dataset terdiri dari 80.000 data dengan berbagai tipe data numerik dan kategorikal, tanpa missing value.

![image](https://github.com/user-attachments/assets/22fd6d9a-cb75-405e-88b5-7e2451c44efe)

2. Penghapusan Kolom student_id

Kolom ini dihapus karena bersifat unik dan tidak memiliki pengaruh terhadap model prediksi.

3. Pengecekan Missing Value
Ditemukan bahwa tidak ada missing value di dataset.

![image](https://github.com/user-attachments/assets/7415e9a7-2d02-4752-8f3b-b97c06d1f4e5)
![image](https://github.com/user-attachments/assets/45989ce2-a467-4262-ba05-21042bda1386)

4. Pengecekan Data Duplikat
   
Hasil pengecekan menunjukkan tidak ada baris data yang duplikat.

![image](https://github.com/user-attachments/assets/61275419-9b00-46d5-b5b8-e4c90e62c76f)


5. Distribusi Fitur Numerik

Distribusi tiap fitur numerik divisualisasikan menggunakan histogram dengan sns.histplot().
Insight:
- Beberapa fitur seperti study_hours_per_day, sleep_hours, dan exam_score menunjukkan distribusi yang relatif normal.
- Fitur exam_anxiety_score dan mental_health_rating memiliki sebaran yang cukup merata, sedangkan exercise_frequency dan social_activity menunjukkan pola cenderung menumpuk di nilai rendah.
![image](https://github.com/user-attachments/assets/c742b647-331d-4de9-9194-fd00d79768e4)
![image](https://github.com/user-attachments/assets/d265b6a7-85da-4064-a1ae-1e874f9f4313)


6. Hubungan Antar Fitur Numerik

Dengan pairplot menggunakan sns.pairplot(), terlihat pola hubungan antar beberapa fitur.
Insight:
- Terdapat hubungan positif antara study_hours_per_day dan exam_score.
- Tidak ditemukan hubungan korelasi yang terlalu kuat antar fitur numerik secara langsung.

### visualisasi pairplot dari semua fitur numerik dalam dataset
![image](https://github.com/user-attachments/assets/f96c60c9-6b07-4c6c-b1af-13b3c29db697)
![image](https://github.com/user-attachments/assets/f4583b1f-647b-45ce-81e1-b5dd301cecf8)

7. Statistik Deskriptif

Menggunakan df.describe(), diperoleh informasi tentang sebaran nilai mean, standar deviasi, minimum, maksimum, dan quartile tiap fitur numerik.
Insight:
- Rata-rata study_hours_per_day berada di sekitar 4.9 jam per hari.
- Rata-rata exam_score mahasiswa sekitar 70 dengan minimum 30 dan maksimum 100.
![image](https://github.com/user-attachments/assets/53cd4310-c32b-46b2-84b4-4888be1a8147)
![image](https://github.com/user-attachments/assets/ad7e9621-7082-401d-8c1b-482387202c10)

8. Heatmap Korelasi

Korelasi antar fitur numerik divisualisasikan menggunakan heatmap sns.heatmap().
Insight:
- Korelasi paling tinggi terlihat antara study_hours_per_day dengan exam_score (sekitar 0.7).
- Sebagian besar fitur numerik lainnya memiliki korelasi rendah satu sama lain.

Analisis korelasi antar fitur numerik.
![image](https://github.com/user-attachments/assets/bda834ee-5c90-4207-a5e7-8e82417b588b)

9. Identifikasi Fitur Kategorikal

Menggunakan df.select_dtypes(include=[object]), ditemukan 12 fitur kategorikal.

Fitur kategorikal: ['gender', 'major', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation', 'dropout_risk', 'study_environment', 'access_to_tutoring', 'family_income_range', 'learning_style']

10. Plot Distribusi Fitur Kategorikal

Distribusi tiap fitur kategorikal divisualisasikan menggunakan countplot.
Insight:
- Mayoritas mahasiswa berasal dari jurusan tertentu (jurusan Business dan Engineering paling dominan).
- Sebagian besar mahasiswa tidak memiliki pekerjaan paruh waktu.
- Distribusi dropout_risk terlihat tidak seimbang, dengan mayoritas berada di kategori 'Low'.
### Visualisasi distribusi fitur kategorikal
![image](https://github.com/user-attachments/assets/f8e60100-ab08-41f5-ada1-dded545f5a22)
![image](https://github.com/user-attachments/assets/c03b46c4-5d8c-4442-b94e-accb41480e29)
![image](https://github.com/user-attachments/assets/20cc1889-7691-4be4-b6a8-f80767074421)
![image](https://github.com/user-attachments/assets/7035b5bb-3591-4ea8-a5ff-a0b56eee526b)
![image](https://github.com/user-attachments/assets/c62cce16-9353-4d50-a1ff-14bb186a27e0)
![image](https://github.com/user-attachments/assets/3b06bae3-dca3-40d7-a919-63245cf532b7)


## 4. Data Preparation
Tahap Data Preparation bertujuan untuk mempersiapkan data sebelum proses pemodelan machine learning. Beberapa langkah yang dilakukan dalam tahap ini meliputi visualisasi dan penanganan outlier, encoding data kategorikal, normalisasi data numerik, serta pemisahan data menjadi data latih dan data uji. Berikut tahapan detail yang dilakukan:

1. Visualisasi Outlier (Box Plot)

Langkah awal dilakukan visualisasi outlier menggunakan boxplot untuk setiap kolom numerik. Visualisasi ini bertujuan untuk mendeteksi apakah terdapat outlier dalam fitur numerik dataset.

Boxplot menampilkan nilai median, kuartil pertama (Q1), kuartil ketiga (Q3), serta outlier yang terlihat sebagai titik-titik di luar whisker.

Visualisasi ini membantu untuk mengidentifikasi fitur mana saja yang memiliki nilai ekstrem di luar rentang normal data.
![image](https://github.com/user-attachments/assets/631cf658-4d14-4396-9619-51f5d6e8d792)
![image](https://github.com/user-attachments/assets/56017204-bd30-4ef7-9dfc-da75e0655b0c)


2. Menghapus Outlier dengan Metode IQR (Interquartile Range)

Setelah melakukan visualisasi, data outlier dihapus menggunakan metode Interquartile Range (IQR).
Langkah-langkahnya:
- Menghitung Q1 dan Q3 untuk setiap fitur numerik.
- Menghitung nilai IQR (Q3 - Q1).
- Menentukan batas bawah dan batas atas outlier menggunakan rumus:
  - Lower Bound = Q1 - 1.5 × IQR
  - Upper Bound = Q3 + 1.5 × IQR
- Menghapus data yang berada di luar rentang tersebut.

Setelah proses ini, hanya data numerik yang berada dalam rentang wajar yang dipertahankan, lalu digabungkan kembali dengan data kategorikal.

![image](https://github.com/user-attachments/assets/0cd2ab7f-5062-4f86-a222-606014149f0e)


3. Visualisasi Ulang Boxplot Setelah Outlier Removal

Setelah proses pembersihan outlier, dilakukan kembali visualisasi boxplot untuk setiap fitur numerik.

Tujuan visualisasi ulang ini adalah untuk memastikan bahwa nilai-nilai ekstrem telah berhasil dihapus, dan melihat sebaran data numerik yang baru.

Hasil visualisasi menunjukkan distribusi yang lebih stabil tanpa adanya nilai ekstrem.
![image](https://github.com/user-attachments/assets/a93accc8-f8df-4c24-9f0b-a3731d9d3ea1)
![image](https://github.com/user-attachments/assets/243eb9c9-e5ca-4a08-ad67-25715ee08f79)



4. Encoding Data Kategorikal

Fitur kategorikal diubah menjadi nilai numerik menggunakan metode Label Encoding agar dapat diproses oleh algoritma machine learning.

Proses encoding dilakukan menggunakan LabelEncoder dari library sklearn.preprocessing.

Selain itu, setiap encoder untuk masing-masing kolom disimpan ke dalam dictionary label_encoders agar dapat digunakan kembali saat melakukan prediksi pada data baru.
![image](https://github.com/user-attachments/assets/0a2c9f12-3f10-47fe-8bd2-d16b7e2d9ade)
![image](https://github.com/user-attachments/assets/a2da224e-540c-4021-8657-047bbd552367)


5. Normalisasi Data Numerik

Langkah ini bertujuan untuk menstandarkan nilai-nilai fitur numerik agar memiliki rata-rata 0 dan standar deviasi 1, sehingga algoritma machine learning dapat melakukan proses pembelajaran secara optimal tanpa bias terhadap skala data yang berbeda.

Langkah-langkahnya:

  5.1 Inisialisasi Scaler

Digunakan StandardScaler dari library sklearn.preprocessing untuk melakukan normalisasi.
![image](https://github.com/user-attachments/assets/eee37454-f509-4837-a9e1-e14fb84ed58b)

  5.2 Memisahkan Fitur dan Target

Mengambil seluruh kolom di df_labeled kecuali kolom dropout_risk sebagai fitur.
![image](https://github.com/user-attachments/assets/46c4482c-ff63-4942-9130-b9e24302d176)

  5.3 Melakukan Normalisasi (Standarisasi)

Normalisasi seluruh fitur numerik menggunakan StandardScaler, sehingga setiap kolom memiliki rata-rata 0 dan standar deviasi 1. 

Hasilnya disimpan ke dalam variabel scaled_features.

![image](https://github.com/user-attachments/assets/da26bf49-c3fb-4a33-87ac-f21568e74460)

  5.4 Membuat DataFrame Baru dari Hasil Normalisasi

Menyusun kembali hasil normalisasi menjadi sebuah DataFrame dengan nama kolom yang sama seperti features.
![image](https://github.com/user-attachments/assets/16a1a9c2-7945-497f-9dd4-b1e15ad18492)

  5.5 Menambahkan Kembali Kolom Target ke DataFrame Hasil Normalisasi

Kolom dropout_risk dari df_labeled ditambahkan kembali ke dalam scaled_df untuk melengkapi data hasil normalisasi.
![image](https://github.com/user-attachments/assets/66636975-e801-4544-b57c-38f1cb20a34c)


  5.6 Melihat Hasil Normalisasi

Menampilkan 5 baris pertama dari DataFrame scaled_df untuk memastikan bahwa fitur numerik sudah ternormalisasi dan target sudah tergabung dengan benar.

![image](https://github.com/user-attachments/assets/8778866c-6780-4f25-a6cd-b72992c79b92)

![image](https://github.com/user-attachments/assets/f307b599-ff3b-4c8e-87cc-d1a5e3783394)

![image](https://github.com/user-attachments/assets/81be2460-b9f7-4283-b47a-f7580ab8460e)

6. Pemisahan Fitur dan Target
Data yang telah dinormalisasi kemudian dipisahkan kembali menjadi:
- Fitur (X) → seluruh kolom kecuali dropout_risk.
- Target (y) → kolom dropout_risk.
  
![image](https://github.com/user-attachments/assets/c84fda0d-0d41-4453-b112-e61e5e93ddd7)

Proses ini bertujuan untuk mempersiapkan data sebelum dilakukan pembagian menjadi data latih dan data uji.

7. Split Data Menjadi Training Set dan Test Set

Dataset kemudian dibagi menjadi 80% data latih dan 20% data uji. Data latih digunakan untuk melatih model machine learning, sedangkan data uji digunakan untuk mengukur performa model terhadap data yang belum pernah dilihat sebelumnya. Pembagian ini penting agar hasil evaluasi bisa merepresentasikan kemampuan model dalam menghadapi data baru.

- Melatih model dengan data training.
- Menguji performa model menggunakan data yang belum pernah dilihat sebelumnya.
![image](https://github.com/user-attachments/assets/98ddd039-46e0-4e61-95d7-dd698c9337cc)

Pembagian ini penting agar hasil evaluasi dapat merepresentasikan kemampuan model saat menghadapi data baru.

8. Menampilkan Ukuran Data Training dan Test

Langkah terakhir pada tahap ini adalah menampilkan jumlah data pada training set dan test set untuk memastikan proporsi pembagian data sudah sesuai, yakni 80% data latih dan 20% data uji.

![image](https://github.com/user-attachments/assets/0a435a37-6a41-411a-826f-2f4f4609b854)


## 6. Modelling

6.1 Pemilihan Algoritma
Dalam proyek ini, dilakukan pemodelan menggunakan tiga algoritma machine learning populer yang cocok untuk kasus klasifikasi biner, yaitu:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM) dengan kernel radial basis function (RBF)

Ketiga model ini dipilih karena karakteristiknya yang saling melengkapi dalam menghadapi data tabular dengan kemungkinan distribusi kelas yang tidak seimbang.


6.2 Penjelasan Cara Kerja Setiap Model

1. Logistic Regression
Logistic Regression adalah algoritma klasifikasi linier yang memodelkan probabilitas suatu data termasuk ke dalam kelas tertentu menggunakan fungsi logistik (sigmoid).

Cara kerja:

Model menghitung weighted sum dari fitur input, kemudian menerapkan fungsi sigmoid untuk menghasilkan output berupa nilai probabilitas antara 0 dan 1. Nilai ini kemudian dibandingkan dengan threshold (biasanya 0.5) untuk menentukan kelas prediksi.

Parameter yang digunakan:
- random_state=42 → untuk memastikan hasil replikasi yang konsisten.

2. Random Forest Classifier
Random Forest adalah algoritma ensemble berbasis decision tree. Model ini membangun banyak decision tree secara acak pada subset data yang berbeda, kemudian menggabungkan hasil prediksi masing-masing pohon dengan metode voting mayoritas.

Cara kerja:
- Data training di-bootstrap (diacak dengan pengembalian) untuk setiap pohon.
- Setiap node pada pohon menggunakan subset acak dari fitur untuk menentukan split terbaik.
- Prediksi akhir diambil berdasarkan mayoritas vote dari semua pohon.

Kelebihan: Tahan terhadap overfitting dan dapat menangani fitur non-linear.

Parameter yang digunakan:
- random_state=42 → untuk konsistensi hasil.

3. Support Vector Machine (SVM) dengan Kernel RBF
SVM bekerja dengan mencari hyperplane yang memisahkan dua kelas dengan margin maksimum. Kernel RBF digunakan untuk memetakan data ke dimensi yang lebih tinggi agar lebih mudah dipisahkan.

Cara kerja:
- Data dipetakan ke ruang berdimensi lebih tinggi.
- Mencari hyperplane dengan margin terlebar yang memisahkan kelas dropout dan non-dropout.
- Hanya support vectors (data di dekat margin) yang memengaruhi posisi hyperplane.

Parameter yang digunakan:
- kernel='rbf' → untuk menangani data non-linear.
- random_state=42 → untuk menjaga konsistensi hasil.

6.3 Inisialisasi dan Training Model
Ketiga model diinisialisasi dalam sebuah dictionary models untuk memudahkan proses training dan evaluasi secara berurutan.

![image](https://github.com/user-attachments/assets/83a79679-4da1-4886-bfdd-85f4dccc5cfc)

## 7. Evaluasi

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








