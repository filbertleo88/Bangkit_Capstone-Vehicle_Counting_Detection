# Introduction

Naiknya populasi dan transmigrasi dari desa ke kota, mengakibatkan pertumbuhan penggunaan kendaraan, khususnya kendaraan roda empat seperti mobil bus dan juga truk sebagai sarana transportasi yang umum digunakan oleh masyarakat. Pembangunan tol yang semakin marak juga perlu diimbangi dengan pengaturan lalulintas yang lebih baik juga.

Salah satu masalah yang sering ditemui dalam perjalanan menggunakan kendaraan pribadi dan umum melalui jalan tol adalah kepadatan jalan yang tinggi, khususnya pada keadaan arus mudik dari desa ke kota dan juga sebaliknya pada masa-masa libur panjang seperti lebaran dan juga akhir dan awal tahun. Untuk membantu menyelesaikan permasalahan tersebut, penerapan Intelligent Traffic System (ITS) pada ruas jalan bisa menjadi pertimbangan utama. Apalagi dengan adanya perkembangan IoT dan juga machine learning, ITS juga bisa semakin dikembangkan untuk menyelesaikan permasalahan yang lebih spesifik.

Salah satu cabang studi pada machine learning adalah computer vision, penerapan computer vision bisa dilakukan untuk mendeteksi dan menghitung kendaraan yang lewat pada jalan raya. Dalam pendeteksian kendaraan, metode yang umum digunakan adalah pengurangan latar belakang, perbedaan frame, aliran optik, dan pendeteksian objek dengan pembelajaran mendalam. Tiga metode pertama mendeteksi kendaraan melalui fitur yang diekstraksi secara manual, yang relatif sederhana, tetapi mereka juga memiliki beberapa keterbatasan dalam hal akurasi atau ketahanan. Alih-alih mengekstraksi fitur secara manual, metode deep learning mensimulasikan pemrosesan informasi otak manusia dan memungkinkan jaringan yang dibangun untuk melakukan ekstraksi fitur secara otomatis dengan melatih dataset beranotasi yang besar. Namun, metode ini bergantung pada kumpulan data pelatihan yang besar dan sulit untuk diterapkan pada berbagai skenario video lalu lintas. Transfer learning dapat dikombinasikan dengan deep learning untuk membangun model tugas target berdasarkan tugas sumber, tetapi menggabungkan transfer learning dengan deep learning tanpa adanya data beranotasi masih menjadi arah penelitian yang penting untuk dipelajari.

# Framework

![Framework]()

Dalam mengembangkan model penghitung kendaraan secara otomatis menggunakan computer vision, kami membagi framework model menjadi tiga fase seperti pada gambar di atas. Dalam fase pertama yaitu dataset building, kami menggunakan gambar teranotasi yang didalamnya terdapat kendaraan pada open dataset untuk menghindari terlalu lama menghabiskan waktu dalam melakukan pelabelan data. Selain itu, ada tambahan dataset yang kami ekstrak dari sebuah video yang kami ambil beberapa frame di dalamnya untuk pelatihan model tambahan dan pengujian model serta validasi. Pembagian dataset untuk model yang kami kembangkan yaitu training sebesar 65%, validation 24%, dan testing 11% dari total gambar sebanyak4000 gambar.

# Vehicle Detection Performance

Algortima pendeteksian kendaraan yang digunakan adalah model YOLOV8n yang dikembangkan oleh ultralytic. Berdasarkan hasil benchmark yang dilakukan untuk masing-masing model deteksi YOLOV8 berikut merupakan perbandingan performa YOLOV8 dibandingkan YOLO versi lainnya.

![Yolov8 Performance]()

Pemilihan model menggunakan YOLOV8n dilakukan karena performanya yang jauh melebihi pilihan alternatif model pertama yaitu YOLOV5n. pemilihan YOLOV8n juga mempertimbangkan mobilitas pilihan alternatif model. YOLOV8n merupakan model YOLOV8 nano atau model terkecil dan ter-ringan dari seluruh model YOLOV8.

![Model Training]()

Setelah dilakukan evaluasi terhadap performa model YOLOV8n yang digunakan dalam deteksi kendaraan. Model tersebut menghasilkan nilai mAP50 sebesar 0,464. Nilai ini menunjukkan tingkat akurasi dan kecepatan yang cukup baik dalam melakukan deteksi kendaraan. Metrik evaluasi yang digunakan adalah mAP50 (Mean Average Precision at 50) yang menggabungkan presisi dan recall pada set data pengujian untuk mendapatkan rata-rata presisi pada tingkat recall 50%. Dalam kasus ini, nilai mAP50 sebesar 0,464 menandakan bahwa model mampu mengidentifikasi sebagian besar kendaraan dengan akurasi yang dapat diterima.

Meskipun nilai mAP50 sebesar 0,464 belum mencapai tingkat akurasi yang tinggi, disimpulkan bahwa nilai tersebut dapat diterima dalam konteks penggunaan model ini. Model ini digunakan khusus untuk deteksi kendaraan dan melakukan penghitungan terhadap jumlah kendaraan yang melewati titik tertentu, sehingga tidak memerlukan tingkat akurasi mAP50 yang sangat tinggi. Tujuan penggunaan model telah dipertimbangkan dan nilai mAP50 tersebut dianggap memadai untuk mencapai tujuan yang ditetapkan. Dalam konteks ini, kesalahan deteksi yang mungkin terjadi dianggap tidak signifikan dan dapat ditoleransi dalam analisis jumlah kendaraan.

Namun, tetap perlu diperhatikan bahwa semakin tinggi nilai mAP50, semakin akurat dan dapat diandalkan deteksi objek yang dihasilkan oleh model. Apabila diperlukan tingkat akurasi yang lebih tinggi atau model ini digunakan dalam aplikasi yang lebih kritis, perlu dilakukan upaya untuk meningkatkan performa model guna mencapai nilai mAP50 yang lebih tinggi. Dalam penelitian selanjutnya, akan dipertimbangkan strategi seperti penambahan data pelatihan yang lebih bervariasi atau penyesuaian arsitektur model. Kesimpulannya, performa model YOLOV8n dengan nilai mAP50 sebesar 0,464 telah memberikan hasil yang dapat diterima untuk tujuan deteksi kendaraan dan penghitungan jumlahnya dalam konteks yang telah dijelaskan.

![Confusion Matrix Normalized]()

# Vehicle Counting

Kami menggunakan PyCharm atau Visual Studio Code sebagai text editor dalam pembuatan program python ini. Program ini berjalan di mesin lokal komputer dimana program ini akan menjalankan proses pendeteksian kendaraan sekaligus perhitungan kendaraan yang melewati batas tertentu yang kemudian data dari jumlah kendaraan yang masuk, keluar dan kapasitas rest area tersebut akan disimpan ke dalam bentuk real-time database menggunakan Realtime Firebase, dan juga dalam bentuk CSV berupa laporan kendaraan yang terdeteksi.

Proses pembuatan program Python dimulai dengan menggunakan model YOLO (You Only Look Once) untuk mendeteksi objek dalam sebuah gambar atau video. Model YOLO adalah sebuah algoritma deep learning yang efisien dalam mendeteksi objek secara real-time. Pertama, kita mengimpor library yang diperlukan, seperti OpenCV untuk membaca gambar atau video, dan library lain yang mendukung implementasi model YOLO.

Setelah mengimpor library yang diperlukan, langkah selanjutnya adalah memuat model YOLO beserta konfigurasinya. Model ini sudah dilatih sebelumnya menggunakan data pelatihan yang mencakup berbagai objek yang ingin dideteksi. Dalam kasus ini, kita akan fokus pada pendeteksian kendaraan di rest area. Model YOLO menghasilkan kotak pembatas (bounding box) dan label untuk setiap objek yang terdeteksi.

Setelah mendeteksi objek, langkah berikutnya adalah menghitung titik tengah dari setiap objek yang berpotongan dengan garis batas yang telah ditentukan. Garis batas ini biasanya ditarik di area masuk atau keluar rest area untuk menghitung kendaraan yang masuk atau keluar. Dengan menggunakan koordinat titik tengah objek dan persamaan garis, kita dapat menentukan apakah objek tersebut berada di dalam atau di luar rest area.

Selanjutnya, data dari jumlah kendaraan yang dihitung akan diolah untuk mendapatkan informasi kendaraan masuk, keluar, dan kapasitas rest area. Jumlah kendaraan masuk akan dihitung berdasarkan objek yang bergerak dari luar ke dalam garis limit, sedangkan jumlah kendaraan keluar dihitung berdasarkan objek yang bergerak dari dalam ke luar garis limit. Kapasitas rest area dapat dihitung dengan membandingkan jumlah kendaraan masuk dan keluar dengan kapasitas maksimal yang telah ditentukan.

Setelah mengolah data tersebut, langkah terakhir adalah menyimpannya ke dalam bentuk file CSV dan Realtime Database di Firebase. File CSV berfungsi sebagai penyimpanan data yang dapat diakses dan diolah nanti. Sementara itu, Realtime Database di Firebase memungkinkan kita untuk menyimpan data secara real-time dan mengaksesnya dari aplikasi mobile Android. Dengan menggunakan API Firebase, kita dapat mengirim dan mengambil data dari aplikasi Android ke database secara langsung.

Data yang telah disimpan ke dalam CSV dan Realtime Database dapat kemudian diolah lagi untuk dibuatkan grafik yang akan ditampilkan di dalam aplikasi mobile Android. Grafik ini akan memberikan visualisasi yang lebih mudah dipahami tentang jumlah kendaraan masuk, keluar, dan kapasitas rest area dalam rentang waktu tertentu.

![Python Program]()
![Realtime Database Output]()
