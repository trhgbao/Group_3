# Data Mining Project: Plant Disease Classification & Preprocessing Pipeline

##  Danh sách thành viên

| STT | Họ và tên            |   MSSV   |
| --- | -------------------- | -------- |
| 1   | Bùi Anh Quân         | 23122017 |
| 2   | Trần Hoàng Gia Bảo   | 23122022 |
| 3   | Chung Tín Đạt        | 23122024 |
| 4   | Ngô Thị Thục Quyên   | 23120348 |
| 3   | Lăng Phú Quý         | 23120415 |



## Tổng quan dự án
- Tiền xử lý dữ liệu là làm sạch dữ liệu, là một quy trình kỹ thuật đòi hỏi hiểu biết sâu về thống kê,
thiết kế hệ thống và khả năng lập luận phân tích. Đồ án áp dụng các kỹ thuật tiền xử lý, khả năng giải thích
bằng các công cụ thống kê, so sánh có kiểm soát và thiết kế pipeline có thể sử dụng lại được cho các loại dữ liệu sau:
  • Dữ liệu ảnh: Biến đổi không gian, phân tích phân phối đặc trưng, augmentation có
  kiểm soát.
  • Dữ liệu bảng: Phát hiện ngoại lai, kiểm định thống kê, giảm chiều dữ liệu để trực quan
  hoá/ trích chọn đặc trưng.
  • Dữ liệu văn bản: Tiền xử lý theo đặc tính ngôn ngữ, biểu diễn ngữ nghĩa, phân tích
  tính thưa của dữ liệu.

### Xử lý văn bản dạng ảnh
- EDA và xây dựng quy trình tiền xử lý cho tập dữ liệu hình ảnh bệnh lý lá cây.
- Mục tiêu là làm sạch dữ liệu, trích xuất đặc trưng hình học/màu sắc.
### Xử lý văn bản dạng bảng
- Từ kết quả EDA, quyết định cách xử lý outlier và scaling, xây dựng pipeline tiền xử lý hợp lý
- Làm sạch và chuẩn hóa dữ liệu đầu vào, đưa dữ liệu về dạng phù hợp cho mô hình học máy, đánh giá định lượng các phương pháp tiền xử lý
### Xử lý văn bản dạng văn bản
- Xử lý dữ liệu văn bản với các text là các bài review sau khi xem phim, label là tích cực hoặc tiêu cực
- Là bài toán dự đoán cảm xúc dựa trên văn bản

---

## Cấu trúc thư mục
```bash
Group_3/
|-- README.md                                  # Tổng quan, hướng dẫn chạy, link tài nguyên
|-- requirements.txt                           # Danh sách thư viện và version cụ thể
|-- data/
|   |-- raw/                                   # Chứa dữ liệu gốc từ Kaggle
|   |   |-- lab_2.1_NewPlantDiseasesDataset    # Tập dữ liệu ảnh số
|   |   |   |-- class_folders
|   |   |   |-- images
|   |   |   |-- labels.csv
|   |   |-- adult.csv                          # Tập dữ liệu dạng bảng
|   |   |-- Movie_Reviews.csv                  # Tập dữ liệu văn bản
|   |-- processed/                             # Chứa dữ liệu sau tiền xử lý (CSV, ảnh biểu đồ, dữ liệu sau tiền xử lý)
|   |   |--  processed_image
|   |   |--  processed_tabular
|   |   |--  processed_text
|-- notebooks/
|   |-- 01_EDA_image.ipynb     
|   |-- 02_preprocessing_image.ipynb 
|   |-- 03_EDA_tabular.ipynb
|   |-- 04_preprocessing_tabular.ipynb
|   |-- 05_text_preprocessing.ipynb
|-- docs/
|   |-- Report.pdf                             # Báo cáo chi tiết kết quả thực nghiệm 
```

---

## Yêu cầu hệ thống & Cài đặt

### Cài đặt thư viện
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
## Windows:
```
venv\Scripts\activate
```
## macOS / Linux:
```
source venv/bin/activate
```
# Tải thư viện
```
pip install -r requirements.txt
```
## Các thư viên và phiên bản của chúng
--extra-index-url https://download.pytorch.org/whl/cpu
Core AI & NLP
torch (>=2.0.0)
sentence-transformers (>=2.2.2)
gensim (>=4.3.0)
nltk (>=3.8)
tokenizers (>=0.15.0)
Data Science & ML
numpy (<2.0.0)
pandas (>=2.0.0)
scipy (>=1.10.0)
scikit-learn (>=1.3.0)
statsmodels (>=0.14.0)
Visualization
matplotlib (>=3.7.0)
seaborn (>=0.12.0)
wordcloud (>=1.9.0)
missingno (>=0.5.0)
opencv-contrib-python (>=4.8.0)
Jupyter Notebook Support
ipykernel
ipython

---

## Dữ liệu 

### 1. Dữ liệu gốc (Raw Data)
#### **Link tải Dataset (Kaggle)**
*   **Dataset dạng ảnh:** https://www.kaggle.com/datasets/tranbao0105/lab-2-1-newplantdiseasesdataset
*   **Dataset dạng bảng:** https://www.kaggle.com/datasets/tranbao0105/lab-1-2
*   **Dataset dạng văn bản:** https://www.kaggle.com/datasets/tranbao0105/lab-2-3

#### **Tải về và giải nén vào thư mục**
*   **Dataset dạng ảnh:** `data/raw/lab_2.1_NewPlantDiseasesDataset/`. Đảm bảo file `labels.csv` nằm trong thư mục này.
*   **Dataset dạng bảng:** `data/raw/`. Đảm bảo file `adult.csv` nằm trong thư mục này.
*   **Dataset dạng văn bản:** `data/raw/`. Đảm bảo file `Movie_Reviews.csv` nằm trong thư mục này.

### 2. Dữ liệu đã xử lý
*   **1:** Chạy lần lượt các notebook. Dữ liệu sẽ tự động được sinh ra trong thư mục `data/processed/processed_['Loại dữ liệu']`
*   **2:** Tải nhanh dữ liệu đã được nhóm xử lý sẵn.
*   **Link tải Processed Data (Google Drive):** https://drive.google.com/drive/folders/1BU1vanjqZDSGPOf7bW4bnTJSD4jfXAEX?usp=sharing
*     - Folder processed_image: Bao gồm các file csv kết quả thống kê, các ảnh biểu đồ và dữ liệu ảnh sau khi tiền xử lý
*     - Folder processed_tabular: Bao gồm file dữ liệu sau khi xử lý
*     _ Folder processed_text: Bao gồm các ảnh biểu đồ và file csv chứa dữ liệu sau khi tiền xử lý

---

## Hướng dẫn chạy Notebook
1.  Mở thư mục dự án bằng VS Code hoặc Jupyter Lab.
2.  Đảm bảo dữ liệu raw đã đặt đúng vị trí.
3.  Chọn môi trường ảo đã thiết lập rồi chạy các file notebook
