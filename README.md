# Phân loại cảm xúc văn bản
Phân loại cảm xúc bằng phương pháp Random Forest
## Cách cài đặt
Để chạy chương trình, ta cần phải cài đặt những thư viện cần thiết

**_(Yêu cầu python phiên bản 3.7)_**

**Cài đặt những package cần thiết cho chương trình**
```
$ pip install -r requirements.txt
```

**Sau khi cài đặt ta chạy lần lượt những lệnh sau để huấn luyện mô hình**

**_(Lưu ý: Thêm dữ liệu huấn luyện vào thư mục raw)_**
```
$ python src/data/data_preprocessing.py -i [INPUT_RAW_DATA] -o [OUTPUT_PREPROCESSED_DATA]
$ python src/models/train_rf_model.py -i [INPUT_PREPROCESSED_DATA] -o [OUTPUT_MODEL]
```
Ví dụ:
```
$ python src/data/data_preprocessing.py -i raw_data.csv -o out.csv
$ python src/models/train_rf_models.py -i out.csv -o rf_model.pickle
```

**Sau khi đã huấn luyện xong mô hình, ta chạy lệnh sau để dự đoán**
```
$ python src/models/predict_model.py -t [SENTENCE] -f [MODEL]
```

Ví dụ:
```
$ python src/models/predict_model.py -t "I hate twitter" -f rf_model.pickle
```
