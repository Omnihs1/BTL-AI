# Bài tập lớn môn Trí tuệ nhân tạo và ứng dụng
## 1. Tên đề tài 
Nhận diện khuôn mặt bằng YOLOv3
## 2. Mô tả đề tài
Implement lại mô hình YOLOv3 và sử dụng để nhận diện khuôn mặt các thành viên trong nhóm
![Cấu trúc mô hình YOLOv3](https://github.com/Omnihs1/BTL-AI/blob/main/yolov3%20architecture.png)
 
## 3. Chuẩn bị dữ liệu
- Tập dữ liệu lớn : 
Sử dụng tập dữ liệu lớn PASCAL_VOC với 11530 ảnh và 20 nhãn được chia thành 2 folder Image, Label và 2 file csv train.csv, test.csv
![alt text](https://github.com/Omnihs1/BTL-AI/blob/main/pascal_voc.png)
- Tập dữ liệu nhỏ : 
Mỗi thành viên lấy 100 ảnh (chụp, chia frame từ video) và resize kích cỡ 416x416
Sử dụng tool LabelImg để gán nhãn cho từng ảnh. 
Trích xuất thông tin bounding box từ file xml đưa thành file txt
Tạo train.csv, test.csv (tỉ lệ 80:20)
![alt text](https://github.com/Omnihs1/BTL-AI/blob/main/dung.png)

## 4. Huấn luyện mô hình
* Trên tập dữ liệu lớn với 20 epochs trên GPU TESLA T4 trên colab
![alt text](https://github.com/Omnihs1/BTL-AI/blob/main/train.png)
* Trên tập dữ liệu nhỏ
![alt text](https://github.com/Omnihs1/BTL-AI/blob/main/fine_tune_model.png)

## 5. Kết quả 
![alt text](https://github.com/Omnihs1/BTL-AI/blob/main/result.png)


