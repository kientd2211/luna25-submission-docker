import requests
import os

# 1. CẤU HÌNH TEST
API_URL = "http://localhost:5000/api/v1/predict/lesion"

# Bạn hãy trỏ đường dẫn này đến 1 file .mha thật trong máy bạn
# Ví dụ: file trong thư mục input mà bạn từng dùng để train
IMAGE_FILE_PATH = "C:\\Users\\tdkie\\OneDrive\\Documents\\luna25-baseline-public-main\\test\\input\\images\\chest-ct\\1.2.840.113654.2.55.294281779470566559919697495520361195429.mha" 

# Nếu chưa có file thật, bạn có thể tạo một file giả để test kết nối (nhưng sẽ lỗi lúc đọc ảnh)
if not os.path.exists(IMAGE_FILE_PATH):
    print(f"⚠️ Cảnh báo: Không tìm thấy file {IMAGE_FILE_PATH}")
    print("Vui lòng sửa đường dẫn IMAGE_FILE_PATH trong code test này thành file .mha thật của bạn.")
    # Tạo file giả tạm thời để test xem API có nhận request không
    with open("dummy.mha", "wb") as f:
        f.write(b"Mock binary data")
    IMAGE_FILE_PATH = "dummy.mha"

# 2. CHUẨN BỊ DỮ LIỆU
payload = {
    'seriesInstanceUID': '1.2.840.113654.2.TEST_UID',
    'lesionID': 1,
    'coordX': -100.5, # Tọa độ giả định
    'coordY': 50.2,
    'coordZ': 120.0
}

files = {
    'file': ('test_scan.mha', open(IMAGE_FILE_PATH, 'rb'), 'application/octet-stream')
}

# 3. GỬI REQUEST
print(f"🚀 Đang gửi request tới {API_URL}...")
print(f"📦 Metadata: {payload}")

try:
    response = requests.post(API_URL, data=payload, files=files, timeout=600)
    
    print("\n✅ KẾT QUẢ TRẢ VỀ:")
    print(f"Status Code: {response.status_code}")
    print("JSON Body:")
    print(response.json())
    
except Exception as e:
    print(f"\n❌ LỖI KHI GỌI API: {e}")

finally:
    # Xóa file giả nếu có tạo
    if IMAGE_FILE_PATH == "dummy.mha" and os.path.exists("dummy.mha"):
        os.remove("dummy.mha")