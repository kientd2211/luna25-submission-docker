import requests
import json
import os

# Đường dẫn file MHA bạn vừa convert ở bước trước
# (Lưu ý dấu gạch chéo / hoặc \\ cho đúng)
FILE_MHA = "output_mha/3-1.25mm_NHU_MO_PHOI.mha"

URL = "http://localhost:5000/api/v1/predict/patient"

# Danh sách tọa độ giả lập (Mock Candidates) để test
candidates = [
    {
        "seriesInstanceUID": "1",
        "CoordX": 0.00,
        "CoordY": 0.00,
        "CoordZ": -148.88
    },
    {
        "seriesInstanceUID": "2", 
        "CoordX": 154.42, 
        "CoordY": 105.35, 
        "CoordZ": -163.54
    }
]

def test():
    if not os.path.exists(FILE_MHA):
        print(f"❌ Không tìm thấy file: {FILE_MHA}")
        return

    print(f"🚀 Đang gửi file {FILE_MHA} lên server...")
    
    payload = {
        'candidates': json.dumps(candidates), 
        'seriesInstanceUID': 'Test_Patient_01'
    }
    
    files = {
        'file': open(FILE_MHA, 'rb')
    }

    try:
        resp = requests.post(URL, data=payload, files=files, timeout=600)
        
        if resp.status_code == 200:
            print("\n✅ KẾT QUẢ THÀNH CÔNG:")
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"\n❌ LỖI ({resp.status_code}):")
            print(resp.text)
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")

if __name__ == "__main__":
    test()