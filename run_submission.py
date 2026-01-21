import os
import json
import requests
import SimpleITK as sitk
import tempfile
import sys

API_URL = "http://localhost:5000/api/v1/predict/patient"

def convert_dicom_to_temp_mha(dicom_folder):
    reader = sitk.ImageSeriesReader()
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        if not dicom_names:
            print(f"Error: Không tìm thấy chuỗi DICOM nào trong {dicom_folder}")
            return None
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        
        temp = tempfile.NamedTemporaryFile(suffix=".mha", delete=False)
        temp.close()
        
        sitk.WriteImage(image, temp.name)
        return temp.name
    except Exception as e:
        print(f"Error converting DICOM: {e}")
        return None

def main(folder_path, candidates_json):
    mha_path = convert_dicom_to_temp_mha(folder_path)
    
    if not mha_path:
        return

    try:
        series_uid = os.path.basename(os.path.normpath(folder_path))
        
        payload = {
            'candidates': json.dumps(candidates_json), 
            'seriesInstanceUID': series_uid
        }
        
        with open(mha_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(API_URL, data=payload, files=files, timeout=600)

        if response.status_code == 200:
            result = response.json()
            print(json.dumps(result, indent=2))
        else:
            print(f"Server Error: {response.text}")

    except Exception as e:
        print(f"Connection Error: {e}")
        
    finally:
        # Dọn dẹp file tạm
        if os.path.exists(mha_path):
            os.remove(mha_path)

if __name__ == "__main__":    
    # 1. Đường dẫn tới folder chứa file .dcm
    DICOM_FOLDER = r"C:\DuLieu_BenhVien\BenhNhan_01_TrungThat"
    
    # 2. Tọa độ các nốt cần soi
    INPUT_CANDIDATES = [
        {
            "seriesInstanceUID": "1", 
            "CoordX": 0.0, 
            "CoordY": 0.0, 
            "CoordZ": -100.0
        },
    ]
    
    if os.path.exists(DICOM_FOLDER):
        main(DICOM_FOLDER, INPUT_CANDIDATES)
    else:
        print(f"Không tìm thấy folder: {DICOM_FOLDER}")
        print("Vui lòng sửa đường dẫn trong file run_submission.py")