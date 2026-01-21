import SimpleITK as sitk
import os
import sys

# --- CẤU HÌNH ---
# Copy chính xác từ output lệnh 'dir' của bạn
PARENT_DIR = r"D:\Code\MTN\MTN\123.255088989375905.1851013063013701\22018841-NGUYEN THI MECH\HA251209.117_CT"
OUTPUT_DIR = "output_mha"

def convert_series_to_mha(series_dir, output_filename):
    print(f"\n📂 Đang xử lý: {os.path.basename(series_dir)}")
    
    reader = sitk.ImageSeriesReader()
    
    # Tìm file DICOM
    try:
        dicom_names = reader.GetGDCMSeriesFileNames(series_dir)
    except Exception as e:
        print(f"   ❌ Lỗi khi quét file: {e}")
        return False

    if not dicom_names:
        print("   ⚠️ Folder này trống hoặc không phải DICOM Series.")
        return False
        
    print(f"   -> Tìm thấy {len(dicom_names)} file .dcm")
    reader.SetFileNames(dicom_names)
    
    try:
        image = reader.Execute()
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        sitk.WriteImage(image, output_path)
        print(f"   ✅ Đã lưu thành công: {output_path}")
        print(f"   📏 Kích thước ảnh: {image.GetSize()}")
        return True
    except Exception as e:
        print(f"   ❌ Lỗi khi convert: {e}")
        return False

def main():
    print("--- BẮT ĐẦU SCRIPT ---")
    print(f"Đường dẫn gốc: {PARENT_DIR}")

    # 1. Kiểm tra đường dẫn gốc có tồn tại không
    if not os.path.exists(PARENT_DIR):
        print("❌ LỖI NGHIÊM TRỌNG: Máy tính không tìm thấy đường dẫn trên!")
        print("   Vui lòng kiểm tra lại xem ổ D: có kết nối không hoặc copy sai đường dẫn.")
        return

    # 2. Tạo folder đầu ra
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Đã tạo thư mục chứa kết quả: {os.path.abspath(OUTPUT_DIR)}")

    # 3. Quét thư mục con
    try:
        subfolders = [f.path for f in os.scandir(PARENT_DIR) if f.is_dir()]
    except Exception as e:
        print(f"❌ Lỗi khi quét thư mục con: {e}")
        return

    print(f"🔍 Tìm thấy {len(subfolders)} folder con.")

    if len(subfolders) == 0:
        print("⚠️ Cảnh báo: Không có folder con nào bên trong đường dẫn gốc.")
        return

    # 4. Thực hiện convert
    count = 0
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        # Tạo tên file output an toàn (thay khoảng trắng bằng gạch dưới)
        safe_name = folder_name.replace(" ", "_") + ".mha"
        
        if convert_series_to_mha(folder, safe_name):
            count += 1
            
    print(f"\n--- HOÀN THÀNH: Đã convert {count} chuỗi ảnh ---")

if __name__ == "__main__":
    main()