import SimpleITK as sitk
import os

# Đường dẫn file MHA bạn đã convert
FILE_MHA = "output_mha/3-1.25mm_NHU_MO_PHOI.mha"

if not os.path.exists(FILE_MHA):
    print("❌ Không tìm thấy file ảnh!")
    exit()

print(f"Đang đọc file: {FILE_MHA}")
image = sitk.ReadImage(FILE_MHA)

# Lấy kích thước và gốc tọa độ
size = image.GetSize()
origin = image.GetOrigin()
spacing = image.GetSpacing()
direction = image.GetDirection()

print(f"Kích thước (Voxel): {size}")
print(f"Gốc (Origin): {origin}")
print(f"Spacing: {spacing}")

# Tính toán điểm giữa ảnh (Center Point)
# Công thức: Center_Index = Size / 2
center_idx = (int(size[0]/2), int(size[1]/2), int(size[2]/2))

# Chuyển Index -> Tọa độ thật (Physical Point mm)
center_mm = image.TransformIndexToPhysicalPoint(center_idx)

print("\n✅ TỌA ĐỘ HỢP LỆ ĐỂ TEST (Copy vào test_patient.py):")
print(f'"CoordX": {center_mm[0]:.2f},')
print(f'"CoordY": {center_mm[1]:.2f},')
print(f'"CoordZ": {center_mm[2]:.2f}')