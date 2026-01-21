import SimpleITK as sitk
import numpy as np
import time
import os
import torch
import cv2
from processor import MalignancyProcessor

# --- IMPORT MEDSAM ---
try:
    from segment_anything import sam_model_registry, SamPredictor
    HAS_MEDSAM = True
except ImportError:
    print("[AI] Warning: 'segment_anything' library not found. Skipping MedSAM.")
    HAS_MEDSAM = False

_model_instance = None
_medsam_predictor = None

def get_model():
    global _model_instance
    if _model_instance is None:
        print("[AI] Loading I3D Model...")
        _model_instance = MalignancyProcessor(
            mode="3D", 
            model_name="I3D-20251215",
            suppress_logs=True
        )
        print("[AI] I3D Model Ready!")
    return _model_instance

def get_medsam():
    global _medsam_predictor
    if not HAS_MEDSAM:
        return None
        
    if _medsam_predictor is None:
        ckpt_path = "/opt/app/resources/MedSAM/medsam_vit_b.pth"
        
        if not os.path.exists(ckpt_path):
            print(f"[AI] MedSAM checkpoint not found at {ckpt_path}. Skipping.")
            return None
            
        print("[AI] Loading MedSAM Model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            medsam_model = sam_model_registry["vit_b"](checkpoint=ckpt_path)
            medsam_model.to(device)
            medsam_model.eval()
            _medsam_predictor = SamPredictor(medsam_model)
            print("[AI] MedSAM Ready!")
        except Exception as e:
            print(f"[AI] Failed to load MedSAM: {e}")
            return None
        
    return _medsam_predictor

def apply_medsam_segmentation(numpy_image, center_idx, spacing):
    """
    Dùng MedSAM để segment nốt phổi
    """
    predictor = get_medsam()
    if predictor is None:
        return numpy_image 

    z, y, x = int(center_idx[0]), int(center_idx[1]), int(center_idx[2])
    
    # Check boundary
    if z < 0 or z >= numpy_image.shape[0]:
        return numpy_image
        
    # Lấy slice 2D
    img_2d = numpy_image[z, :, :]
    
    # Chuẩn hóa ảnh cho MedSAM (yêu cầu 3 kênh, 0-255)
    img_min, img_max = np.min(img_2d), np.max(img_2d)
    if img_max - img_min > 0:
        img_2d_norm = (img_2d - img_min) / (img_max - img_min) * 255.0
    else:
        img_2d_norm = img_2d
        
    img_2d_uint8 = img_2d_norm.astype(np.uint8)
    img_3c = cv2.cvtColor(img_2d_uint8, cv2.COLOR_GRAY2RGB)
    
    # Tạo Box Prompt (40mm box)
    box_size_mm = 40.0 
    box_size_px_x = int(box_size_mm / spacing[2] / 2)
    box_size_px_y = int(box_size_mm / spacing[1] / 2)
    
    bbox = np.array([
        x - box_size_px_x, y - box_size_px_y, 
        x + box_size_px_x, y + box_size_px_y 
    ])
    
    try:
        predictor.set_image(img_3c)
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False
        )
        
        mask = masks[0] 
        
        # Áp dụng Mask vào slice hiện tại
        # Logic: Giữ lại phần trong mask, phần ngoài cho về giá trị thấp nhất (đen)
        mask_binary = mask.astype(np.float32)
        
        # Lấy nền là giá trị min của ảnh để không bị đen tuyệt đối gây nhiễu biên
        bg_value = np.min(numpy_image[z, :, :])
        
        # Apply mask: Chỗ nào mask=1 giữ nguyên, mask=0 gán bg_value
        segmented_slice = np.where(mask_binary > 0, numpy_image[z, :, :], bg_value)
        
        numpy_image[z, :, :] = segmented_slice
        print("[AI] MedSAM Segmentation applied successfully.")
        
    except Exception as e:
        print(f"[AI] MedSAM Error: {e}")
        
    return numpy_image

def run_patient_inference(image_path, series_uid, candidates):
    model = get_model()
    
    print(f"[AI] Reading Image: {image_path}")
    image = sitk.ReadImage(image_path)
    
    numpyImage = sitk.GetArrayFromImage(image)
    
    spacing_sitk = image.GetSpacing()
    spacing_numpy = np.array(list(reversed(spacing_sitk)))
    
    # --- SỬA LỖI QUAN TRỌNG: DÙNG MA TRẬN 3x3 ---
    # Để khớp với vector đầu vào 3 chiều [z, y, x]
    header_bypass = {
        "origin": np.array([0.0, 0.0, 0.0]), 
        "spacing": spacing_numpy,            
        "transform": np.eye(3) # Fix: Dùng 3x3 thay vì 4x4
    }

    results = []
    
    for i, cand in enumerate(candidates):
        start_t = time.time()
        
        cx = float(cand.get("CoordX", cand.get("x", 0)))
        cy = float(cand.get("CoordY", cand.get("y", 0)))
        cz = float(cand.get("CoordZ", cand.get("z", 0)))
        c_id = cand.get("seriesInstanceUID", str(i + 1)) 
        
        prob = 0.0
        try:
            # 1. Tính toán tọa độ Voxel
            idx_x, idx_y, idx_z = image.TransformPhysicalPointToIndex((cx, cy, cz))
            center_idx = [idx_z, idx_y, idx_x] 
            
            # 2. SEGMENTATION (MedSAM)
            # Copy ảnh để xử lý riêng cho nốt này
            # (Quan trọng: Không sửa trực tiếp lên numpyImage gốc vì sẽ ảnh hưởng nốt sau)
            input_image_for_model = numpyImage.copy()
            input_image_for_model = apply_medsam_segmentation(input_image_for_model, center_idx, spacing_numpy)
            
            # 3. CLASSIFICATION (I3D)
            # Tính tọa độ input bypass
            input_z = idx_z * spacing_numpy[0]
            input_y = idx_y * spacing_numpy[1]
            input_x = idx_x * spacing_numpy[2]
            
            coords_bypass = np.array([input_z, input_y, input_x])
            
            # Đưa ảnh ĐÃ SEGMENT vào model phân loại
            model.define_inputs(input_image_for_model, header_bypass, [coords_bypass])
            prob_arr, _ = model.predict()
            
            if isinstance(prob_arr, (list, np.ndarray)) and len(prob_arr) > 0:
                prob = float(np.array(prob_arr).reshape(-1)[0])
            
        except Exception as e:
            print(f"[AI] Error processing nodule {c_id}: {e}")
            prob = 0.0

        proc_time = int((time.time() - start_t) * 1000)
        
        results.append({
            "seriesInstanceUID": str(c_id),
            "probability": prob,
            "predictionLabel": 1 if prob > 0.5 else 0,
            "processingTimeMs": proc_time,
            "CoordX": cx, "CoordY": cy, "CoordZ": cz
        })
        
    return results