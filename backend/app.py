import os
import time
import logging
import SimpleITK as sitk
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cấu hình log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- KẾT NỐI VỚI CODE AI ---
try:
    # Import hàm run_inference vừa viết thêm ở Bước 1
    from inference import run_inference 
    logger.info("Successfully imported run_inference from inference.py")
except ImportError as e:
    logger.error(f"Critical Error: Could not import inference.py. Details: {e}")
    run_inference = None
# ---------------------------

@app.route('/health', methods=['GET'])
def health_check():
    status = "healthy" if run_inference else "degraded (no model)"
    return jsonify({"status": status, "service": "LUNA25 Submission"}), 200

@app.route('/api/v1/predict/lesion', methods=['POST'])
def predict_lesion():
    start_time = time.time()
    
    # 1. KIỂM TRA MODEL
    if run_inference is None:
        return jsonify({"error": "INTERNAL_SERVER_ERROR", "details": "Model not loaded properly"}), 500

    # 2. VALIDATE REQUEST
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    required_fields = ['seriesInstanceUID', 'lesionID', 'coordX', 'coordY', 'coordZ']
    for field in required_fields:
        if field not in request.form:
            return jsonify({"error": f"Missing field: {field}"}), 400

    temp_path = ""
    try:
        # 3. LƯU FILE TẠM
        # Lưu vào thư mục /app/uploads để code inference đọc được
        upload_folder = "/opt/app/uploads"
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
            
        filename = file.filename
        temp_path = os.path.join(upload_folder, filename)
        file.save(temp_path)

        # 4. LẤY DỮ LIỆU TỌA ĐỘ
        coord_x = float(request.form['coordX'])
        coord_y = float(request.form['coordY'])
        coord_z = float(request.form['coordZ'])
        lesion_id = int(request.form['lesionID'])
        series_uid = request.form['seriesInstanceUID']

        # 5. GỌI HÀM DỰ ĐOÁN THẬT
        logger.info(f"Processing lesion {lesion_id} at ({coord_x}, {coord_y}, {coord_z})")
        probability, label = run_inference(temp_path, coord_x, coord_y, coord_z)

        # 6. TRẢ KẾT QUẢ
        processing_time = int((time.time() - start_time) * 1000)

        response_data = {
            "status": "success",
            "data": {
                "seriesInstanceUID": series_uid,
                "lesionID": lesion_id,
                "probability": probability,
                "predictionLabel": label,
                "processingTimeMs": processing_time
            }
        }
        return jsonify(response_data), 200

    except Exception as e:
        logger.error(f"Prediction Error: {str(e)}")
        return jsonify({"error": "INTERNAL_SERVER_ERROR", "details": str(e)}), 500
    
    finally:
        # Dọn dẹp file tạm
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    # Khi chạy trực tiếp (không qua Gunicorn), ta load model trước để test
    try:
        from inference import get_model_instance
        get_model_instance()
    except:
        pass
    app.run(host='0.0.0.0', port=5000)