import os
import time
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kết nối với inference.py
try:
    from inference import run_patient_inference
    logger.info("Successfully imported run_patient_inference")
except ImportError as e:
    logger.error(f"Critical Error: {e}")
    run_patient_inference = None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ready"}), 200

@app.route('/api/v1/predict/patient', methods=['POST'])
def predict_patient():
    """
    API dự đoán cho cả ca bệnh (nhiều nốt phổi cùng lúc)
    """
    start_time = time.time()
    
    if run_patient_inference is None:
        return jsonify({"error": "Model not loaded"}), 500

    # 1. Nhận File
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    
    # 2. Nhận danh sách tọa độ (Candidates) dạng chuỗi JSON
    candidates_json = request.form.get('candidates')
    if not candidates_json:
        return jsonify({"error": "Missing 'candidates' list"}), 400
    
    try:
        candidates = json.loads(candidates_json)
    except:
        return jsonify({"error": "Invalid JSON format for candidates"}), 400

    series_uid = request.form.get('seriesInstanceUID', 'Unknown')
    temp_path = ""

    try:
        # 3. Lưu file tạm vào /opt/app/uploads
        upload_folder = "/opt/app/uploads"
        os.makedirs(upload_folder, exist_ok=True)
        
        filename = file.filename
        temp_path = os.path.join(upload_folder, filename)
        file.save(temp_path)

        # 4. Gọi Inference xử lý cả danh sách
        results = run_patient_inference(temp_path, series_uid, candidates)

        # 5. Trả về kết quả (List JSON)
        return jsonify(results), 200

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        # Dọn dẹp file tạm
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)