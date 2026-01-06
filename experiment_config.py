# experiment_config.py
import os
from pathlib import Path

class Configuration(object):
    def __init__(self) -> None:
        # Trong Docker, code sẽ nằm ở /app, dữ liệu sẽ mount vào /data
        # Hoặc dùng biến môi trường để linh hoạt
        
        # Đường dẫn gốc chứa dữ liệu (được mount từ bên ngoài vào)
        self.DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))

        # Working directory
        self.WORKDIR = Path("/opt/app")
        self.RESOURCES = self.WORKDIR / "resources"
        
        # Starting weights
        self.MODEL_RGB_I3D = self.RESOURCES / "model_rgb.pth"
        
        # Data parameters
        self.DATADIR = self.DATA_ROOT / "output_test_2" # Folder chứa nodule blocks
        self.DATADIR_TRAIN = self.DATA_ROOT / "output_test_2"
        self.DATADIR_VALID = self.DATA_ROOT / "output_valid_2"
        
        # CSV paths
        self.CSV_DIR = self.DATA_ROOT / "dataset_csv"
        self.CSV_DIR_TRAIN = self.DATA_ROOT / "train_filtered.csv"
        self.CSV_DIR_VALID = self.DATA_ROOT / "test_filtered.csv"

        # Results
        self.EXPERIMENT_DIR = self.WORKDIR / "results"
        if not self.EXPERIMENT_DIR.exists():
            self.EXPERIMENT_DIR.mkdir(parents=True)
            
        self.EXPERIMENT_NAME = "I3D"
        self.MODE = "3D" 

        # Training parameters (Giữ nguyên)
        self.SEED = 2025
        self.NUM_WORKERS = 8 # Lưu ý: Docker cần share memory lớn nếu worker nhiều
        self.SIZE_MM = 50
        self.SIZE_PX = 64
        self.BATCH_SIZE = 32
        self.ROTATION = ((-20, 20), (-20, 20), (-20, 20))
        self.TRANSLATION = True
        self.EPOCHS = 50
        self.PATIENCE = 20
        self.PATCH_SIZE = [64, 128, 128]
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 5e-4

config = Configuration()