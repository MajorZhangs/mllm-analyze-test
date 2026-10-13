import os

#yolo訓練參數
MODEL = "yolov8n.pt"
EPOCHS = 30
IMGSZ = 640
BATCH = 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_YAML = os.path.join(BASE_DIR, "night_data_v3_split", "data.yaml")

#測試集圖片
TEST_IMAGES = os.path.join(BASE_DIR, "night_data_v3_split", "images","test")
#測試集座標(BOUNDING BOX)
TEST_LABELS = os.path.join(BASE_DIR, "night_data_v3_split","labels","test")

#驗證集圖片
VALIDS_IMAGES = os.path.join(BASE_DIR, "night_data_v3_split", "images","valid")
#驗證集座標
VALIDS_LABELS = os.path.join(BASE_DIR, "night_data_v3_split", "labels","valid")


#增強後的圖片
ENHANCED_DIR = os.path.join(BASE_DIR, "enhanced_cases")

#訓練好最佳模型
BEST_MODEL = os.path.join(
    BASE_DIR,
    "runs",
    "detect",
    "personModel",
    "weights",
    "best.pt"
)

#mllm可使用調整圖片方法與參數範圍
ALLOWED_OPS = {
    "gamma": {
        "gamma": (0.6, 1.6, 1.2)
    },

    "clahe": {
        "clip_limit": (1.0, 3.0, 1.5),
        "grid_size": (4, 12, 8)
    },

    "log_transform": {
        "strength": (0.5, 1.8, 0.8)
    },

    "retinex": {
        "sigma": (15, 80, 30)
    },

    "gaussian_illumination": {
        "sigma": (15, 80, 30),
        "strength": (0.1, 0.6, 0.3)
    }
}