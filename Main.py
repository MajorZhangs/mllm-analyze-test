import os
import cv2
import YoloTrain
import Config
import MllmEnhance
from ultralytics import YOLO

def main():

    print("1. Yolo訓練")
    print("2. 測試資料集")
    print("3. MLLM分析")
    print("4. 離開")

    choice = input("請輸入選項：")

    if choice == "1":
        print("開始訓練 YOLO 模型...")
        model = YOLO(Config.MODEL)
        
        model.train(
            data = Config.DATA_YAML,
            epochs = Config.EPOCHS,
            imgsz = Config.IMGSZ,
            batch = Config.BATCH,
            name = "personModel",
            exist_ok = True
        )

    elif choice == "2":
        print("1. Test")
        print("2. Valid")
        print("3. Enhanced")

        sub_choice = input("請輸入選項：")

        if sub_choice == "1":
            max_images = int(input("測試圖片數量："))
            YoloTrain.test(Config.BEST_MODEL,Config.TEST_IMAGES,Config.TEST_LABELS, max_images)
        elif sub_choice == "2":
            max_images = int(input("測試圖片數量："))
            YoloTrain.test(Config.BEST_MODEL,Config.VALIDS_IMAGES,Config.VALIDS_LABELS, max_images)
        elif sub_choice == "3":
            max_images = int(input("測試圖片數量："))
            YoloTrain.test(Config.BEST_MODEL,Config.ENHANCED_DIR,Config.TEST_LABELS, max_images)

            
    elif choice == "3":
        max_images = int(input("增強圖片數量："))
        MllmEnhance.main(max_images)

    elif choice == "4":
        print("離開程式。")
        return
        

if __name__ == "__main__":
    main()