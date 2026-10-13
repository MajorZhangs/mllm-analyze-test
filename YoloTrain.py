import os
import cv2
from ultralytics import YOLO
import Config
import numpy as np
from scipy.optimize import linear_sum_assignment

#計算iou(ground truth 與 prediction 的重疊程度)
def iou(box1, box2):

    x1 = max(box1[0], box2[0])#left
    y1 = max(box1[1], box2[1])#top
    x2 = min(box1[2], box2[2])#right
    y2 = min(box1[3], box2[3])#bottom
    #重疊面積計算(max防止兩個框根本沒重疊會出現負數max(0,minus) = 0)
    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])#box1面積
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])#box2面積

    union = area1 + area2 - inter#兩張圖總佔面積(重疊部分只算一次)

    return inter / union if union > 0 else 0#union為0時回傳0避免除以0

#座標轉換(中心點轉四方座標)
def transformCoordinate(path, w, h):
    
    boxes = []#yolo格式轉成iou格式

    if not os.path.exists(path):#找不到檔案就回傳空的list
        return boxes

    with open(path, "r", encoding="utf-8") as f:#開啟yolo標註檔
        for line in f:#每行讀取
            v = line.split()#行切割

            if len(v) < 5:#確認數據正確(有5項)
                continue

            # 只處理 person(cuz class person = 0)
            if int(v[0]) != 0:
                continue
            # 中心點座標(因為yolo是比例 所以要乘上圖片寬高)
            xc = float(v[1]) * w
            yc = float(v[2]) * h
            # 寬高座標
            bw = float(v[3]) * w
            bh = float(v[4]) * h
            # 十字型座標轉換成四方座標(max跟min防止超出圖片範圍)
            x1 = max(0, int(xc - bw / 2))
            y1 = max(0, int(yc - bh / 2))
            x2 = min(w - 1, int(xc + bw / 2))
            y2 = min(h - 1, int(yc + bh / 2))

            if x2 <= x1 or y2 <= y1:#確認框正確(不會出現負數)
                continue
            
            boxes.append([x1, y1, x2, y2])#儲存結果

    return boxes


def test(model, data_dir, label_dir, max_images=None):
    CONF = 0.3
    IOU_THRESHOLD = 0.5

    TP = 0
    FP = 0
    FN = 0

    # 載入 YOLO 模型
    model = YOLO(model) if isinstance(model, str) else model

    # 取得測試圖片
    images = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    total_images = len(images)

    if max_images is not None:
        if max_images <= 0:
            print("測試圖片數量必須大於 0")
            return None

        images = images[:max_images]

    print(f"\n測試圖片數量：{len(images)} / {total_images}")
    print("測試中...")

    for name in images:
        image_path = os.path.join(data_dir, name)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(name)[0] + ".txt"
        )

        # 確認圖片能否讀取
        image = cv2.imread(image_path)

        if image is None:
            print(f"無法讀取圖片，跳過：{name}")
            continue

        # 缺少標註檔不能直接當作沒有人物
        if not os.path.exists(label_path):
            print(f"缺少標註檔，跳過：{name}")
            continue

        h, w = image.shape[:2]

        # 取得 GT 標註框
        gt = transformCoordinate(label_path, w, h)

        # YOLO 偵測
        result = model.predict(
            image,
            conf=CONF,
            verbose=False
        )[0]

        pred = []

        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    pred.append(
                        box.xyxy[0].cpu().numpy()
                    )

        # 沒有 GT 或沒有預測框時，不需要進行配對
        if len(gt) == 0 or len(pred) == 0:
            TP += 0
            FP += len(pred)
            FN += len(gt)
            continue

        # 建立 IoU 矩陣
        # 每一列代表一個預測框，每一欄代表一個 GT 框
        iou_matrix = np.zeros(
            (len(pred), len(gt)),
            dtype=float
        )

        for p_idx, p in enumerate(pred):
            for g_idx, g in enumerate(gt):
                iou_matrix[p_idx, g_idx] = iou(p, g)

        # 建立配對成本矩陣
        # 符合門檻的配對給予負成本
        # 越大的 IoU，成本越低
        # 不符合門檻的配對成本設為 0
        cost_matrix = np.zeros_like(iou_matrix)

        valid_pairs = iou_matrix >= IOU_THRESHOLD
        cost_matrix[valid_pairs] = -(
            1.0 + iou_matrix[valid_pairs]
        )

        # Hungarian Algorithm：
        # 找出整體較好的預測框與 GT 配對
        pred_indices, gt_indices = linear_sum_assignment(
            cost_matrix
        )

        matched_pred = set()
        matched_gt = set()

        # 只接受 IoU 達到門檻的配對
        for p_idx, g_idx in zip(pred_indices, gt_indices):
            if iou_matrix[p_idx, g_idx] >= IOU_THRESHOLD:
                matched_pred.add(p_idx)
                matched_gt.add(g_idx)

        # 計算這張圖片的 TP、FP、FN
        image_tp = len(matched_pred)
        image_fp = len(pred) - image_tp
        image_fn = len(gt) - len(matched_gt)

        TP += image_tp
        FP += image_fp
        FN += image_fn

    # 計算整體指標
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    print("\nTEST 結果")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print("------------------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


"""
def test(model, data_dir, label_dir, max_images=None):
    CONF = 0.3
    IOU = 0.5

    TP = 0
    FP = 0
    FN = 0

    model = YOLO(model)

    # 取得所有圖片
    images = sorted([
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    total_images = len(images)

    if max_images is not None:
        images = images[:max_images]

    print(f"\n測試圖片數量：{len(images)} / {total_images}")
    print("測試中...")

    # 逐張圖片進行測試
    for name in images:
        image_path = os.path.join(data_dir, name)
        label_path = os.path.join(
            label_dir,
            os.path.splitext(name)[0] + ".txt"
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        # 讀取 Ground Truth，轉換成 xyxy 格式
        gt = transformCoordinate(label_path, w, h)

        # YOLO 推論，只保留 person 類別
        result = model.predict(
            image,
            conf=CONF,
            verbose=False
        )[0]

        pred = []

        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls[0]) == 0:
                    pred.append(box.xyxy[0].cpu().numpy())

        matched_gt = set()
        matched_pred = set()

        # 依照 IoU 配對 Prediction 與 Ground Truth
        for p_idx, p in enumerate(pred):
            best_iou = 0
            best_g_idx = -1

            for g_idx, g in enumerate(gt):
                if g_idx in matched_gt:
                    continue

                score = iou(p, g)

                if score > best_iou:
                    best_iou = score
                    best_g_idx = g_idx

            if best_iou >= IOU and best_g_idx >= 0:
                TP += 1
                matched_gt.add(best_g_idx)
                matched_pred.add(p_idx)
            else:
                FP += 1

        # 未被配對的 Ground Truth 都是 FN
        image_fn = len(gt) - len(matched_gt)
        FN += image_fn

    # 計算評估指標
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    # 顯示測試結果
    print("\nTEST 結果")
    print(f"TP: {TP} | FP: {FP} | FN: {FN}")
    print("------------------------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
"""