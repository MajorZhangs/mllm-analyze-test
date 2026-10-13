import os
import json
import base64

import cv2
import numpy as np
import ollama

import Config


# -------------------- 基本設定 --------------------

MLLM_MODEL = "qwen3-vl:8b"
MAX_RETRY = 4
MAX_IMAGE_DIM = 1024
MAX_OPERATIONS = 3
CONTEXT_WINDOW = 8192

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
ALLOWED_OPS = Config.ALLOWED_OPS


# -------------------- 圖片與參數工具 --------------------

def image_to_base64(image):
    """將 OpenCV 圖片編碼成 JPEG Base64，供 Ollama 使用。"""
    success, buffer = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("圖片編碼失敗")
    return base64.b64encode(buffer).decode("utf-8")


def resize_for_mllm(image):
    """只縮小送給 MLLM 分析的圖片；輸出增強仍使用原始尺寸。"""
    height, width = image.shape[:2]
    scale = min(1.0, MAX_IMAGE_DIM / max(height, width))
    if scale < 1:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def get_param(operation, name, fallback):
    """取得參數；未提供時優先使用 Config 中定義的預設值。"""
    try:
        return ALLOWED_OPS[operation][name][2]
    except (KeyError, IndexError, TypeError):
        return fallback


def build_prompt():
    prompt = f"""
你是一位專門研究夜間 CCTV 影像增強與 YOLO 人物偵測的視覺分析專家。

你的任務是分析輸入的夜間 CCTV 圖片，根據實際觀察到的影像特徵，
選擇適當的 OpenCV 影像處理操作，以改善人物的可辨識程度，
並盡可能幫助既有的 YOLO 人物偵測模型。

注意：
影像變亮不代表偵測效果一定會提升。
你的目標是保留真實影像資訊、改善有問題的區域，
而不是讓圖片看起來更漂亮。

====================
第一階段：視覺分析
====================

請依照以下順序分析目前輸入的圖片：

1. 整體亮度
   - 判斷影像是否過暗、正常或過亮。
   - 觀察暗部是否仍保有可辨識的輪廓。
   - 避免將已經足夠明亮的影像再次大幅提亮。

2. 局部對比度
   - 觀察人物與背景之間是否缺乏對比。
   - 判斷人物是否因陰影、低對比或背景干擾而難以辨識。
   - 只有局部細節確實不清楚時，才考慮 CLAHE。

3. 光照分布
   - 判斷影像是否存在明顯的局部暗區、強光或不均勻照明。
   - 不要只因為夜間場景就假設需要 Retinex 或光照補償。

4. 雜訊與細節
   - 觀察是否存在明顯顆粒雜訊、壓縮失真或模糊。
   - 不要將真實的人物細節誤判為雜訊。
   - 銳化只能強化現有邊緣，不能恢復已經遺失的資訊。

5. 人物可辨識程度
   - 觀察可見人物的輪廓、人體與背景的分離程度。
   - 注意遠處人物、暗處人物與低對比人物。
   - 不要假設畫面中一定有人，也不要憑空推測看不清楚的物體。

請只根據實際可觀察到的視覺證據做出判斷。
如果無法確認某種影像問題，請採取保守策略。

====================
第二階段：選擇操作
====================

允許的操作名稱：
{list(ALLOWED_OPS.keys())}

請只使用上述允許的操作。

操作選擇原則：

- 整體偏暗：
  可以考慮 gamma 或 log_transform。

- 局部對比度不足：
  可以考慮 clahe。

- 光照明顯不均勻：
  可以考慮 retinex 或 gaussian_illumination。

- 亮度、對比度與輪廓均正常：
  不要為了執行增強而強行選擇操作。

操作限制：

1. 每張圖片必須獨立分析，不可固定使用相同操作組合。
2. 最多選擇 3 個操作。
3. 優先選擇 0 至 2 個必要操作，不追求操作數量。
4. 只有在存在明確問題且操作具有合理改善依據時才使用。
5. 不要同時選擇多個功能高度重複的操作。
6. 如果前一個操作可能已解決主要問題，避免增加不必要的操作。
7. 不要因為圖片很暗就同時套用所有亮度增強方法。
8. 如果無法判斷增強是否有幫助，優先不處理。

====================
第三階段：參數選擇
====================

每個操作允許的參數與數值範圍如下：
{json.dumps(ALLOWED_OPS, ensure_ascii=False)}

請嚴格遵守以上參數定義。

參數選擇原則：

1. 所有參數都必須位於指定範圍內。
2. 根據實際問題的嚴重程度選擇參數。
3. 輕微問題使用溫和參數，明顯問題才考慮較強的參數。
4. 不要無條件使用預設值、最小值或最大值。
5. 不要輸出參數定義中不存在的參數。
6. 不要輸出字串形式的數值，數值必須使用 JSON number。
7. 不要輸出 NaN、Infinity 或其他非標準 JSON 值。
8. 如果某操作的參數需要整數，請輸出整數。
9. 避免過度提亮、過度銳化、雜訊放大及亮部過曝。
10. 不要為了讓圖片看起來清楚而犧牲真實人物細節。

====================
第四階段：實驗限制
====================

1. 不得修改、重新訓練或更換 YOLO 模型。
2. 只能透過允許的影像處理操作改善輸入圖片。
3. 不得裁切、平移、翻轉、旋轉或改變圖片尺寸。
4. 不得新增、移除、生成或改變人物。
5. 不得改變人物原本的位置、形狀與數量。
6. 不得假設影像增強一定能提高 YOLO 的 Precision、Recall 或 F1-score。
7. 不得根據未提供的資訊推測 YOLO 的實際偵測結果。
8. 不得為了提高理論上的偵測率而輸出缺乏影像證據的操作。
9. 如果沒有明顯且合理可改善的問題，operations 必須是空陣列。
10. 不得使用允許列表以外的操作。
11. 不得輸出超出指定範圍的參數。

====================
第五階段：輸出格式
====================

只能輸出一個有效的 JSON 物件。
禁止輸出 Markdown、程式碼區塊、前言、結語或額外說明。

輸出格式：

{{
    "reason": "根據實際影像特徵，簡潔說明主要問題及選擇操作的原因",
    "operations": [
        {{
            "name": "允許的操作名稱",
            "params": {{
                "參數名稱": 數值
            }}
        }}
    ]
}}

輸出規則：

- reason 必須是簡短且具體的繁體中文。
- name 必須完全符合允許的操作名稱。
- params 必須符合該操作的參數定義。
- operations 必須是 JSON 陣列。
- 不需要增強時，operations 必須為 []。
- 不得在 JSON 前後添加任何文字。
- 不得輸出註解或 Markdown 標記。
- 不得輸出未定義的額外欄位。

如果圖片沒有明顯需要處理的問題，請輸出：

{{
    "reason": "目前可觀察到的亮度、對比度及人物輪廓沒有明顯可改善的問題，因此不進行額外增強。",
    "operations": []
}}

現在請獨立分析輸入圖片，依照上述規則輸出 JSON。
"""
    return prompt


def sanitize_plan(data):
    """過濾未知操作、補上預設參數並將參數限制在 Config 範圍內。"""
    if not isinstance(data, dict):
        return {"reason": "MLLM 回傳格式錯誤", "operations": []}

    clean_operations = []
    for operation in data.get("operations", [])[:MAX_OPERATIONS]:
        if not isinstance(operation, dict):
            continue

        name = operation.get("name")
        if name not in ALLOWED_OPS:
            continue

        raw_params = operation.get("params", {})
        if not isinstance(raw_params, dict):
            raw_params = {}

        params = {}
        for param_name, limits in ALLOWED_OPS[name].items():
            low, high, default = limits
            try:
                value = float(raw_params.get(param_name, default))
            except (TypeError, ValueError):
                value = default

            value = max(low, min(value, high))

            # OpenCV kernel 必須為正奇數；grid_size 則必須為正整數。
            if param_name in ("grid_size", "ksize"):
                value = max(1, int(round(value)))
                if param_name == "ksize" and value % 2 == 0:
                    value += 1
            params[param_name] = value

        clean_operations.append({"name": name, "params": params})

    return {
        "reason": str(data.get("reason", "未提供分析原因")),
        "operations": clean_operations,
    }


# -------------------- MLLM 分析 --------------------

def analyze_with_mllm(image, prompt):
    """呼叫本機 Ollama；失敗時重試，全部失敗則回傳 None。"""
    encoded_image = image_to_base64(resize_for_mllm(image))

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = ollama.chat(
                model=MLLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [encoded_image],
                }],
                options={"num_ctx": CONTEXT_WINDOW},
                format="json",
            )
            content = response["message"]["content"]
            return sanitize_plan(json.loads(content))

        except Exception as error:
            print(f"MLLM 分析失敗（{attempt}/{MAX_RETRY}）：{error}")

    return None


# -------------------- OpenCV 影像增強 --------------------

def enhance_image(image, plan):
    """依照已驗證的操作清單處理圖片；未支援的操作會略過。"""
    result = image.copy()

    for operation in plan.get("operations", []):
        name = operation["name"]
        p = operation["params"]
        print(f"  執行：{name} -> {p}")

        if name == "gamma":
            gamma = max(float(p.get("gamma", 1.0)), 0.01)
            lut = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            result = cv2.LUT(result, lut)

        elif name == "clahe":
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lightness, a, b = cv2.split(lab)
            grid = max(1, int(p.get("grid_size", 8)))
            clahe = cv2.createCLAHE(
                clipLimit=float(p.get("clip_limit", 2.0)),
                tileGridSize=(grid, grid),
            )
            lightness = clahe.apply(lightness)
            result = cv2.cvtColor(
                cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR
            )

        elif name == "log_transform":
            strength = max(float(p.get("strength", 1.0)), 0.01)
            lut = np.array([
                255 * np.log1p(strength * i / 255) / np.log1p(strength)
                for i in range(256)
            ], dtype=np.uint8)
            result = cv2.LUT(result, lut)

        elif name == "retinex":
            sigma = max(float(p.get("sigma", 30)), 0.1)
            source = result.astype(np.float32) + 1
            illumination = cv2.GaussianBlur(source, (0, 0), sigma)
            corrected = np.log(source) - np.log(illumination + 1)
            channels = [
                cv2.normalize(corrected[:, :, i], None, 0, 255, cv2.NORM_MINMAX)
                for i in range(3)
            ]
            result = cv2.merge([channel.astype(np.uint8) for channel in channels])

        elif name == "gaussian_illumination":
            sigma = max(float(p.get("sigma", 30)), 0.1)
            strength = float(p.get("strength", 1.0))
            source = result.astype(np.float32)
            background = cv2.GaussianBlur(source, (0, 0), sigma)
            average = np.mean(background, axis=(0, 1), keepdims=True)
            result = np.clip(
                source + strength * (average - background), 0, 255
            ).astype(np.uint8)

        else:
            print(f"  尚未實作此操作，略過：{name}")

    return result


# -------------------- 輸出資料夾與主流程 --------------------

def clear_output_folder(folder):
    """清除輸出資料夾中的檔案；不刪除子資料夾。"""
    os.makedirs(folder, exist_ok=True)
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            os.remove(path)


def main(max_images=None):
    """處理 Test 圖片；max_images=None 代表處理全部圖片。"""
    input_dir = Config.TEST_IMAGES
    output_dir = Config.ENHANCED_DIR
    clear_output_folder(output_dir)

    if not os.path.isdir(input_dir):
        print(f"找不到 Test 資料夾：{input_dir}")
        return

    image_names = sorted(
        name for name in os.listdir(input_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    )
    if max_images is not None:
        image_names = image_names[:max(0, max_images)]

    if not image_names:
        print("找不到可處理的 Test 圖片。")
        return

    prompt = build_prompt()
    print(f"準備處理 {len(image_names)} 張圖片。")

    for index, name in enumerate(image_names, start=1):
        print(f"\n[{index}/{len(image_names)}] {name}")
        input_path = os.path.join(input_dir, name)
        image = cv2.imread(input_path)

        if image is None:
            print("圖片讀取失敗，跳過。")
            continue

        plan = analyze_with_mllm(image, prompt)
        if plan is None:
            print("MLLM 多次分析失敗，保留原圖。")
            enhanced = image
        else:
            print(f"分析原因：{plan['reason']}")
            enhanced = enhance_image(image, plan)

        output_path = os.path.join(output_dir, name)
        if cv2.imwrite(output_path, enhanced):
            print(f"已儲存：{output_path}")
        else:
            print(f"儲存失敗：{output_path}")

    print("\n影像處理完成。")
    print(f"原始圖片：{input_dir}")
    print(f"增強圖片：{output_dir}")