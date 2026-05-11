# SuperGlue 中文操作手冊

> 原始論文：SuperGlue: Learning Feature Matching with Graph Neural Networks（CVPR 2020, Magic Leap）  
> 本手冊以**你自己的資料（Bullpen_CalibrationBar）**為主軸說明操作流程，適用 Windows + PowerShell 環境。

---

## 一、這個專案在做什麼？

SuperGlue 是一個特徵點匹配系統，給定兩張影像，找出彼此對應的像素位置（特徵點對）。

整個流程分兩步：

```
影像 A ──┐
          ├──► SuperPoint（偵測關鍵點 + 提取描述子）──► SuperGlue（GNN 匹配）──► 匹配點對
影像 B ──┘
```

- **SuperPoint**：CNN，在每張圖上偵測出最顯著的角點，並計算每個點的 256 維特徵向量（描述子）
- **SuperGlue**：圖神經網路，比較兩張圖的所有描述子，找出最佳匹配，輸出匹配索引與信心分數

用途：相機定位、三維重建、影像對齊、視覺里程計等。

---

## 二、環境建立（uv 虛擬環境）

使用 `uv` 建立乾淨的專案虛擬環境，避免與其他專案的套件衝突。

**需求**：
- Anaconda（已安裝）
- uv（已安裝，v0.9.18）
- NVIDIA RTX 5080 + CUDA 12.8

### 建立虛擬環境（只需做一次）

```powershell
cd "c:\Mo\program\SuperGlueGnn"

# 建立 .venv 虛擬環境（Python 3.11）
uv venv .venv --python 3.11

# 安裝一般套件
uv pip install --python .venv/Scripts/python.exe numpy matplotlib opencv-python PyQt5

# 安裝 PyTorch（CUDA 12.8 版，約 2.6GB，需要一段時間）
$env:UV_HTTP_TIMEOUT = "300"
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 驗證環境是否正常

```powershell
.venv/Scripts/python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
# 應該印出：CUDA: True
```

### 之後每次使用，執行指令前加上 `.venv/Scripts/python.exe`

```powershell
# 範例
.venv/Scripts/python.exe match_pairs.py --help
```

| 套件 | 版本需求 | 已安裝版本 |
|------|----------|-----------|
| Python | >= 3.5 | 3.11（venv）|
| PyTorch | >= 1.1 | 2.11+cu128 |
| OpenCV | >= 3.4 | 4.13 |
| NumPy | >= 1.18 | 2.4.4 |
| Matplotlib | >= 3.1 | 3.10 |
| PyQt5 | 任意 | 5.15.11 |

---

## 三、專案檔案結構

```
SuperGlueGnn/
├── match_pairs.py            ← 主程式（批次匹配 + 評估）
├── demo_superglue.py         ← 即時 webcam demo
├── superglue_viewer.py       ← 查看輸出結果的 UI
├── models/
│   ├── superpoint.py         ← 關鍵點偵測模型
│   ├── superglue.py          ← GNN 匹配模型
│   ├── matching.py           ← 將上面兩個串在一起的 pipeline
│   ├── utils.py              ← 讀圖、視覺化、評估等工具
│   └── weights/
│       ├── superpoint_v1.pth         ← SuperPoint 預訓練權重
│       ├── superglue_indoor.pth      ← 室內場景模型（ScanNet）
│       └── superglue_outdoor.pth     ← 戶外場景模型（MegaDepth）
├── Bullpen_CalibrationBar/   ← 你的自訂資料集
│   ├── cf/                   ← Camera Front 影像（1.jpg ~ 21.jpg）
│   ├── cs/                   ← Camera Side 影像（1.jpg ~ 21.jpg）
│   ├── pairs.txt             ← 影像配對清單（已生成）
│   └── output/               ← 輸出結果會放這裡
└── assets/                   ← 官方範例資料集
```

---

## 四、快速開始：跑你自己的資料

### Step 1：確認 pairs.txt 存在

檔案已在 `Bullpen_CalibrationBar/pairs.txt`，內容如下：

```
cf/1.jpg cs/1.jpg
cf/2.jpg cs/2.jpg
...
cf/21.jpg cs/21.jpg
```

每一行代表一對要匹配的影像（左邊是 Camera Front，右邊是 Camera Side）。

---

### Step 2：執行匹配（最基本的指令）

開啟 PowerShell，切換到專案目錄，執行：

```powershell
cd "c:\Mo\program\SuperGlueGnn"

.venv/Scripts/python.exe match_pairs.py `
  --input_pairs Bullpen_CalibrationBar/pairs.txt `
  --input_dir Bullpen_CalibrationBar `
  --output_dir Bullpen_CalibrationBar/output `
  --superglue outdoor `
  --viz --fast_viz `
  --resize 640 480
```

> 注意：PowerShell 中換行要用反引號 `` ` ``。也可以直接寫成一行。

執行完成後，`Bullpen_CalibrationBar/output/` 會產生：
- `1_1_matches.png`：視覺化結果圖（cf/1.jpg 配 cs/1.jpg）
- `1_1_matches.npz`：原始數值資料（關鍵點座標、匹配索引、信心分數）
- ... 共 21 對

---

### Step 3：查看視覺化結果

**方法一：用 SuperGlue 瀏覽器**（推薦）

```powershell
.venv/Scripts/python.exe superglue_viewer.py
```

啟動後選擇 `Bullpen_CalibrationBar/output/` 資料夾，可逐張瀏覽。

**方法二：直接用檔案總管開啟 PNG**

到 `Bullpen_CalibrationBar/output/` 打開任一 `*_matches.png`。

---

### 讀懂結果圖

```
┌────────────────────────────────────────────────────┐
│  左半邊：Camera Front 影像    右半邊：Camera Side 影像  │
│                                                    │
│  線條顏色 = 匹配信心度（jet colormap）                 │
│    藍色 → 低信心度                                    │
│    紅色 → 高信心度                                    │
│                                                    │
│  左上角顯示：關鍵點數量、匹配數量                        │
└────────────────────────────────────────────────────┘
```

---

## 五、完整參數說明（`match_pairs.py`）

### 輸入輸出

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--input_pairs` | `assets/scannet_sample_pairs_with_gt.txt` | pairs.txt 路徑 |
| `--input_dir` | `assets/scannet_sample_images/` | 影像所在目錄 |
| `--output_dir` | `dump_match_pairs/` | 結果輸出目錄 |
| `--max_length` | `-1`（全部） | 最多處理幾對（測試用可設 `2`） |

### 模型選擇

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--superglue` | `indoor` | 選 `indoor`（室內）或 `outdoor`（戶外）|
| `--max_keypoints` | `1024` | 每張圖最多偵測幾個關鍵點（-1 = 不限）|
| `--keypoint_threshold` | `0.005` | 關鍵點偵測閾值（越低 = 偵測越多點）|
| `--nms_radius` | `4` | 關鍵點非極大值抑制半徑（像素）|
| `--sinkhorn_iterations` | `20` | Sinkhorn 最優傳輸迭代次數 |
| `--match_threshold` | `0.2` | 匹配接受閾值（越低 = 保留更多匹配）|

### 影像處理

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--resize` | `640 480` | 縮放到指定尺寸（寬 高）|
| `--resize` `1280` | - | 只給一個數字 = 最長邊縮到此值 |
| `--resize` `-1` | - | 不縮放，使用原始解析度 |
| `--resize_float` | 關閉 | 縮放時使用浮點數（戶外場景建議加）|

### 視覺化

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--viz` | 關閉 | 輸出視覺化 PNG/PDF |
| `--fast_viz` | 關閉 | 用 OpenCV 繪圖（速度快，需搭配 `--viz`）|
| `--show_keypoints` | 關閉 | 在圖上顯示所有偵測到的關鍵點 |
| `--viz_extension` | `png` | 輸出格式（`png` 或 `pdf`）|

### 其他

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--eval` | 關閉 | 啟用評估模式（需要 GT，見第六節）|
| `--cache` | 關閉 | 跳過已處理的配對（重跑時節省時間）|
| `--force_cpu` | 關閉 | 強制使用 CPU（預設自動用 GPU）|
| `--shuffle` | 關閉 | 隨機打亂處理順序 |

---

## 六、常用指令範本

> 所有指令都在 `c:\Mo\program\SuperGlueGnn` 目錄下執行，並使用 `.venv` 虛擬環境的 Python。

### 基本匹配（你的資料，戶外模型）

```powershell
cd "c:\Mo\program\SuperGlueGnn"
.venv/Scripts/python.exe match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output --superglue outdoor --viz --fast_viz --resize 640 480
```

### 先只跑前 2 對快速確認（測試用）

```powershell
.venv/Scripts/python.exe match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output --superglue outdoor --viz --fast_viz --resize 640 480 --max_length 2
```

### 提高解析度 + 更多關鍵點（匹配點可能更多）

```powershell
.venv/Scripts/python.exe match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output_hires --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float --viz --fast_viz --resize 1280
```

### 顯示所有關鍵點（含未匹配的）

```powershell
.venv/Scripts/python.exe match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output --superglue outdoor --viz --fast_viz --resize 640 480 --show_keypoints
```

### 官方室內範例（驗證環境是否正常）

```powershell
.venv/Scripts/python.exe match_pairs.py --viz --fast_viz
```

結果會出現在 `dump_match_pairs/`，應該可以看到 ScanNet 室內場景的匹配圖。

### 官方戶外範例

```powershell
.venv/Scripts/python.exe match_pairs.py --resize 1600 --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float --input_dir assets/phototourism_sample_images/ --input_pairs assets/phototourism_sample_pairs.txt --output_dir dump_match_pairs_outdoor --viz --fast_viz
```

---

## 七、讀取 .npz 輸出資料（Python）

每對影像匹配完會輸出一個 `.npz` 檔，可以這樣讀取：

```python
import numpy as np

npz = np.load('Bullpen_CalibrationBar/output/1_1_matches.npz')

print(npz.files)
# ['keypoints0', 'keypoints1', 'matches', 'match_confidence']

print(npz['keypoints0'].shape)   # (N, 2) — cf 影像的關鍵點座標 (x, y)
print(npz['keypoints1'].shape)   # (M, 2) — cs 影像的關鍵點座標 (x, y)
print(npz['matches'].shape)      # (N,)   — 每個 kp0 對應的 kp1 索引，-1 = 未匹配
print(npz['match_confidence'].shape)  # (N,) — 每個匹配的信心分數 (0~1)

# 篩選有效匹配
valid = npz['matches'] > -1
matched_kps0 = npz['keypoints0'][valid]           # cf 匹配點
matched_kps1 = npz['keypoints1'][npz['matches'][valid]]  # 對應的 cs 匹配點
confidence   = npz['match_confidence'][valid]
print(f'有效匹配數：{valid.sum()}')
```

---

## 八、評估模式（需要 GT）

如果你有相機內參 K 和相對姿態 T（地面真值），可以啟用 `--eval` 計算定量評估指標。

### pairs.txt 格式（38 欄）

```
name0 name1 rot0 rot1  K0_11 K0_12 ... K0_33  K1_11 ... K1_33  T_11 T_12 ... T_44
```

- `rot0`, `rot1`：EXIF 旋轉（0=不旋轉，無資訊填 0）
- `K0`, `K1`：3×3 內參矩陣，**逐行展平**成 9 個數值
- `T_0to1`：4×4 相對姿態矩陣（cam1 相對於 cam0），**逐行展平**成 16 個數值

### 如何取得 K 和 T

**相機內參 K（一次性校正）**：
```python
import cv2
# 用棋盤格拍 10+ 張照片，執行 calibrateCamera
ret, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
# reprojection error < 1 px 才算可信
```

**相對姿態 T（每對影像都要）**：
```python
import cv2, numpy as np

# 用已知 3D 座標的標記點 + solvePnP
ret, rvec0, tvec0 = cv2.solvePnP(pts_3d, pts2d_cam0, K, dist)
ret, rvec1, tvec1 = cv2.solvePnP(pts_3d, pts2d_cam1, K, dist)

def make_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3,:3] = R
    T[:3, 3] = tvec.ravel()
    return T

T_0to1 = make_T(rvec1, tvec1) @ np.linalg.inv(make_T(rvec0, tvec0))
```

**注意**：建議先用 `cv2.undistort()` 去除鏡頭畸變再送入匹配，可以降低 epipolar error。

### 執行評估

```powershell
python match_pairs.py `
  --input_pairs Bullpen_CalibrationBar/pairs_with_gt.txt `
  --input_dir Bullpen_CalibrationBar `
  --output_dir Bullpen_CalibrationBar/output_eval `
  --superglue outdoor `
  --eval --viz --fast_viz `
  --resize 640 480
```

終端機會印出評估結果：

```
Evaluation Results (mean over 21 pairs):
AUC@5    AUC@10  AUC@20  Prec    MScore
XX.XX    XX.XX   XX.XX   XX.XX   XX.XX
```

評估模式的視覺化顏色意義不同：
- **綠色**：對極誤差小（inlier，正確匹配）
- **紅色**：對極誤差大（outlier，錯誤匹配）

---

## 九、即時 Demo（`demo_superglue.py`）

如果想用 webcam 或影像資料夾做即時匹配展示：

```powershell
# 用 webcam（預設 ID=0）
.venv/Scripts/python.exe demo_superglue.py

# 用影像目錄（不顯示視窗，輸出到資料夾）
.venv/Scripts/python.exe demo_superglue.py --input assets/freiburg_sequence/ --output_dir dump_demo --resize 320 240 --no_display
```

鍵盤控制（需要顯示視窗）：

| 按鍵 | 功能 |
|------|------|
| `n` | 設定當前影像為錨點（anchor） |
| `e` / `r` | 提高 / 降低關鍵點偵測閾值 |
| `d` / `f` | 提高 / 降低匹配過濾閾值 |
| `k` | 切換顯示所有關鍵點 |
| `q` | 離開 |

---

## 十、常見問題

**Q：跑起來但匹配數量很少（< 10）怎麼辦？**
- 試試 `--max_keypoints 2048`（增加偵測點數）
- 試試 `--match_threshold 0.1`（放寬接受閾值）
- 試試 `--resize 1280`（提高解析度）
- 確認你選對模型：戶外場景用 `--superglue outdoor`

**Q：`indoor` 和 `outdoor` 模型怎麼選？**
- `indoor`：室內場景（ScanNet 訓練），適合走廊、辦公室等
- `outdoor`：戶外場景（MegaDepth 訓練），適合建築、地標、開放空間

你的 Bullpen 是戶外棒球場 → 用 `--superglue outdoor`

**Q：`--resize` 要設多少？**
- 預設 `640 480` 是平衡速度與品質的起點
- 匹配點少時可以試 `--resize 1280` 或 `--resize -1`（原圖）
- 建議範圍：160×120 ~ 2000×1500

**Q：輸出的 `.npz` 怎麼用？**
- 見第七節，可以拿到所有匹配點的像素座標和信心分數
- 後續可用來做：pose estimation、homography estimation、三維重建等

**Q：能不能不用 SuperPoint，用自己的關鍵點？**
- 需要修改 `models/matching.py`，可以直接傳入已有的 keypoints 和 descriptors 跳過 SuperPoint
- 這是後續客製化的方向之一
