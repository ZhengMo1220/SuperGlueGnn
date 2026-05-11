# SuperGlueGnn 專案說明（Claude 工作記憶）

> 此文件是給 Claude 看的工作記憶，用來在對話壓縮或新對話開啟時快速還原進度。
> 每次有重要進展、檔案新增、目標變更，都必須更新此文件。

---

## 專案目標

**最終目標**：利用 SuperGlue 完成棒球牛棚場景的 **Pose Estimation**。

具體做法：
- 場景中擺放兩根棒子，棒子上黏有白色保麗龍球作為標定點
- 用兩台固定相機（cf = Camera Front、cs = Camera Side）同時拍攝
- 透過 SuperGlue 找出兩個視角之間的對應特徵點（corresponding points）
- 以此完成相機之間的相對姿態估計（Pose Estimation）

目前人工校正流程（尚未自動化）：
- 在影像上人工點選白色保麗龍球作為特徵點
- 透過點選的 2D 對應點 + 已知 3D 座標做 PnP 求解姿態

---

## 目標清單（待完成）

### 階段一：建立基礎匹配流程 ✅ 已完成
- [x] 確認專案結構與程式碼
- [x] 建立 uv 虛擬環境（`.venv`，Python 3.11，PyTorch 2.11+cu128）
- [x] 生成 `Bullpen_CalibrationBar/pairs.txt`（21 對，cf/N.jpg cs/N.jpg 格式）
- [x] 驗證 match_pairs.py 可正常執行（輸出 .npz + .png）
- [x] 撰寫中文操作手冊（README_zh.md）
- [x] 製作 HTML 規格書（spec.html）
- [x] 更新 CLAUDE.md 工作記憶

### 階段二：取得相機校正資料（進行中）
- [ ] 取得 cf 相機內參 K0（張氏棋盤格法，reprojection error < 1px）
- [ ] 取得 cs 相機內參 K1（同上）
- [ ] 對每對影像取得相對姿態 T_0to1（solvePnP + 已知 3D 標記點座標）
- [ ] 生成 38 欄版本的 pairs_with_gt.txt
- [ ] 執行 `--eval` 模式，取得 AUC / Precision / Matching Score 定量評估

### 階段三：驗證與優化
- [ ] 分析 SuperGlue 匹配結果品質（觀察 output/ 中的 .png）
- [ ] 調整超參數（max_keypoints / match_threshold / resize）以提升匹配數量
- [ ] 決定是否需要先做 undistort 再送入匹配
- [ ] 比較 indoor vs outdoor 模型效果

### 階段四：Pose Estimation 整合
- [ ] 從 .npz 讀取匹配點對，整合進姿態估計 pipeline
- [ ] 結合保麗龍球標定點的 3D 座標，完成完整 Pose Estimation
- [ ] 評估姿態估計精度

---

## 當前專案狀態

### 環境
- 虛擬環境：`.venv/`（uv 建立，Python 3.11）
- 啟動方式：`.venv/Scripts/python.exe` 或先執行 `.venv/Scripts/Activate.ps1`
- GPU：NVIDIA RTX 5080，CUDA 12.8，自動使用

### 已建立的重要檔案
| 檔案 | 說明 | 狀態 |
|------|------|------|
| `Bullpen_CalibrationBar/pairs.txt` | 21 對影像配對清單（2 欄，無 GT） | ✅ |
| `Bullpen_CalibrationBar/output/` | 匹配結果輸出目錄 | ✅（跑過 2 對測試）|
| `README_zh.md` | 中文操作手冊 | ✅ |
| `spec.html` | HTML 規格書（瀏覽器或 VSCode Live Preview 開啟）| ✅ |
| `CLAUDE.md` | 本文件 | ✅ |

### 資料集
- `Bullpen_CalibrationBar/cf/`：Camera Front 影像，21 張（1.jpg ~ 21.jpg）
- `Bullpen_CalibrationBar/cs/`：Camera Side 影像，21 張（1.jpg ~ 21.jpg）
- 場景：戶外棒球牛棚，棒子 + 白色保麗龍球作為標定點
- 模型選擇：`--superglue outdoor`（戶外場景）

---

## 目錄結構

```
SuperGlueGnn/
├── models/
│   ├── superglue.py          # 核心 GNN 匹配網路
│   ├── superpoint.py         # 關鍵點偵測與描述子提取
│   ├── matching.py           # SuperPoint + SuperGlue 整合推理 pipeline
│   ├── utils.py              # I/O、視覺化、評估工具函式
│   └── weights/
│       ├── superpoint_v1.pth
│       ├── superglue_indoor.pth
│       └── superglue_outdoor.pth
├── assets/                   # 官方範例資料集
├── Bullpen_CalibrationBar/   # 【自訂資料集】棒球場標定棒
│   ├── cf/                   # Camera Front（21 張）
│   ├── cs/                   # Camera Side（21 張）
│   ├── pairs.txt             # 配對清單（2 欄，已生成）
│   └── output/               # 匹配輸出（.npz + .png）
├── demo_superglue.py
├── match_pairs.py            # 主程式
├── superglue_viewer.py       # PyQt5 結果瀏覽 UI
├── README_zh.md              # 中文操作手冊
├── spec.html                 # HTML 規格書
├── CLAUDE.md                 # 本文件（工作記憶）
├── .venv/                    # uv 虛擬環境
└── requirements.txt
```

---

## 核心模型架構

### 推理 Pipeline
```
影像對 (cf, cs)
    │
    ▼
SuperPoint（每張各跑一次）
├── 輸出：keypoints [N×2], scores [N], descriptors [N×256]
    │
    ▼
SuperGlue（GNN 匹配）
├── Keypoint Encoding：座標 + 分數 → 256 維特徵融合
├── Attentional GNN：18 層交替 self/cross-attention
├── Optimal Transport（Sinkhorn 算法）
└── 閾值過濾（match_threshold=0.2）
    │
    ▼
輸出：matches [M×2], confidence [M]
```

### 關鍵超參數（目前使用值）
| 參數 | 目前值 | 說明 |
|------|--------|------|
| `--superglue` | `outdoor` | 戶外場景模型 |
| `--max_keypoints` | `1024` | 每張圖最多關鍵點數 |
| `--match_threshold` | `0.2` | 匹配信心度閾值 |
| `--nms_radius` | `4` | 關鍵點 NMS 半徑 |
| `--keypoint_threshold` | `0.005` | 關鍵點分數閾值 |
| `--resize` | `640 480` | 輸入影像尺寸 |

---

## 常用指令（複製即用）

```powershell
# 啟動環境
cd "c:\Mo\program\SuperGlueGnn"
.venv/Scripts/Activate.ps1

# 快速測試（2 對）
python match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output --superglue outdoor --viz --fast_viz --resize 640 480 --max_length 2

# 全部 21 對
python match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output --superglue outdoor --viz --fast_viz --resize 640 480

# 高品質版（更多關鍵點）
python match_pairs.py --input_pairs Bullpen_CalibrationBar/pairs.txt --input_dir Bullpen_CalibrationBar --output_dir Bullpen_CalibrationBar/output_hires --superglue outdoor --max_keypoints 2048 --nms_radius 3 --resize_float --viz --fast_viz --resize 1280

# 查看結果
python superglue_viewer.py
```

---

## pairs.txt 格式說明

### 目前（2 欄，無 GT）
```
cf/1.jpg cs/1.jpg
cf/2.jpg cs/2.jpg
...
```

### 未來需要（38 欄，含 GT，啟用 --eval）
```
name0 name1 rot0 rot1  K0(9值)  K1(9值)  T_0to1(16值)
```
- K：3×3 內參矩陣逐行展平成 9 個數值
- T_0to1：4×4 相對姿態矩陣逐行展平成 16 個數值

---

## 評估指標（--eval 模式）
| 指標 | 說明 |
|------|------|
| AUC@5/10/20 | 姿態誤差在 N° 以內的比例曲線下面積 |
| Precision | 對極誤差 < 5e-4 的匹配比例 |
| Matching Score | 正確匹配數 / img0 關鍵點總數 |

---

## 視覺化顏色說明
- **一般匹配模式**（`--viz`）：藍（低信心）→ 紅（高信心）
- **評估模式**（`--eval --viz`）：綠（inlier，誤差小）→ 紅（outlier，誤差大）

---

## 版本控制紀錄

| 日期 | 說明 |
|------|------|
| 2026-05-09 | 新增 Bullpen_CalibrationBar 資料集、superglue_viewer.py、chat.md |
| 2026-05-10 | 建立 uv 虛擬環境（.venv）、生成 pairs.txt、撰寫 README_zh.md、製作 spec.html、更新 CLAUDE.md |

---

## 相關文件
- `README_zh.md`：中文操作手冊（給人看）
- `spec.html`：HTML 規格書，含所有檔案規格與參數說明（VSCode Live Preview 或瀏覽器開啟）
- `chat.md`：早期技術對話筆記（epipolar error、GT 格式等）
