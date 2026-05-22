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

目前規劃方向：
- SuperGlue 自動找 corresponding points → RANSAC → F matrix → Essential Matrix → R, t
- 已有相機內參 K（Bullpen_Calibration/Intrinsic/），可直接走 Essential Matrix 路線

---

## 目標清單

### 階段一：建立基礎匹配流程 ✅ 已完成
- [x] 確認專案結構與程式碼
- [x] 建立 uv 虛擬環境（`.venv`，Python 3.11，PyTorch 2.11+cu128）
- [x] 驗證 match_pairs.py 可正常執行（輸出 .npz + .png）
- [x] 撰寫中文操作手冊（README_zh.md）
- [x] 製作 HTML 規格書（spec.html）

### 階段二：F matrix Pipeline ✅ 已完成（2026-05-14~15）
- [x] 生成 `TSG_Bullpen/pairs.txt`（20 對，格式：`cf/01.jpg cs/01.jpg`）
- [x] 執行 SuperGlue 匹配（1920×1084 原圖解析度，4096 keypoints，threshold 0.1）
- [x] 寫後處理腳本 `compute_fundamental.py`（支援 --dataset_dir / --output_dir）
- [x] 驗證模式（`--verify_gt`）確認 pipeline 正確（Frobenius norm 0.006）
- [x] 評估 SuperGlue F matrix 品質（Frobenius norm 0.021 vs GT，非常好）
- [x] 視覺化腳本 `visualize_inliers.py`（inlier 連線圖 + epipolar line 驗證圖）

### 階段三：Essential Matrix → Pose Estimation ❌ 失敗（2026-05-22 確認瓶頸）
- [x] 用 F matrix + 內參 K 換算 Essential Matrix（E = K_cs^T F K_cf）
- [x] 從 E 分解 R, t（cv2.recoverPose）→ R ≈ 160°，應為 ~90°，失敗
- [x] 四組 F matrix 實驗（A/B/C/D）→ 全部 3D 骨架重建失敗（躺平或壓成線）
- [x] 根本原因確認：K_cf 條件數 8165，F→E 轉換數值病態，非特徵匹配問題

### 階段四：下一步方向（待決定）
- [ ] 方案 1：ChArUco board 直接標定（cv2.solvePnP，繞過 F/E 矩陣）
- [ ] 方案 2：評估換新 matching 論文（LoFTR 等）是否有助於改善

---

## Pipeline 執行結果總覽（TSG_Bullpen）

| 指標 | SG 640×480 | SG 1280×960 | **SG 1920×1084（原圖）** | GT（人工） |
|------|------|------|------|------|
| 總點數 | 396 | 1701 | 1358 | 120 |
| RANSAC inliers | 67 (16.9%) | 310 (18.2%) | **290 (21.4%)** | 22 (18.3%) |
| Frobenius norm vs GT | 0.045 | 0.135 | **0.021** | — |

**使用 1920×1084（原圖解析度）結果最佳**，與 GT F 幾何誤差只有 0.021。

---

## 關鍵技術決策

- **不需要相機內參** 來算 Fundamental Matrix（F 是純影像幾何）
- 有內參後改走 **Essential Matrix 路線**更精確（E = K2^T F K1）
- SuperGlue 偵測不到保麗龍球（表面無紋理），匹配來自背景紋理（草皮、圍網、棒子邊緣）
- 保麗龍球的用途是 PnP（已知 3D 座標），可提供絕對尺度；純場景只能得到相對 R 和方向性 t

---

## 當前專案狀態

### 環境
- 虛擬環境：`.venv/`（uv 建立，Python 3.11）
- 啟動方式：`.venv/Scripts/python.exe` 或先執行 `.venv/Scripts/Activate.ps1`
- GPU：NVIDIA RTX 5080，CUDA 12.8，自動使用

### 相機內參
| 相機 | 焦距 fx | 焦距 fy | cx | cy | 檔案 |
|------|------|------|------|------|------|
| cf (Camera Front) | 7927.87 | 8039.59 | 869.87 | 716.81 | `Bullpen_Calibration/Intrinsic/Cf_Intrinsic.txt` |
| cs (Camera Side) | 1405.84 | 1406.62 | 964.22 | 575.70 | `Bullpen_Calibration/Intrinsic/Cs_Intrinsic.txt` |

### 已建立的重要檔案
| 檔案 | 說明 | 狀態 |
|------|------|------|
| `Bullpen_Calibration/TSG_Bullpen/pairs.txt` | 20 對影像配對清單 | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/output_1920/` | 1920×1084 匹配結果（.npz + .png） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/output/` | 1280×960 匹配結果（備用） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/superglue_F_output_1920.txt` | SuperGlue F（1920，**主要使用**） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/superglue_F_output.txt` | SuperGlue F（1280，備用） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/fundamental matrix.txt` | GT F（人工標記，22 inliers） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/inlier_viz/` | inlier 連線圖 + epipolar 驗證圖 | ✅ |
| `Bullpen_Calibration/Intrinsic/Cf_Intrinsic.txt` | cf 相機內參 K | ✅ |
| `Bullpen_Calibration/Intrinsic/Cs_Intrinsic.txt` | cs 相機內參 K | ✅ |
| `compute_fundamental.py` | F matrix pipeline（支援 --dataset_dir / --output_dir，輸出 cf-first convention） | ✅ |
| `compute_pose.py` | E matrix → recoverPose → R,t（已確認失敗，K_cf 條件數問題） | ✅ |
| `visualize_inliers.py` | inlier 連線 + epipolar 視覺化 | ✅ |
| `visualize_four_groups.py` | 四組 F matrix 視覺化（A/B/C/D，各自獨立資料夾） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_A_sg_all/` | 組 A 視覺化 + F_matrix.txt（SG 全20對） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_B_sg_manual_all/` | 組 B 視覺化 + F_matrix.txt（SG+人工全20對） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_C_sg_best/` | 組 C 視覺化 + F_matrix.txt（SG pair08） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_D_sg_manual_best/` | 組 D 視覺化 + F_matrix.txt（SG+人工 pair08） | ✅ |
| `README_zh.md` | 中文操作手冊 | ✅ |
| `spec.html` | HTML 規格書（含 Q1~Q15 QA） | ✅ |

### 資料集結構
```
Bullpen_Calibration/
├── Intrinsic/
│   ├── Cf_Intrinsic.txt      # cf 相機內參
│   └── Cs_Intrinsic.txt      # cs 相機內參
├── TSG_Bullpen/              # 台糖牛棚場景（主要資料集）
│   ├── cf/                   # Camera Front 影像（01.jpg ~ 20.jpg）
│   ├── cs/                   # Camera Side 影像（01.jpg ~ 20.jpg）
│   ├── pairs.txt             # 配對清單
│   ├── output_1920/          # 1920×1084 匹配結果（主要使用）
│   ├── output/               # 1280×960 匹配結果（備用）
│   ├── inlier_viz/           # 視覺化圖片
│   ├── fundamental matrix.txt  # GT F（人工標記）
│   ├── superglue_F_output_1920.txt  # SuperGlue F（1920，主要）
│   ├── superglue_F_output.txt       # SuperGlue F（1280，備用）
│   ├── selected_points_cf.json      # 人工標記點（cf）
│   └── selected_points_cs.json      # 人工標記點（cs）
└── Ncku_Bullpen/             # 成大牛棚場景（待處理）
    ├── cf/
    ├── cs/
    ├── pairs.txt
    └── output/
```

---

## 目錄結構

```
SuperGlueGnn/
├── models/
│   ├── superglue.py
│   ├── superpoint.py
│   ├── matching.py
│   ├── utils.py
│   └── weights/
├── assets/
├── Bullpen_Calibration/      # 【自訂資料集】棒球牛棚
├── Volleyball_Calibration/   # 排球場景（待處理）
├── match_pairs.py            # 主程式
├── compute_fundamental.py    # F matrix pipeline
├── visualize_inliers.py      # 視覺化腳本
├── superglue_viewer.py       # PyQt5 結果瀏覽 UI
├── README_zh.md
├── spec.html
├── CLAUDE.md
└── .venv/
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
└── 閾值過濾（match_threshold=0.1）
    │
    ▼
輸出：matches [M×2], confidence [M]
    │
    ▼
RANSAC + 8-point → F matrix → E matrix → R, t
```

### 關鍵超參數（TSG_Bullpen 最佳設定）
| 參數 | 值 | 說明 |
|------|--------|------|
| `--superglue` | `outdoor` | 戶外場景模型 |
| `--max_keypoints` | `4096` | 每張圖最多關鍵點數 |
| `--match_threshold` | `0.1` | 匹配信心度閾值（放寬後 RANSAC 過濾） |
| `--resize` | `1920 1084` | 原圖解析度（與 GT F 座標系一致） |

---

## 常用指令（複製即用）

```powershell
# 啟動環境
cd "c:\Mo\program\SuperGlueGnn"
.venv/Scripts/Activate.ps1

# SuperGlue 匹配（TSG_Bullpen，原圖解析度）
.venv/Scripts/python.exe match_pairs.py `
    --input_pairs Bullpen_Calibration/TSG_Bullpen/pairs.txt `
    --input_dir Bullpen_Calibration/TSG_Bullpen `
    --output_dir Bullpen_Calibration/TSG_Bullpen/output_1920 `
    --superglue outdoor --max_keypoints 4096 --match_threshold 0.1 `
    --resize 1920 1084 --viz --fast_viz

# F matrix 評估（1920 解析度）
.venv/Scripts/python.exe compute_fundamental.py `
    --dataset_dir Bullpen_Calibration/TSG_Bullpen `
    --output_dir Bullpen_Calibration/TSG_Bullpen/output_1920 `
    --save_F

# 視覺化 inlier 連線（單張）
.venv/Scripts/python.exe visualize_inliers.py

# 視覺化 epipolar line 驗證
.venv/Scripts/python.exe visualize_inliers.py --epipolar
```

---

## 版本控制紀錄

| 日期 | 說明 |
|------|------|
| 2026-05-09 | 初始化專案、superglue_viewer.py |
| 2026-05-10 | 建立 uv 虛擬環境、撰寫 README_zh.md、製作 spec.html |
| 2026-05-14 | 建立 F matrix pipeline（compute_fundamental.py）；Frobenius norm 0.045（640×480） |
| 2026-05-15 | 視覺化腳本（visualize_inliers.py）；1920×1084 結果 Frobenius norm 0.021 |
| 2026-05-16 | 更新 CLAUDE.md 反映新資料夾結構（Bullpen_Calibration/TSG_Bullpen）；新增內參資訊 |
| 2026-05-22 | 四組 F matrix 實驗（A/B/C/D）3D 骨架重建全部失敗；確認根本原因為 K_cf 條件數 8165 導致 F→E 數值病態；修正 F matrix 輸出為 cf-first convention（.T）；新增 visualize_four_groups.py；更新 spec.html QA Q13/Q14 |

---

## Future Work（備選改進方案）

當 SuperGlue pipeline 品質到頂後，依優先順序考慮以下替代方案。詳細比較見 `spec.html` 第十五節。

**2026-05-22 更新：問題不在特徵匹配演算法，換論文無法解決根本問題。**

根本原因是 K_cf 條件數 8165（50mm 長焦），F→E 轉換數值病態。即使換 LoFTR 或 LightGlue 得到更好的 corresponding points，F→E→R,t 這條路仍然不穩定。

**建議優先方向：**

| 優先順序 | 方案 | 說明 |
|---------|------|------|
| 1 | **ChArUco board 直接標定** | 場景中放標定板，cv2.solvePnP 直接得 R,t，繞過 F/E，1~2 天可跑通 |
| 2 | **LoFTR** | CVPR 2021，dense matching，學術價值高，但無法解決 K_cf 條件數問題 |
| 3 | **LightGlue** | ICCV 2023，SuperGlue 原作者新作，同上，換匹配不換 pipeline |

---

## 相關文件
- `README_zh.md`：中文操作手冊（給人看）
- `spec.html`：HTML 規格書，含所有檔案規格與參數說明
- `chat.md`：早期技術對話筆記
