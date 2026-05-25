# SuperGlueGnn 專案說明（Claude 工作記憶）

> 此文件是給 Claude 看的工作記憶，用來在對話壓縮或新對話開啟時快速還原進度。
> 每次有重要進展、檔案新增、目標變更，都必須更新此文件。

---

## 專案背景（來自 PPT：slide_01~23，路徑 C:\Users\user\Downloads\ppt_slides\）

**論文題目**：Four-Phases Segmentation for Bullpen Pitching Motion Analysis
**學生**：施邑穎　**指導教授**：連震杰　**單位**：Robotics Lab, CSIE NCKU　**日期**：2026/5/22

### 最終目標
投球動作 4 階段切割，提取三項物理特徵：
1. 肩髖分離角度（Shoulder-Hip Separation Angle）
2. 肩最大外旋角（Max External Rotation, MER）
3. 右手腕速度 / 出手點距離

三個關鍵幀：Foot Contact → Shoulder MER → Ball Release

### 系統 Pipeline（PPT slide_05~07）
```
影像 (CF+CS) 180fps → 60fps → ViT-Pose → 2D Skeleton
    ↓
Fundamental Matrix F（RANSAC 8-point）
    ↓ K 矩陣
Essential Matrix E = K_cs^T · F · K_cf
    ↓ SVD
R, t（相機外參）
    ↓
Projection Matrix:
  CS: PS = K̃_s · [I|0]
  CF: PF = K̃_f · [R|t]
    ↓
cv2.triangulatePoints(PS, PF, p_s, p_f) → 3D 骨架節點 P^w (X,Y,Z)
    ↓
物理特徵計算 → 4 階段切割
```

### 世界座標原點（PPT slide_07, slide_11）
**cs（投手區相機，side camera）為世界座標原點**。
- cs 的 Projection Matrix = K̃_s · [I|0]（cs 自身是 [I|0]）
- cf 的 Projection Matrix = K̃_f · [R|t]（R,t 是 cf 相對 cs 的外參）
- 三角測量輸出的 3D 座標是以 cs 為原點的世界座標

### F matrix 計算方式（PPT slide_16~17）
- 輸入：cs 影像的點 p_s（2D）, cf 影像的點 p_f（2D）
- RANSAC 8-point algorithm
- **使用 Hartley normalization**（座標先 normalize 再算 F）
- 輸出 F 滿足：p_s^T · F · p_f = 0（cs 為左，cf 為右）
- 每 iteration 隨機選 8 點，計算 inlier（Sampson distance < threshold）

### 解析度（PPT slide_13, slide_18）
- 相機最大解析度：1920×1084（Real），180fps
- **F matrix 估計用的是原始影像尺寸 1920×1084**（slide_18 明確說明 1920×1084 下算 F）
- ViT-Pose 輸入：slide_18 中 resize 為固定大小（內部 resize，輸出骨架點座標對應原圖 1920×1084）
- 因此三角測量所用的 2D 點座標是**1920×1084 原圖座標系**

### 相機硬體（PPT slide_12~13）
| 角色 | 相機型號 | 鏡頭 | 焦距 |
|------|---------|------|------|
| c1, c2（side / cs） | FLIR GS3-U3-23S6C-C | KOWA LM8HC | 8mm |
| c3（front / cf） | FLIR GS3-U3-23S6C-C | KOWA LM50HC | 50mm |
- Pixel size: 5.86 μm
- 場景配置（slide_12）：相機距投手約 2m（側面）/ 投手與捕手距離 18.44m（本壘板前）

---

## 專案目標

**最終目標**：用 RoMa（現階段最佳方案）的 Direct E → R,t 提供給開發人員做三角測量與 3D 骨架重建，驗證是否可用於投球動作分析。

目前已確認：
- SuperGlue F→E 路線：❌ 完全失敗（planar degeneracy + K_cf 條件數問題）
- LoFTR Direct E：✅ Pitch 正確（−82°）但單對點數太少（9 inlier），跨對不穩定
- **RoMa V1 B_hires（upsample_res=1024）Direct E：✅ 當前最佳，Pitch mean=−79.0°, std=1.2°, E-inliers~2338**
- RoMa V2（DINOv3）：mean=−79.5°, std=2.3°，mean 更好但穩定性不如 V1 B_hires

---

## 目標清單

### 階段一：建立基礎匹配流程 ✅ 已完成
### 階段二：F matrix Pipeline ✅ 已完成（2026-05-14~15）
### 階段三：Essential Matrix → Pose Estimation（SuperGlue F→E）❌ 失敗
### 階段四：換用 Dense Matching ✅ RoMa 成功
- [x] LoFTR 測試（Pitch=−82° 正確，但點數少，跨對不穩）
- [x] LightGlue 測試（完全失效）
- [x] RoMa V1 全 20 對穩定性測試（mean=−78.6°, std=2.1°）
- [x] RoMa V1 4 種參數優化（最佳：upsample_res=1024, std=1.2°）
- [x] RoMa V2 全 20 對測試（mean=−79.5°, std=2.3°）

### 階段五：待完成
- [ ] 測試 RoMa V1 upsample_res=1280（更高解析度，預計 std 再降）
- [ ] 輸出 RoMa Direct E 矩陣給開發人員測試 3D 骨架重建
- [ ] 輸出 LoFTR Direct E 矩陣給開發人員測試（對比用）

---

## 關鍵測試結果總覽

| 方法 | Pitch mean | std | E-inliers mean | 狀態 |
|------|-----------|-----|---------------|------|
| 人工點 Direct E（全20對，120pts） | −82.0° | — | 60 | ✅ 基準 |
| LoFTR conf≥0.5（pair08） | −82.0° | — | 9 | ✅ 單對正確 |
| RoMa V1 A_baseline（864, N=5000, conf=0.5） | −78.3° | 2.4° | 2325 | ✅ |
| **RoMa V1 B_hires（1024, N=5000, conf=0.5）** | **−79.0°** | **1.2°** | **2338** | ✅ 最佳穩定 |
| RoMa V1 C_moresamples（864, N=8000, conf=0.5） | −78.6° | 2.3° | 3633 | ✅ |
| RoMa V1 D_lowconf（864, N=5000, conf=0.3） | −78.3° | 2.3° | 2294 | ✅ |
| RoMa V2（DINOv3） | −79.5° | 2.3° | 2446 | ✅ mean 更接近 |
| SuperGlue（pair08，對照） | −41.3° | — | 58 | ❌ |

---

## 相機內參

| 相機 | 焦距 fx（pixel） | 焦距 fy | cx | cy | 實體焦距 |
|------|------|------|------|------|------|
| cf (Camera Front / c3) | 7927.87 | 8039.59 | 869.87 | 716.81 | 50mm（KOWA LM50HC） |
| cs (Camera Side / c1,c2) | 1405.84 | 1406.62 | 964.22 | 575.70 | 8mm（KOWA LM8HC） |

- cf 主點偏移：cx=869.87（理論中心 960），偏移 90px，影響 E=K₂ᵀFK₁ 精度
- 焦距比：7927/1405 ≈ 5.6 倍（像素焦距），實體焦距 50/8 = 6.25 倍
- cf 條件數：8165（數值病態，F→E 轉換誤差大）

---

## 當前專案狀態

### 環境
- 虛擬環境：`.venv/`（uv 建立，Python 3.11）
- 啟動方式：`.venv/Scripts/python.exe` 或先執行 `.venv/Scripts/Activate.ps1`
- GPU：NVIDIA RTX 5080，CUDA 12.8，自動使用

### 已建立的重要檔案
| 檔案 | 說明 | 狀態 |
|------|------|------|
| `Bullpen_Calibration/TSG_Bullpen/pairs.txt` | 20 對影像配對清單 | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/output/` | 1280×960 匹配結果 | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/fundamental matrix.txt` | GT F（人工標記，22 inliers） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/selected_points_cf.json` | 人工標記點（cf） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/selected_points_cs.json` | 人工標記點（cs） | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_A_sg_all/` | 組A視覺化+F_matrix.txt | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_B_sg_manual_all/` | 組B視覺化+F_matrix.txt | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_C_sg_best/` | 組C視覺化+F_matrix.txt | ✅ |
| `Bullpen_Calibration/TSG_Bullpen/viz_D_sg_manual_best/` | 組D視覺化+F_matrix.txt | ✅ |
| `Bullpen_Calibration/Intrinsic/Cf_Intrinsic.txt` | cf 相機內參 K | ✅ |
| `Bullpen_Calibration/Intrinsic/Cs_Intrinsic.txt` | cs 相機內參 K | ✅ |
| `compute_fundamental.py` | F matrix pipeline | ✅ |
| `visualize_inliers.py` | inlier 連線 + epipolar 視覺化 | ✅ |
| `visualize_four_groups.py` | 四組 F matrix 視覺化 | ✅ |
| `test_loftr.py` | LoFTR 測試腳本 | ✅ |
| `test_roma.py` | RoMa V1 單對測試 | ✅ |
| `test_roma_all_pairs.py` | RoMa V1 全 20 對測試 | ✅ |
| `test_roma_tuned.py` | RoMa V1 4 種參數優化測試 | ✅ |
| `test_romav2.py` | RoMa V2 全 20 對測試 | ✅ |
| `spec.html` | HTML 規格書（QA Q1~Q17） | ✅ |
| `README_zh.md` | 中文操作手冊 | ✅ |

### PPT 截圖路徑
`C:\Users\user\Downloads\ppt_slides\slide_01.png` ~ `slide_23.png`

---

## 資料集結構

```
Bullpen_Calibration/
├── Intrinsic/
│   ├── Cf_Intrinsic.txt      # cf 相機內參（50mm）
│   └── Cs_Intrinsic.txt      # cs 相機內參（8mm）
├── TSG_Bullpen/              # 台糖牛棚場景（主要資料集）
│   ├── cf/                   # Camera Front 影像（01.jpg ~ 20.jpg）
│   ├── cs/                   # Camera Side 影像（01.jpg ~ 20.jpg）
│   ├── pairs.txt
│   ├── output/               # 1280×960 匹配結果
│   ├── fundamental matrix.txt
│   ├── selected_points_cf.json
│   ├── selected_points_cs.json
│   ├── viz_A_sg_all/
│   ├── viz_B_sg_manual_all/
│   ├── viz_C_sg_best/
│   └── viz_D_sg_manual_best/
└── Ncku_Bullpen/             # 成大牛棚場景（待處理）
```

---

## 版本控制紀錄

| 日期 | 說明 |
|------|------|
| 2026-05-09 | 初始化專案 |
| 2026-05-10 | 建立 uv 虛擬環境、README_zh.md、spec.html |
| 2026-05-14 | F matrix pipeline；Frobenius norm 0.045（640×480） |
| 2026-05-15 | 視覺化腳本；1920×1084 結果 Frobenius norm 0.021 |
| 2026-05-16 | 更新資料夾結構 |
| 2026-05-22 | 四組 F matrix 實驗全部失敗；確認根本原因 K_cf 條件數 8165 |
| 2026-05-24 | LoFTR/LightGlue/RoMa V1/V2 測試完成；RoMa V1 B_hires 為最佳（std=1.2°）；spec.html 更新至 Q17；CLAUDE.md 全面更新含 PPT 關鍵資訊 |

---

## 待辦事項（下次對話繼續）

1. 測試 RoMa V1 upsample_res=1280
2. 輸出 RoMa B_hires 的 E matrix（3×3 數值）供開發人員使用
3. 輸出 LoFTR 的 E matrix 供對比
4. 等開發人員跑完 3D 骨架重建後回報結果
