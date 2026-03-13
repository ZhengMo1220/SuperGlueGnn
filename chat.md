# SuperGlue 對話重點整理

## 1. 線條顏色的意義

### 一般匹配模式（`--viz`）
- 顏色來源：`cm.jet(mconf)`（matplotlib 內建 jet colormap）
- 依據：**匹配置信度（matching confidence）**
- 顏色對應：藍（低）→ 青→ 綠→ 黃→ 紅（高）
- **越紅 = SuperGlue 對這對匹配越有把握**

### 評估模式（`--eval --viz`）
- 顏色來源：`error_colormap(1 - color)`（自訂函式，位於 `models/utils.py`）
- 依據：**對極幾何誤差（epipolar error）**
- 顏色對應：**綠（誤差小 / inlier）→ 紅（誤差大 / outlier）**
- 函式實作：`R = 2-2x, G = 2x`（RGBA，裁切到 0~1）

---

## 2. Inlier / Outlier 判定方式

### 方法一：Epipolar Error 閾值（評估統計用）
```python
epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
correct = epi_errs < 5e-4   # inlier 判定閾值
```
- 用來計算 `num_correct`、`precision`、`matching_score`
- 視覺化：`color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)` → `error_colormap(1 - color)`

### 方法二：RANSAC（pose recovery 用）
```python
E, mask = cv2.findEssentialMat(..., method=cv2.RANSAC)
n, R, t, _ = cv2.recoverPose(_E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
```
- mask 即為 RANSAC 判定的 inlier

---

## 3. 使用自己資料集評估

### 需要的 GT 資料
要使用 `match_pairs.py --eval`，每對影像需要：
1. **相機內參 K**（3×3 矩陣，9 個數值）
2. **相對姿態 T_0to1**（4×4 矩陣，16 個數值）

### pairs.txt 格式（每行 38 欄）
```
name0 name1 rot0 rot1  K0(9值)  K1(9值)  T_0to1(16值)
```

### 執行指令
```bash
python match_pairs.py \
  --input_pairs path/to/pairs.txt \
  --input_dir path/to/images \
  --output_dir dump_match_pairs \
  --eval --viz --fast_viz
```

---

## 4. 如何取得 GT（intrinsics & relative pose）

### 相機內參（intrinsics K）— 張氏演算法
- 用棋盤格拍多角度照片（≥10 張）
- 用 OpenCV `calibrateCamera()` 取得 K 與 distortion
- **可視為 GT**，但需確認 reprojection error < 0.5~1 px

### 相對姿態（relative pose T_0to1）
- 使用已知 3D 座標標記物（棒子+球）+ `cv2.solvePnP()`：
```python
ret, rvec, tvec = cv2.solvePnP(points_3d, points2d, K, dist)
R, _ = cv2.Rodrigues(rvec)
cam_T_w[:3,:3] = R;  cam_T_w[:3,3] = tvec.ravel()
T_0to1 = cam1_T_w @ np.linalg.inv(cam0_T_w)
```
- **可視為 GT**，前提：3D 座標測量精確、2D 標記點精準

### 張氏法 / solvePnP 是否算 GT？
| 方法 | 是否算 GT | 條件 |
|------|-----------|------|
| 張氏法（棋盤格）校正 K | **是** | reprojection error 小，多角度資料充足 |
| solvePnP + 已知 3D 標記求 pose | **是** | 3D 點測量精確，重投影誤差小 |
| COLMAP / SfM 自動估計 | **近似 GT** | 自估尺度不確定，需 bundle adjustment 優化 |

### 驗證方式
1. 計算每張影像的 reprojection error（< 1 px 為佳）
2. 比較多組照片估出的 T_0to1 是否穩定（低方差）

---

## 5. 注意事項
- `match_pairs.py` 會依 `--resize` 自動用 `scale_intrinsics()` 縮放 K，輸入原始解析度的 K 即可。
- 建議先做 `cv2.undistort()` 去除畸變，再做特徵匹配，否則 epipolar error 會變大。
- 若影像來自同一台相機、同樣設定，K0 = K1 相同即可。
