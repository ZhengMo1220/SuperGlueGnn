"""
compute_fundamental.py

Pipeline:
  1. Load SuperGlue .npz outputs (auto matched points)
  2. Run RANSAC + 8-point to estimate Fundamental Matrix
  3. Compare against GT F matrix (fundamental matrix.txt)
  4. [verify_gt mode] Reproduce GT F matrix from manual GT points first

Usage:
  # Step 1: run SuperGlue matching
  .venv/Scripts/python.exe match_pairs.py ^
      --input_pairs Bullpen_CalibrationBar/Calibrate_Picture/pairs.txt ^
      --input_dir Bullpen_CalibrationBar/Calibrate_Picture ^
      --output_dir Bullpen_CalibrationBar/Calibrate_Picture/output ^
      --superglue outdoor --viz --fast_viz --resize 640 480

  # Step 2: run this script
  .venv/Scripts/python.exe compute_fundamental.py               # main eval
  .venv/Scripts/python.exe compute_fundamental.py --verify_gt   # sanity check first
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path


# ── 路徑設定 ──────────────────────────────────────────
DATASET_DIR = Path('Bullpen_CalibrationBar/Calibrate_Picture')
OUTPUT_DIR  = DATASET_DIR / 'output'
GT_F_FILE   = DATASET_DIR / 'fundamental matrix.txt'
GT_CF_JSON  = DATASET_DIR / 'selected_points_cf.json'
GT_CS_JSON  = DATASET_DIR / 'selected_points_cs.json'
IMAGE_NAMES = [f'{i:02d}.jpg' for i in range(1, 21)]


# ── 工具函式 ──────────────────────────────────────────

def load_gt_F():
    """讀取 fundamental matrix.txt，回傳 3x3 numpy array。"""
    with open(GT_F_FILE, 'r', encoding='utf-8') as f:
        text = f.read()

    # 取出矩陣數值那三行
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith('[')]
    rows = []
    for line in lines:
        clean = line.replace('[', '').replace(']', '').replace(',', ' ')
        rows.append([float(x) for x in clean.split()])
    F = np.array(rows)
    assert F.shape == (3, 3), f'GT F matrix shape 錯誤：{F.shape}'
    return F


def load_gt_points():
    """讀取人工標記的對應點，回傳所有圖的 (pts_cf, pts_cs)。"""
    with open(GT_CF_JSON, 'r') as f:
        cf_data = json.load(f)
    with open(GT_CS_JSON, 'r') as f:
        cs_data = json.load(f)

    all_cf, all_cs = [], []
    for name in IMAGE_NAMES:
        if name in cf_data and name in cs_data:
            all_cf.extend(cf_data[name])
            all_cs.extend(cs_data[name])

    return np.array(all_cf, dtype=np.float64), np.array(all_cs, dtype=np.float64)


def compute_F_from_points(pts0, pts1, ransac_thresh=3.0):
    """用 RANSAC + 8-point 算 Fundamental Matrix，回傳 (F, mask)。"""
    assert len(pts0) >= 8, f'點數不足（{len(pts0)}），至少需要 8 個點'
    F, mask = cv2.findFundamentalMat(
        pts0, pts1,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.99
    )
    return F, mask.ravel().astype(bool)


def sampson_distance(F, pts0, pts1):
    """
    計算每個匹配點對的 Sampson distance（對極幾何誤差的對稱近似）。
    值越小代表該點對越符合 F 描述的幾何關係。
    """
    pts0_h = np.hstack([pts0, np.ones((len(pts0), 1))])  # [N, 3]
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])  # [N, 3]

    Fp0  = (F  @ pts0_h.T).T   # [N, 3]
    FTp1 = (F.T @ pts1_h.T).T  # [N, 3]

    num = (pts1_h * Fp0).sum(axis=1) ** 2
    den = Fp0[:, 0]**2 + Fp0[:, 1]**2 + FTp1[:, 0]**2 + FTp1[:, 1]**2
    return num / den


def compare_F(F_est, F_gt, pts0, pts1, label=''):
    """Print comparison report between F_est and F_gt."""
    tag = f'[{label}] ' if label else ''

    err_est = sampson_distance(F_est, pts0, pts1)
    err_gt  = sampson_distance(F_gt,  pts0, pts1)

    print(f'\n{"-"*55}')
    print(f'{tag}Sampson distance (lower is better)')
    print(f'  F_estimated  mean={err_est.mean():.4f}  median={np.median(err_est):.4f}')
    print(f'  F_gt         mean={err_gt.mean():.4f}  median={np.median(err_gt):.4f}')

    def normalize(M):
        return M / np.linalg.norm(M)

    diff = np.linalg.norm(normalize(F_est) - normalize(F_gt))
    print(f'\n{tag}Normalized F matrix difference (Frobenius norm): {diff:.6f}')
    print(f'  (0 = identical, lower is better; < 0.1 is close)')
    print(f'{"-"*55}')


def load_superglue_matches(npz_path):
    """讀取 SuperGlue .npz，回傳有效匹配的點對座標。"""
    data = np.load(npz_path)
    kpts0   = data['keypoints0']
    kpts1   = data['keypoints1']
    matches = data['matches']
    conf    = data['match_confidence']

    valid    = matches > -1
    mkpts0   = kpts0[valid]
    mkpts1   = kpts1[matches[valid]]
    mconf    = conf[valid]
    return mkpts0, mkpts1, mconf


# ── 主程式 ────────────────────────────────────────────

def run_verify_gt():
    """Sanity check: reproduce F matrix from manual GT points."""
    print('\n=== Verify mode: reproduce F from manual GT points ===')

    F_gt = load_gt_F()
    pts_cf, pts_cs = load_gt_points()
    print(f'GT points total: {len(pts_cf)} pairs ({len(IMAGE_NAMES)} images x 6 pts)')

    F_reproduced, mask = compute_F_from_points(pts_cf, pts_cs)
    n_inliers = mask.sum()
    print(f'RANSAC inliers: {n_inliers} / {len(pts_cf)}')
    print(f'\nReproduced F:\n{F_reproduced}')
    print(f'\nGT F:\n{F_gt}')

    compare_F(F_reproduced, F_gt, pts_cf[mask], pts_cs[mask], label='GT-reproduced')

    if n_inliers >= 15:
        print('\n[OK] Sanity check passed. Pipeline is correct.')
    else:
        print('\n[WARNING] Too few inliers. Check GT point format.')


def run_superglue_eval(ransac_thresh, min_matches):
    """Main eval: load SuperGlue matches, compute F, compare to GT."""
    print('\n=== SuperGlue matching evaluation ===')

    F_gt = load_gt_F()
    print('GT F matrix loaded.')

    all_mkpts0, all_mkpts1 = [], []
    per_image_stats = []

    for name in IMAGE_NAMES:
        stem = Path(name).stem
        npz_path = OUTPUT_DIR / f'{stem}_{stem}_matches.npz'
        if not npz_path.exists():
            print(f'  [SKIP] {npz_path.name} not found')
            continue

        mkpts0, mkpts1, mconf = load_superglue_matches(npz_path)
        all_mkpts0.append(mkpts0)
        all_mkpts1.append(mkpts1)
        per_image_stats.append({
            'name': name,
            'n_matches': len(mkpts0),
            'mean_conf': mconf.mean() if len(mconf) > 0 else 0
        })

    if not all_mkpts0:
        print('[FAIL] No .npz files found. Run match_pairs.py first.')
        return

    print(f'\n{"Image":<12} {"Matches":>8} {"Mean Conf":>10}')
    print('-' * 32)
    for s in per_image_stats:
        print(f'{s["name"]:<12} {s["n_matches"]:>8} {s["mean_conf"]:>10.4f}')

    all_mkpts0 = np.vstack(all_mkpts0)
    all_mkpts1 = np.vstack(all_mkpts1)
    total = len(all_mkpts0)
    print(f'\nTotal matched points: {total}')

    if total < min_matches:
        print(f'[FAIL] Too few matches ({total} < {min_matches}). Adjust --max_keypoints or --match_threshold.')
        return

    F_sg, mask = compute_F_from_points(all_mkpts0, all_mkpts1, ransac_thresh)
    n_inliers = mask.sum()
    print(f'RANSAC inliers: {n_inliers} / {total}  ({100*n_inliers/total:.1f}%)')
    print(f'\nSuperGlue F:\n{F_sg}')
    print(f'\nGT F:\n{F_gt}')

    compare_F(F_sg, F_gt, all_mkpts0[mask], all_mkpts1[mask], label='SuperGlue')

    print('\n--- Conclusion ---')
    if n_inliers >= 30:
        print('[OK] Enough inliers. F matrix estimate is reliable.')
    elif n_inliers >= 15:
        print('[WARNING] Moderate inliers. Result is indicative but try to get more matches.')
    else:
        print('[FAIL] Too few inliers. Improve SuperGlue matching quality first.')


# ── Entry point ───────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='計算並比較 Fundamental Matrix')
    parser.add_argument('--verify_gt', action='store_true',
                        help='驗證模式：用人工 GT 點重現 F matrix，確認 pipeline 正確')
    parser.add_argument('--ransac_thresh', type=float, default=3.0,
                        help='RANSAC 重投影誤差閾值（像素，預設 3.0）')
    parser.add_argument('--min_matches', type=int, default=20,
                        help='最少需要幾個匹配點才進行 F matrix 估計（預設 20）')
    opt = parser.parse_args()

    if opt.verify_gt:
        run_verify_gt()
    else:
        run_superglue_eval(opt.ransac_thresh, opt.min_matches)
