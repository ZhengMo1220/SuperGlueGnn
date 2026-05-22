"""
compute_fundamental.py

Pipeline:
  1. Load SuperGlue .npz outputs (auto matched points)
  2. Run RANSAC + 8-point to estimate Fundamental Matrix
  3. Compare against GT F matrix (fundamental matrix.txt)
  4. [verify_gt mode] Reproduce GT F matrix from manual GT points first

Usage:
  # Default dataset (TSG_Bullpen, output/)
  .venv/Scripts/python.exe compute_fundamental.py
  .venv/Scripts/python.exe compute_fundamental.py --verify_gt

  # Custom dataset / output directory
  .venv/Scripts/python.exe compute_fundamental.py \
      --dataset_dir Bullpen_Calibration/TSG_Bullpen \
      --output_dir  Bullpen_Calibration/TSG_Bullpen/output_1920
"""

import argparse
import json
import numpy as np
import cv2
from pathlib import Path


IMAGE_NAMES = [f'{i:02d}.jpg' for i in range(1, 21)]

# 預設值；在 __main__ 區塊會依 CLI 參數覆蓋
DATASET_DIR = Path('Bullpen_Calibration/TSG_Bullpen')
OUTPUT_DIR  = DATASET_DIR / 'output'
GT_F_FILE   = DATASET_DIR / 'fundamental matrix.txt'
GT_CF_JSON  = DATASET_DIR / 'selected_points_cf.json'
GT_CS_JSON  = DATASET_DIR / 'selected_points_cs.json'


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


def save_F_txt(F, n_inliers, n_total, path):
    """輸出 F matrix 成與 GT 相同格式的 txt。"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'總共匹配點對數： {n_total}\n')
        f.write('估計出的基本矩陣 F：\n')
        rows = []
        for row in F:
            rows.append(f' [{row[0]:.8e}  {row[1]:.8e}  {row[2]:.8e}]')
        f.write('\n'.join(rows) + '\n')
        f.write(f'內點數量： {n_inliers} / {n_total}\n')
    print(f'F matrix saved to: {path}')


def collect_superglue_points():
    """讀取所有 .npz，回傳堆疊後的匹配點對。"""
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
    return (np.vstack(all_mkpts0) if all_mkpts0 else np.empty((0, 2))),\
           (np.vstack(all_mkpts1) if all_mkpts1 else np.empty((0, 2))),\
           per_image_stats


def run_superglue_eval(ransac_thresh, min_matches, save_F=False, use_gt_points=False):
    """Main eval: load SuperGlue matches, optionally merge GT points, compute F, compare to GT."""

    mode = 'SuperGlue + GT manual points (merged)' if use_gt_points else 'SuperGlue only'
    print(f'\n=== SuperGlue matching evaluation  [{mode}] ===')

    F_gt = load_gt_F()
    print('GT F matrix loaded.')

    sg_pts0, sg_pts1, per_image_stats = collect_superglue_points()

    if len(sg_pts0) == 0:
        print('[FAIL] No .npz files found. Run match_pairs.py first.')
        return

    print(f'\n{"Image":<12} {"Matches":>8} {"Mean Conf":>10}')
    print('-' * 32)
    for s in per_image_stats:
        print(f'{s["name"]:<12} {s["n_matches"]:>8} {s["mean_conf"]:>10.4f}')

    if use_gt_points:
        gt_pts0, gt_pts1 = load_gt_points()
        all_mkpts0 = np.vstack([sg_pts0, gt_pts0])
        all_mkpts1 = np.vstack([sg_pts1, gt_pts1])
        print(f'\nSuperGlue points : {len(sg_pts0)}')
        print(f'GT manual points : {len(gt_pts0)}  (appended)')
        print(f'Total combined   : {len(all_mkpts0)}')
    else:
        all_mkpts0 = sg_pts0
        all_mkpts1 = sg_pts1
        print(f'\nTotal matched points: {len(all_mkpts0)}')

    total = len(all_mkpts0)
    if total < min_matches:
        print(f'[FAIL] Too few matches ({total} < {min_matches}).')
        return

    # ── SuperGlue-only F (always computed for comparison baseline) ──
    F_sg_only, mask_sg = compute_F_from_points(sg_pts0, sg_pts1, ransac_thresh)
    n_in_sg = mask_sg.sum()

    # ── Main F (either merged or SG-only) ──
    if use_gt_points:
        F_main, mask_main = compute_F_from_points(all_mkpts0, all_mkpts1, ransac_thresh)
        n_inliers = mask_main.sum()

        print(f'\n[SuperGlue only]   RANSAC inliers: {n_in_sg} / {len(sg_pts0)}  ({100*n_in_sg/len(sg_pts0):.1f}%)')
        print(f'[SG + GT merged]   RANSAC inliers: {n_inliers} / {total}  ({100*n_inliers/total:.1f}%)')

        print(f'\n--- F matrix comparison ---')
        print(f'SuperGlue-only F:\n{F_sg_only}')
        print(f'\nSG + GT merged F:\n{F_main}')
        print(f'\nGT F:\n{F_gt}')

        compare_F(F_sg_only, F_gt, sg_pts0[mask_sg], sg_pts1[mask_sg],   label='SG-only   ')
        compare_F(F_main,    F_gt, all_mkpts0[mask_main], all_mkpts1[mask_main], label='SG+GT-merged')

        # how different are the two estimated F matrices from each other
        def norm(M): return M / np.linalg.norm(M)
        diff_between = np.linalg.norm(norm(F_sg_only) - norm(F_main))
        print(f'\nDifference between SG-only F and merged F (Frobenius): {diff_between:.6f}')
        print('(close to 0 = GT points did not change the geometry much)')
    else:
        F_main, mask_main = F_sg_only, mask_sg
        n_inliers = n_in_sg
        print(f'RANSAC inliers: {n_inliers} / {total}  ({100*n_inliers/total:.1f}%)')
        print(f'\nSuperGlue F:\n{F_main}')
        print(f'\nGT F:\n{F_gt}')
        compare_F(F_main, F_gt, all_mkpts0[mask_main], all_mkpts1[mask_main], label='SuperGlue')

    if save_F:
        tag      = 'merged' if use_gt_points else 'superglue'
        res_tag  = OUTPUT_DIR.name  # e.g. 'output_1920', 'output'
        out_path = DATASET_DIR / f'{tag}_F_{res_tag}.txt'
        # Transpose F so convention becomes p_cf^T F p_cs = 0 (cf first)
        save_F_txt(F_main.T, n_inliers, total, out_path)

    print('\n--- Conclusion ---')
    if n_inliers >= 30:
        print('[OK] Enough inliers. F matrix estimate is reliable.')
    elif n_inliers >= 15:
        print('[WARNING] Moderate inliers. Result is indicative.')
    else:
        print('[FAIL] Too few inliers.')


# ── Entry point ───────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='計算並比較 Fundamental Matrix')
    parser.add_argument('--dataset_dir', type=str, default='Bullpen_Calibration/TSG_Bullpen',
                        help='資料集根目錄（預設：Bullpen_Calibration/TSG_Bullpen）')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='SuperGlue .npz 所在目錄（預設：<dataset_dir>/output）')
    parser.add_argument('--verify_gt', action='store_true',
                        help='驗證模式：用人工 GT 點重現 F matrix，確認 pipeline 正確')
    parser.add_argument('--ransac_thresh', type=float, default=3.0,
                        help='RANSAC 重投影誤差閾值（像素，預設 3.0）')
    parser.add_argument('--min_matches', type=int, default=20,
                        help='最少需要幾個匹配點才進行 F matrix 估計（預設 20）')
    parser.add_argument('--save_F', action='store_true',
                        help='將估計的 F matrix 輸出成 txt（與 GT 格式相同）')
    parser.add_argument('--use_gt_points', action='store_true',
                        help='將人工標記的 120 個對應點與 SuperGlue 點合併後一起算 F matrix')
    opt = parser.parse_args()

    # 依 CLI 參數覆蓋全域路徑
    DATASET_DIR = Path(opt.dataset_dir)
    OUTPUT_DIR  = Path(opt.output_dir) if opt.output_dir else DATASET_DIR / 'output'
    GT_F_FILE   = DATASET_DIR / 'fundamental matrix.txt'
    GT_CF_JSON  = DATASET_DIR / 'selected_points_cf.json'
    GT_CS_JSON  = DATASET_DIR / 'selected_points_cs.json'

    if opt.verify_gt:
        run_verify_gt()
    else:
        run_superglue_eval(opt.ransac_thresh, opt.min_matches,
                           save_F=opt.save_F, use_gt_points=opt.use_gt_points)
