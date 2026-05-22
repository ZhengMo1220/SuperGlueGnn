"""
compute_pose.py

從 SuperGlue RANSAC inlier 點 + 相機內參，計算兩台相機的相對姿態。

Pipeline:
  SuperGlue .npz → RANSAC inliers → E matrix → cv2.recoverPose → R, t

Usage:
  .venv/Scripts/python.exe compute_pose.py
  .venv/Scripts/python.exe compute_pose.py --dataset_dir Bullpen_Calibration/TSG_Bullpen --output_dir Bullpen_Calibration/TSG_Bullpen/output_1920
"""

import argparse
import numpy as np
import cv2
from pathlib import Path


# ── 預設路徑 ──────────────────────────────────────────
DATASET_DIR  = Path('Bullpen_Calibration/TSG_Bullpen')
OUTPUT_DIR   = DATASET_DIR / 'output_1920'
INTRINSIC_DIR = Path('Bullpen_Calibration/Intrinsic')
IMAGE_NAMES  = [f'{i:02d}.jpg' for i in range(1, 21)]
RANSAC_THRESH = 3.0


# ── 讀取內參 ──────────────────────────────────────────

def load_K(path: Path) -> np.ndarray:
    """從 Cf_Intrinsic.txt / Cs_Intrinsic.txt 解析 3x3 K matrix。"""
    text = path.read_text(encoding='utf-8')
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('['):
            continue
        clean = line.replace('[', '').replace(']', '').replace(',', ' ')
        vals = [float(x) for x in clean.split()]
        if vals:
            rows.append(vals)
    K = np.array(rows[:3], dtype=np.float64)
    assert K.shape == (3, 3), f'K shape error: {K.shape}'
    return K


# ── 讀取 SuperGlue 匹配 ───────────────────────────────

def load_superglue_matches(npz_path: Path):
    data    = np.load(npz_path)
    kpts0   = data['keypoints0']
    kpts1   = data['keypoints1']
    matches = data['matches']
    conf    = data['match_confidence']
    valid   = matches > -1
    return kpts0[valid], kpts1[matches[valid]], conf[valid]


# ── 主流程 ────────────────────────────────────────────

def run(dataset_dir: Path, output_dir: Path, intrinsic_dir: Path, ransac_thresh: float, save: bool):
    # 讀取內參
    K_cf = load_K(intrinsic_dir / 'Cf_Intrinsic.txt')
    K_cs = load_K(intrinsic_dir / 'Cs_Intrinsic.txt')
    print('K_cf (Camera Front):')
    print(K_cf)
    print('\nK_cs (Camera Side):')
    print(K_cs)

    # 收集所有 SuperGlue 匹配點
    all_pts0, all_pts1 = [], []
    for name in IMAGE_NAMES:
        stem = Path(name).stem
        npz  = output_dir / f'{stem}_{stem}_matches.npz'
        if not npz.exists():
            print(f'  [SKIP] {npz.name}')
            continue
        pts0, pts1, _ = load_superglue_matches(npz)
        all_pts0.append(pts0)
        all_pts1.append(pts1)

    if not all_pts0:
        print('[FAIL] No .npz files found.')
        return

    all_pts0 = np.vstack(all_pts0)
    all_pts1 = np.vstack(all_pts1)
    print(f'\nTotal SuperGlue matches: {len(all_pts0)}')

    # ── Step 1: F matrix via RANSAC ──────────────────
    F, mask_f = cv2.findFundamentalMat(
        all_pts0, all_pts1,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=ransac_thresh,
        confidence=0.99
    )
    mask_f = mask_f.ravel().astype(bool)
    inlier_pts0 = all_pts0[mask_f]
    inlier_pts1 = all_pts1[mask_f]
    print(f'RANSAC inliers (F): {mask_f.sum()} / {len(all_pts0)}  ({100*mask_f.sum()/len(all_pts0):.1f}%)')
    print(f'\nFundamental Matrix F:\n{F}')

    # ── Step 2: E matrix = K2^T @ F @ K1 ────────────
    # F convention here: p_cs^T F p_cf = 0  (findFundamentalMat(cf, cs))
    # E convention:      p_cs^T E p_cf = 0  -> E = K_cs^T F K_cf
    # After transposing for cf-first output, E would be K_cf^T F^T K_cs,
    # but recoverPose below uses the internal F directly so keep as-is.
    E = K_cs.T @ F @ K_cf
    print(f'\nEssential Matrix E:\n{E}')

    # ── Step 3: recoverPose → R, t ───────────────────
    # recoverPose 接受其中一台相機的 K；
    # 因兩台內參不同，用平均 focal length 或 K_cf 皆可，
    # 但最精確的做法是先把兩組點各自用各自 K undistort 到 normalized 座標。
    pts0_n = cv2.undistortPoints(inlier_pts0.reshape(-1,1,2), K_cf, None).reshape(-1,2)
    pts1_n = cv2.undistortPoints(inlier_pts1.reshape(-1,1,2), K_cs, None).reshape(-1,2)

    # 用 identity K（已 normalize），讓 recoverPose 正確運作
    K_eye = np.eye(3, dtype=np.float64)
    n_inliers, R, t, mask_e = cv2.recoverPose(E, pts0_n, pts1_n, K_eye)
    print(f'\nrecoverPose inliers: {n_inliers} / {mask_f.sum()}')

    print(f'\n{"="*55}')
    print('Rotation matrix R (cf → cs):')
    print(R)

    # 轉成 Euler angles（ZYX 順序，單位：度）
    def rot_to_euler(R):
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        if not singular:
            x = np.degrees(np.arctan2( R[2,1], R[2,2]))
            y = np.degrees(np.arctan2(-R[2,0], sy))
            z = np.degrees(np.arctan2( R[1,0], R[0,0]))
        else:
            x = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            y = np.degrees(np.arctan2(-R[2,0], sy))
            z = 0
        return x, y, z

    rx, ry, rz = rot_to_euler(R)
    print(f'\nEuler angles (ZYX, degrees):')
    print(f'  Roll  (X): {rx:.3f} deg')
    print(f'  Pitch (Y): {ry:.3f} deg')
    print(f'  Yaw   (Z): {rz:.3f} deg')

    print(f'\nTranslation direction t (unit vector, cf → cs):')
    print(f'  t = {t.ravel()}')
    print('  (scale unknown — only direction is recoverable without 3D reference)')
    print(f'{"="*55}')

    # ── Step 4: Sampson distance 驗證 ────────────────
    def sampson(F_, p0, p1):
        p0h = np.hstack([p0, np.ones((len(p0),1))])
        p1h = np.hstack([p1, np.ones((len(p1),1))])
        Fp0  = (F_ @ p0h.T).T
        FTp1 = (F_.T @ p1h.T).T
        num  = (p1h * Fp0).sum(1)**2
        den  = Fp0[:,0]**2 + Fp0[:,1]**2 + FTp1[:,0]**2 + FTp1[:,1]**2
        return num / den

    samp = sampson(F, inlier_pts0, inlier_pts1)
    print(f'\nSampson distance on inliers:  mean={samp.mean():.4f}  median={np.median(samp):.4f}')
    print('(lower = better epipolar consistency)')

    # ── Step 5: 存檔 ──────────────────────────────────
    if save:
        out_path = dataset_dir / 'pose_result.txt'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('=== Pose Estimation Result ===\n')
            f.write(f'Dataset   : {dataset_dir}\n')
            f.write(f'Output dir: {output_dir}\n\n')
            f.write(f'RANSAC inliers (F): {mask_f.sum()} / {len(all_pts0)}\n')
            f.write(f'recoverPose inliers: {n_inliers}\n\n')
            f.write('Rotation matrix R (cf -> cs):\n')
            for row in R:
                f.write(f'  {row}\n')
            f.write(f'\nEuler angles (ZYX, degrees):\n')
            f.write(f'  Roll  (X): {rx:.6f}\n')
            f.write(f'  Pitch (Y): {ry:.6f}\n')
            f.write(f'  Yaw   (Z): {rz:.6f}\n')
            f.write(f'\nTranslation direction t (unit vector):\n')
            f.write(f'  {t.ravel()}\n')
            f.write('\nFundamental Matrix F:\n')
            for row in F:
                f.write(f'  {row}\n')
            f.write('\nEssential Matrix E:\n')
            for row in E:
                f.write(f'  {row}\n')
        print(f'\nResult saved to: {out_path}')


# ── Entry point ───────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Essential Matrix + Pose Estimation')
    parser.add_argument('--dataset_dir',   default='Bullpen_Calibration/TSG_Bullpen')
    parser.add_argument('--output_dir',    default=None,
                        help='SuperGlue .npz 所在目錄（預設：<dataset_dir>/output_1920）')
    parser.add_argument('--intrinsic_dir', default='Bullpen_Calibration/Intrinsic')
    parser.add_argument('--ransac_thresh', type=float, default=3.0)
    parser.add_argument('--save', action='store_true', help='將結果存成 pose_result.txt')
    opt = parser.parse_args()

    dataset_dir   = Path(opt.dataset_dir)
    output_dir    = Path(opt.output_dir) if opt.output_dir else dataset_dir / 'output_1920'
    intrinsic_dir = Path(opt.intrinsic_dir)

    run(dataset_dir, output_dir, intrinsic_dir, opt.ransac_thresh, opt.save)
