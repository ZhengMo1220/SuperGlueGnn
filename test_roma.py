"""
test_roma.py

用 RoMa (CVPR 2024) 對 pair 08 做 matching，和 SuperGlue / LoFTR 做 E matrix / Pose 比較。
RoMa 是 dense matching，使用 DINOv2 backbone，不需要 SuperPoint。

Usage:
  .venv/Scripts/python.exe test_roma.py
"""

import numpy as np
import cv2
import torch
from pathlib import Path

DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
INTRINSIC_DIR = Path('Bullpen_Calibration/Intrinsic')
PAIR          = '08'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES     = 5000   # sparse samples from dense warp


def load_K(path):
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
    return np.array(rows[:3], dtype=np.float64)


def rot_to_euler(R):
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    x = np.degrees(np.arctan2( R[2, 1],  R[2, 2]))
    y = np.degrees(np.arctan2(-R[2, 0],  sy))
    z = np.degrees(np.arctan2( R[1, 0],  R[0, 0]))
    return x, y, z


def run_pose(pts0, pts1, K_cf, K_cs, label):
    if len(pts0) < 8:
        print(f'[{label}] Too few points ({len(pts0)}), skip.')
        return
    pts0_n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K_cf, None).reshape(-1, 2)
    pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K_cs, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts0_n, pts1_n, np.eye(3),
                                   cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        print(f'[{label}] findEssentialMat failed.')
        return
    if E.shape[0] > 3:
        E = E[:3]
    mask = mask.ravel().astype(bool)
    sv = np.linalg.svd(E, compute_uv=False)
    n_in, R, t, _ = cv2.recoverPose(E, pts0_n[mask], pts1_n[mask], np.eye(3))
    rx, ry, rz = rot_to_euler(R)
    print(f'[{label}] points={len(pts0)}  E-inliers={mask.sum()}  sv_ratio={sv[0]/sv[1]:.4f}')
    print(f'[{label}] Pose: Roll={rx:.1f}  Pitch={ry:.1f}  Yaw={rz:.1f}')
    print(f'[{label}] t={t.ravel()}')


def main():
    print(f'Device: {DEVICE}')
    K_cf = load_K(INTRINSIC_DIR / 'Cf_Intrinsic.txt')
    K_cs = load_K(INTRINSIC_DIR / 'Cs_Intrinsic.txt')

    img_cf_path = str(DATASET_DIR / 'cf' / f'{PAIR}.jpg')
    img_cs_path = str(DATASET_DIR / 'cs' / f'{PAIR}.jpg')

    img_cf = cv2.imread(img_cf_path)
    img_cs = cv2.imread(img_cs_path)
    H0, W0 = img_cf.shape[:2]
    H1, W1 = img_cs.shape[:2]
    print(f'cf: {W0}x{H0}  cs: {W1}x{H1}')

    # ── RoMa ──────────────────────────────────────────────────────────────────
    print('\nLoading RoMa outdoor model...')
    import ssl, os
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    from romatch import roma_outdoor
    model = roma_outdoor(device=DEVICE)
    ssl._create_default_https_context = _orig

    print('Running RoMa matching...')
    with torch.no_grad():
        warp, certainty = model.match(img_cf_path, img_cs_path, device=DEVICE)

    # Sample sparse matches
    matches, conf = model.sample(warp, certainty, num=N_SAMPLES)
    kpts0, kpts1 = model.to_pixel_coordinates(matches, H0, W0, H1, W1)
    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    conf  = conf.cpu().numpy()

    print(f'RoMa raw samples: {len(kpts0)}')
    for thresh in [0.0, 0.5, 0.9]:
        mask = conf >= thresh
        print(f'\n--- RoMa conf>={thresh:.1f} ({mask.sum()} points) ---')
        run_pose(kpts0[mask], kpts1[mask], K_cf, K_cs, f'RoMa conf>={thresh:.1f}')

    # ── SuperGlue (for comparison) ────────────────────────────────────────────
    print('\n--- SuperGlue pair 08 (comparison) ---')
    npz = DATASET_DIR / 'output' / f'{PAIR}_{PAIR}_matches.npz'
    data = np.load(str(npz))
    kp0, kp1, mts = data['keypoints0'], data['keypoints1'], data['matches']
    valid = mts > -1
    run_pose(kp0[valid].astype(np.float64), kp1[mts[valid]].astype(np.float64),
             K_cf, K_cs, 'SG pair08')

    # ── Save RoMa visualization ───────────────────────────────────────────────
    print('\nSaving RoMa visualization...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    mask_hi = conf >= 0.5
    p0 = kpts0[mask_hi]
    p1 = kpts1[mask_hi]
    n = len(p0)

    h0, w0 = img_cf.shape[:2]
    h1, w1 = img_cs.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + 4 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]   = img_cf
    canvas[:h1, w0+4:] = img_cs
    canvas[:, w0:w0+4] = 40

    for i in range(min(n, 300)):   # draw at most 300 lines to keep readable
        t = i / max(n - 1, 1)
        rgba = cm.plasma(t)
        c = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        cv2.line(canvas, (int(p0[i,0]), int(p0[i,1])),
                 (int(p1[i,0]) + w0 + 4, int(p1[i,1])), c, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(p0[i,0]), int(p0[i,1])), 4, c, -1)
        cv2.circle(canvas, (int(p1[i,0]) + w0 + 4, int(p1[i,1])), 4, c, -1)

    out = DATASET_DIR / 'inlier_viz' / 'matching_roma_08.jpg'
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
