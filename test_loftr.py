"""
test_loftr.py

用 LoFTR 對 pair 08 做 matching，和 SuperGlue 做 E matrix / Pose 比較。
LoFTR 是 dense matching，不需要 SuperPoint，直接輸出 corresponding points。

Usage:
  .venv/Scripts/python.exe test_loftr.py
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from kornia.feature import LoFTR

DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
INTRINSIC_DIR = Path('Bullpen_Calibration/Intrinsic')
OUTPUT_DIR    = DATASET_DIR / 'output_1920'
PAIR          = '08'
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    mask = mask.ravel().astype(bool)
    sv = np.linalg.svd(E, compute_uv=False)
    n_in, R, t, _ = cv2.recoverPose(E, pts0_n[mask], pts1_n[mask], np.eye(3))
    rx, ry, rz = rot_to_euler(R)
    print(f'[{label}] points={len(pts0)}  E-inliers={mask.sum()}  sv_ratio={sv[0]/sv[1]:.4f}')
    print(f'[{label}] Pose: Roll={rx:.1f}  Pitch={ry:.1f}  Yaw={rz:.1f}')
    print(f'[{label}] t={t.ravel()}')
    return mask.sum()


def main():
    print(f'Device: {DEVICE}')
    K_cf = load_K(INTRINSIC_DIR / 'Cf_Intrinsic.txt')
    K_cs = load_K(INTRINSIC_DIR / 'Cs_Intrinsic.txt')

    # Load images (grayscale float32 for LoFTR)
    img_cf_path = DATASET_DIR / 'cf' / f'{PAIR}.jpg'
    img_cs_path = DATASET_DIR / 'cs' / f'{PAIR}.jpg'
    img_cf = cv2.imread(str(img_cf_path), cv2.IMREAD_GRAYSCALE)
    img_cs = cv2.imread(str(img_cs_path), cv2.IMREAD_GRAYSCALE)
    print(f'cf image: {img_cf.shape}  cs image: {img_cs.shape}')

    # LoFTR expects 640x480 or similar; resize to avoid OOM, keep aspect ratio
    # Use 640px on the shorter side
    def resize_for_loftr(img, max_side=640):
        h, w = img.shape
        scale = max_side / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        # LoFTR needs dimensions divisible by 8
        nh = (nh // 8) * 8
        nw = (nw // 8) * 8
        return cv2.resize(img, (nw, nh)), scale, (w, h)

    img_cf_small, scale_cf, orig_cf = resize_for_loftr(img_cf, 640)
    img_cs_small, scale_cs, orig_cs = resize_for_loftr(img_cs, 640)
    print(f'LoFTR input: cf={img_cf_small.shape}  cs={img_cs_small.shape}')

    # Convert to tensor
    def to_tensor(img):
        t = torch.from_numpy(img).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,H,W]

    # Run LoFTR
    print('\nRunning LoFTR (outdoor weights)...')
    import ssl, os
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    matcher = LoFTR(pretrained='outdoor').to(DEVICE)
    ssl._create_default_https_context = _orig_ctx
    matcher.eval()

    with torch.no_grad():
        batch = {
            'image0': to_tensor(img_cf_small),
            'image1': to_tensor(img_cs_small),
        }
        out = matcher(batch)
        mkpts0 = out['keypoints0'].cpu().numpy()  # [N,2] in resized coords
        mkpts1 = out['keypoints1'].cpu().numpy()
        conf   = out['confidence'].cpu().numpy()

    print(f'LoFTR raw matches: {len(mkpts0)}  (conf>0.5: {(conf>0.5).sum()}  conf>0.9: {(conf>0.9).sum()})')

    # Scale back to original pixel coords
    mkpts0[:, 0] *= orig_cf[0] / img_cf_small.shape[1]
    mkpts0[:, 1] *= orig_cf[1] / img_cf_small.shape[0]
    mkpts1[:, 0] *= orig_cs[0] / img_cs_small.shape[1]
    mkpts1[:, 1] *= orig_cs[1] / img_cs_small.shape[0]

    # Filter by confidence
    for thresh in [0.0, 0.5, 0.9]:
        mask_conf = conf >= thresh
        pts0_f = mkpts0[mask_conf]
        pts1_f = mkpts1[mask_conf]
        print(f'\n--- LoFTR conf>={thresh:.1f}  ({mask_conf.sum()} points) ---')
        run_pose(pts0_f, pts1_f, K_cf, K_cs, f'LoFTR conf>={thresh:.1f}')

    # Compare: SG pair 08
    print('\n--- SuperGlue pair 08 (for comparison) ---')
    npz = OUTPUT_DIR / f'{PAIR}_{PAIR}_matches.npz'
    data = np.load(npz)
    kpts0, kpts1, matches = data['keypoints0'], data['keypoints1'], data['matches']
    valid = matches > -1
    sg0 = kpts0[valid].astype(np.float64)
    sg1 = kpts1[matches[valid]].astype(np.float64)
    run_pose(sg0, sg1, K_cf, K_cs, 'SG pair08')

    # Save LoFTR viz
    print('\nSaving LoFTR visualization...')
    img_cf_color = cv2.imread(str(img_cf_path))
    img_cs_color = cv2.imread(str(img_cs_path))
    h0, w0 = img_cf_color.shape[:2]
    h1, w1 = img_cs_color.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + 4 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]   = img_cf_color
    canvas[:h1, w0+4:] = img_cs_color
    canvas[:, w0:w0+4] = 40

    mask_hi = conf >= 0.5
    pts0_hi = mkpts0[mask_hi]
    pts1_hi = mkpts1[mask_hi]
    n = len(pts0_hi)
    import matplotlib.cm as cm
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts0_hi, pts1_hi)):
        rgba = cm.plasma(i / max(n-1, 1))
        c = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        p0 = (int(x0), int(y0))
        p1 = (int(x1)+w0+4, int(y1))
        cv2.line(canvas, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 4, c, -1, cv2.LINE_AA)

    out_path = DATASET_DIR / 'loftr_pair08.jpg'
    cv2.imwrite(str(out_path), canvas)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
