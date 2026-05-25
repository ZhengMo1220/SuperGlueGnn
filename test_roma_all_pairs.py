"""
test_roma_all_pairs.py

用 RoMa 對全部 20 對圖片做 matching，驗證 Pitch 是否穩定落在特定範圍。
每對圖片計算 Essential Matrix 後輸出 Roll/Pitch/Yaw。
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import ssl, os

DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
INTRINSIC_DIR = Path('Bullpen_Calibration/Intrinsic')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLES     = 5000
CONF_THRESH   = 0.5


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


def run_pose(pts0, pts1, K_cf, K_cs):
    if len(pts0) < 8:
        return None, None
    pts0_n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K_cf, None).reshape(-1, 2)
    pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K_cs, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts0_n, pts1_n, np.eye(3),
                                   cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        return None, None
    if E.shape[0] > 3:
        E = E[:3]
    mask = mask.ravel().astype(bool)
    n_in, R, t, _ = cv2.recoverPose(E, pts0_n[mask], pts1_n[mask], np.eye(3))
    rx, ry, rz = rot_to_euler(R)
    return (rx, ry, rz, int(mask.sum()))


def main():
    print(f'Device: {DEVICE}')
    K_cf = load_K(INTRINSIC_DIR / 'Cf_Intrinsic.txt')
    K_cs = load_K(INTRINSIC_DIR / 'Cs_Intrinsic.txt')

    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    print('Loading RoMa outdoor model...')
    from romatch import roma_outdoor
    model = roma_outdoor(device=DEVICE)
    ssl._create_default_https_context = _orig

    results = []
    for i in range(1, 21):
        pair = f'{i:02d}'
        img_cf_path = str(DATASET_DIR / 'cf' / f'{pair}.jpg')
        img_cs_path = str(DATASET_DIR / 'cs' / f'{pair}.jpg')

        if not Path(img_cf_path).exists() or not Path(img_cs_path).exists():
            print(f'[{pair}] Image not found, skip.')
            continue

        img_cf = cv2.imread(img_cf_path)
        img_cs = cv2.imread(img_cs_path)
        H0, W0 = img_cf.shape[:2]
        H1, W1 = img_cs.shape[:2]

        print(f'\n[{pair}] Matching...', flush=True)
        with torch.no_grad():
            warp, certainty = model.match(img_cf_path, img_cs_path, device=DEVICE)
        matches, conf = model.sample(warp, certainty, num=N_SAMPLES)
        kpts0, kpts1 = model.to_pixel_coordinates(matches, H0, W0, H1, W1)
        kpts0 = kpts0.cpu().numpy()
        kpts1 = kpts1.cpu().numpy()
        conf  = conf.cpu().numpy()

        mask_conf = conf >= CONF_THRESH
        pts0 = kpts0[mask_conf]
        pts1 = kpts1[mask_conf]

        pose = run_pose(pts0, pts1, K_cf, K_cs)
        if pose is None:
            print(f'[{pair}] Pose failed.')
            results.append((pair, None))
            continue

        rx, ry, rz, n_in = pose
        print(f'[{pair}] conf>={CONF_THRESH}: {mask_conf.sum()} pts  E-inliers={n_in}  Roll={rx:.1f}  Pitch={ry:.1f}  Yaw={rz:.1f}')
        results.append((pair, (rx, ry, rz, n_in)))

    print('\n\n========== SUMMARY ==========')
    print(f'{"Pair":<6} {"Roll":>7} {"Pitch":>7} {"Yaw":>7} {"E-inliers":>10}')
    print('-' * 45)
    pitches = []
    for pair, res in results:
        if res is None:
            print(f'{pair:<6} {"FAILED":>7}')
        else:
            rx, ry, rz, n_in = res
            print(f'{pair:<6} {rx:>7.1f} {ry:>7.1f} {rz:>7.1f} {n_in:>10}')
            pitches.append(ry)

    if pitches:
        print(f'\nPitch stats: min={min(pitches):.1f}  max={max(pitches):.1f}  mean={np.mean(pitches):.1f}  std={np.std(pitches):.1f}')


if __name__ == '__main__':
    main()
