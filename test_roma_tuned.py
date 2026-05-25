"""
test_roma_tuned.py

測試不同 RoMa 參數組合對全 20 對的 Pitch 穩定性影響。
比較基準：
  A: Baseline   upsample_res=864,  N=5000, conf=0.5
  B: 高解析度   upsample_res=1024, N=5000, conf=0.5
  C: 多樣本     upsample_res=864,  N=8000, conf=0.5
  D: 低 conf    upsample_res=864,  N=5000, conf=0.3

Usage:
  .venv/Scripts/python.exe test_roma_tuned.py
"""

import numpy as np
import cv2
import torch
from pathlib import Path
import ssl, os

DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
INTRINSIC_DIR = Path('Bullpen_Calibration/Intrinsic')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIGS = [
    {'name': 'A_baseline',   'upsample_res': 864,  'n_samples': 5000, 'conf': 0.5},
    {'name': 'B_hires',      'upsample_res': 1024, 'n_samples': 5000, 'conf': 0.5},
    {'name': 'C_moresamples','upsample_res': 864,  'n_samples': 8000, 'conf': 0.5},
    {'name': 'D_lowconf',    'upsample_res': 864,  'n_samples': 5000, 'conf': 0.3},
]


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
        return None
    pts0_n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K_cf, None).reshape(-1, 2)
    pts1_n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K_cs, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts0_n, pts1_n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        return None
    if E.shape[0] > 3:
        E = E[:3]
    mask = mask.ravel().astype(bool)
    n_in, R, t, _ = cv2.recoverPose(E, pts0_n[mask], pts1_n[mask], np.eye(3))
    rx, ry, rz = rot_to_euler(R)
    return rx, ry, rz, int(mask.sum())


def run_config(model, cfg, K_cf, K_cs):
    pitches, inliers_list = [], []
    for i in range(1, 21):
        pair = f'{i:02d}'
        img_cf = str(DATASET_DIR / 'cf' / f'{pair}.jpg')
        img_cs = str(DATASET_DIR / 'cs' / f'{pair}.jpg')
        if not Path(img_cf).exists():
            continue
        img = cv2.imread(img_cf)
        H0, W0 = img.shape[:2]
        img = cv2.imread(img_cs)
        H1, W1 = img.shape[:2]

        with torch.no_grad():
            warp, certainty = model.match(img_cf, img_cs, device=DEVICE)
        matches, conf = model.sample(warp, certainty, num=cfg['n_samples'])
        kpts0, kpts1 = model.to_pixel_coordinates(matches, H0, W0, H1, W1)
        kpts0 = kpts0.cpu().numpy()
        kpts1 = kpts1.cpu().numpy()
        conf  = conf.cpu().numpy()

        mask = conf >= cfg['conf']
        pts0, pts1 = kpts0[mask], kpts1[mask]
        result = run_pose(pts0, pts1, K_cf, K_cs)
        if result is None:
            print(f"  [{pair}] pose failed")
            continue
        rx, ry, rz, n_in = result
        print(f"  [{pair}] E-inliers={n_in:4d}  Pitch={ry:7.2f}°")
        pitches.append(ry)
        inliers_list.append(n_in)

    return pitches, inliers_list


def main():
    print(f'Device: {DEVICE}')
    K_cf = load_K(INTRINSIC_DIR / 'Cf_Intrinsic.txt')
    K_cs = load_K(INTRINSIC_DIR / 'Cs_Intrinsic.txt')

    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context

    from romatch import roma_outdoor

    summary = []

    for cfg in CONFIGS:
        print(f'\n{"="*55}')
        print(f'Config {cfg["name"]}  upsample_res={cfg["upsample_res"]}  N={cfg["n_samples"]}  conf>={cfg["conf"]}')
        print('='*55)

        model = roma_outdoor(device=DEVICE, upsample_res=(cfg['upsample_res'], cfg['upsample_res']))
        ssl._create_default_https_context = _orig

        pitches, inliers_list = run_config(model, cfg, K_cf, K_cs)

        if pitches:
            pmean = np.mean(pitches)
            pstd  = np.std(pitches)
            pmin  = np.min(pitches)
            pmax  = np.max(pitches)
            imean = np.mean(inliers_list)
            summary.append({**cfg,
                             'pitch_mean': pmean, 'pitch_std': pstd,
                             'pitch_min': pmin,   'pitch_max': pmax,
                             'inliers_mean': imean})
            print(f'  Pitch: mean={pmean:.1f}  std={pstd:.1f}  min={pmin:.1f}  max={pmax:.1f}')
            print(f'  E-inliers mean: {imean:.0f}')
        else:
            print('  No valid results.')

        # reload ssl for next iter
        ssl._create_default_https_context = ssl._create_unverified_context

    ssl._create_default_https_context = _orig

    print('\n\n' + '='*60)
    print('FINAL SUMMARY')
    print('='*60)
    print(f'{"Config":<18} {"up_res":>7} {"N":>6} {"conf":>5} | {"Pitch mean":>10} {"std":>5} {"min":>7} {"max":>7} | {"E-inliers":>9}')
    print('-'*80)
    for s in summary:
        print(f'{s["name"]:<18} {s["upsample_res"]:>7} {s["n_samples"]:>6} {s["conf"]:>5.1f} | '
              f'{s["pitch_mean"]:>10.1f} {s["pitch_std"]:>5.1f} {s["pitch_min"]:>7.1f} {s["pitch_max"]:>7.1f} | '
              f'{s["inliers_mean"]:>9.0f}')
    print(f'\n參考基準：人工點 Pitch = -82.0°  (Direct E, 全20對 120pts)')


if __name__ == '__main__':
    main()
