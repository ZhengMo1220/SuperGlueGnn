"""
gen_epipolar_groups.py
Generate epipolar evaluation images for four F matrix groups (A/B/C/D).
Uses output/ (1280x960) keypoints and manual points scaled to match.
Convention: OpenCV findFundamentalMat -> p_cs^T F p_cf = 0
"""
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

DATASET_DIR = Path('Bullpen_Calibration/TSG_Bullpen')
OUTPUT_DIR  = DATASET_DIR / 'output'
IMAGE_NAMES = [f'{i:02d}.jpg' for i in range(1, 21)]
BEST_PAIR   = '08'
OUT_DIR     = DATASET_DIR / 'inlier_viz'
OUT_DIR.mkdir(exist_ok=True)

# Scale factors: manual points at 1920x1084, output/ keypoints at 1280x960
SCALE_X = 1280.0 / 1920.0
SCALE_Y = 960.0  / 1084.0


def load_sg_matches(stem):
    npz = OUTPUT_DIR / f'{stem}_{stem}_matches.npz'
    data = np.load(str(npz))
    kpts0, kpts1, matches = data['keypoints0'], data['keypoints1'], data['matches']
    valid = matches > -1
    return kpts0[valid].astype(np.float64), kpts1[matches[valid]].astype(np.float64)


def load_manual_points():
    with open(DATASET_DIR / 'selected_points_cf.json') as f:
        cf_data = json.load(f)
    with open(DATASET_DIR / 'selected_points_cs.json') as f:
        cs_data = json.load(f)
    result = {}
    for key in cf_data:
        stem = Path(key).stem
        pts_cf = np.array(cf_data[key], dtype=np.float64)
        pts_cs = np.array(cs_data[key], dtype=np.float64)
        pts_cf[:, 0] *= SCALE_X
        pts_cf[:, 1] *= SCALE_Y
        pts_cs[:, 0] *= SCALE_X
        pts_cs[:, 1] *= SCALE_Y
        result[stem] = (pts_cf, pts_cs)
    return result


def compute_F_ransac(pts_cf, pts_cs, thresh=3.0):
    if len(pts_cf) < 8:
        return None, None
    F, mask = cv2.findFundamentalMat(pts_cf, pts_cs, cv2.FM_RANSAC, thresh, 0.99)
    if F is None:
        return None, None
    if F.shape[0] > 3:
        F = F[:3]
    return F, mask


def sampson_dist(F, pts_cf, pts_cs):
    N = len(pts_cf)
    p0 = np.hstack([pts_cf, np.ones((N, 1))])
    p1 = np.hstack([pts_cs, np.ones((N, 1))])
    Fp0  = (F   @ p0.T).T
    FTp1 = (F.T @ p1.T).T
    num = ((p1 * Fp0).sum(1)) ** 2
    den = Fp0[:, 0]**2 + Fp0[:, 1]**2 + FTp1[:, 0]**2 + FTp1[:, 1]**2
    return np.sqrt(num / (den + 1e-10))


def draw_line(img, l, color, thickness=1):
    h, w = img.shape[:2]
    a, b, c = float(l[0]), float(l[1]), float(l[2])
    if abs(b) > 1e-8:
        x0, x1 = 0, w - 1
        y0 = int((-c - a * x0) / b)
        y1 = int((-c - a * x1) / b)
    else:
        y0, y1 = 0, h - 1
        x0 = int((-c - b * y0) / a) if abs(a) > 1e-8 else 0
        x1 = int((-c - b * y1) / a) if abs(a) > 1e-8 else 0
    cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)


# Load all data
manual = load_manual_points()

all_sg0, all_sg1 = [], []
for stem in [f'{i:02d}' for i in range(1, 21)]:
    sg0, sg1 = load_sg_matches(stem)
    all_sg0.append(sg0)
    all_sg1.append(sg1)
all_sg0 = np.vstack(all_sg0)
all_sg1 = np.vstack(all_sg1)

all_m_cf = np.vstack([manual[s][0] for s in sorted(manual)])
all_m_cs = np.vstack([manual[s][1] for s in sorted(manual)])

sg08_cf, sg08_cs = load_sg_matches(BEST_PAIR)
m08_cf, m08_cs = manual.get(BEST_PAIR, (np.zeros((0, 2)), np.zeros((0, 2))))

viz_names = {
    'A': 'viz_A_sg_all',
    'B': 'viz_B_sg_manual_all',
    'C': 'viz_C_sg_best',
    'D': 'viz_D_sg_manual_best',
}
labels = {
    'A': 'SG 全20對',
    'B': 'SG+人工 全20對',
    'C': 'SG pair08',
    'D': 'SG+人工 pair08',
}

if len(m08_cf) > 0:
    D_cf = np.vstack([sg08_cf, m08_cf])
    D_cs = np.vstack([sg08_cs, m08_cs])
else:
    D_cf, D_cs = sg08_cf, sg08_cs

groups = {
    'A': (all_sg0, all_sg1),
    'B': (np.vstack([all_sg0, all_m_cf]), np.vstack([all_sg1, all_m_cs])),
    'C': (sg08_cf, sg08_cs),
    'D': (D_cf, D_cs),
}

# Load pair08 color images, resize to 1280x960
img_cf = cv2.resize(cv2.imread(str(DATASET_DIR / 'cf' / f'{BEST_PAIR}.jpg')), (1280, 960))
img_cs = cv2.resize(cv2.imread(str(DATASET_DIR / 'cs' / f'{BEST_PAIR}.jpg')), (1280, 960))

for gname in ['A', 'B', 'C', 'D']:
    pts_cf, pts_cs = groups[gname]
    label = labels[gname]
    print(f'\n=== Group {gname}: {label} ===')

    F, mask = compute_F_ransac(pts_cf, pts_cs)
    if F is None:
        print('  F matrix computation failed')
        continue

    n_total = len(pts_cf)
    n_ransac = int(mask.ravel().sum())
    print(f'  Total: {n_total}  RANSAC inliers: {n_ransac}')

    # Evaluate on pair08 SG matches
    sd = sampson_dist(F, sg08_cf, sg08_cs)
    n_in3  = int((sd < 3.0).sum())
    n_in10 = int((sd < 10.0).sum())
    print(f'  Pair08 epipolar inliers: <3px={n_in3}  <10px={n_in10}  (of {len(sg08_cf)})')
    print(f'  Sampson: min={sd.min():.2f} median={np.median(sd):.2f} max={sd.max():.2f}')

    # Save updated F matrix
    viz_dir = DATASET_DIR / viz_names[gname]
    viz_dir.mkdir(exist_ok=True)
    fmat_path = viz_dir / f'F_matrix_{gname}.txt'
    with open(str(fmat_path), 'w') as fout:
        fout.write(f'Convention: p_cs^T F p_cf = 0 (OpenCV, 1280x960 coords)\n')
        fout.write(f'Group {gname}: {viz_names[gname]}\n')
        fout.write(f'Total points: {n_total}\n')
        fout.write(f'RANSAC inliers: {n_ransac} / {n_total}\n\n')
        fout.write('Fundamental Matrix F:\n')
        for row in F:
            fout.write(f' {list(row)}\n')

    # Draw epipolar evaluation using pair08 matches with Sampson < 10px
    show_mask = sd < 10.0
    pts0_show = sg08_cf[show_mask]
    pts1_show = sg08_cs[show_mask]
    n_show = len(pts0_show)

    left  = img_cf.copy()
    right = img_cs.copy()

    for i in range(n_show):
        t = i / max(n_show - 1, 1)
        rgba = cm.plasma(t)
        c = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))

        p_cf_h = np.array([pts0_show[i, 0], pts0_show[i, 1], 1.0])
        p_cs_h = np.array([pts1_show[i, 0], pts1_show[i, 1], 1.0])

        # cv2 convention: p_cs^T F p_cf = 0
        # epipolar line in cs image: l_cs = F * p_cf
        l_cs = F @ p_cf_h
        draw_line(right, l_cs, c)
        cv2.circle(right, (int(pts1_show[i, 0]), int(pts1_show[i, 1])), 6, c, -1, cv2.LINE_AA)

        # epipolar line in cf image: l_cf = F^T * p_cs
        l_cf = F.T @ p_cs_h
        draw_line(left, l_cf, c)
        cv2.circle(left, (int(pts0_show[i, 0]), int(pts0_show[i, 1])), 6, c, -1, cv2.LINE_AA)

    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]
    H = max(h_l, h_r)
    canvas = np.zeros((H + 70, w_l + 4 + w_r, 3), dtype=np.uint8)
    canvas[70:70+h_l, :w_l] = left
    canvas[70:70+h_r, w_l+4:] = right
    canvas[:70, :, :] = 30
    canvas[70:, w_l:w_l+4] = 40

    title = f'Group {gname} ({label}) | Epipolar line evaluation on pair {BEST_PAIR}'
    sub   = f'RANSAC inliers: {n_ransac}/{n_total}  |  Sampson<3px: {n_in3}/{len(sg08_cf)}  Sampson<10px: {n_in10}  |  Showing {n_show} matches'
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, sub,   (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    out_path = OUT_DIR / f'epipolar_group{gname}_08.jpg'
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'  Saved: {out_path}')

print('\nAll done.')
