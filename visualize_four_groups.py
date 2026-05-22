"""
visualize_four_groups.py

四組 F matrix 視覺化：
  A: 全部 20 對 SG（1920）合併 -> 全局 F -> viz_A_sg_all/
  B: 全部 20 對 SG + 全部人工標記點合併 -> 全局 F -> viz_B_sg_manual_all/
  C: pair 08（最佳單對）SG only -> F -> viz_C_sg_best/
  D: pair 08 SG + pair 08 人工標記點 -> F -> viz_D_sg_manual_best/

視覺化說明：
  - SG inlier 連線：plasma colormap 彩色圓點 + 連線
  - 人工標記點連線：白色圓點 + 連線
  - 用全局 F 的 Sampson distance 判斷 SG inlier（threshold=3.0）

Usage:
  .venv/Scripts/python.exe visualize_four_groups.py
"""

import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm


DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
OUTPUT_DIR    = DATASET_DIR / 'output_1920'
IMAGE_NAMES   = [f'{i:02d}.jpg' for i in range(1, 21)]
BEST_PAIR     = '08'
SAMPSON_THRESH = 3.0


# ── helpers ───────────────────────────────────────────────────────────────────

def load_superglue_matches(npz_path):
    data    = np.load(npz_path)
    kpts0   = data['keypoints0']
    kpts1   = data['keypoints1']
    matches = data['matches']
    valid   = matches > -1
    return kpts0[valid], kpts1[matches[valid]]


def load_manual_points():
    """Load all manual points from JSON. Returns dict keyed by stem (e.g. '01')."""
    with open(DATASET_DIR / 'selected_points_cf.json') as f:
        cf_data = json.load(f)
    with open(DATASET_DIR / 'selected_points_cs.json') as f:
        cs_data = json.load(f)
    result = {}
    for key in cf_data:
        stem = Path(key).stem
        result[stem] = (
            np.array(cf_data[key], dtype=np.float64),
            np.array(cs_data[key], dtype=np.float64)
        )
    return result


def compute_F(pts0, pts1):
    if len(pts0) < 8:
        return None
    F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, SAMPSON_THRESH, 0.99)
    return F


def sampson_distance(F, pts0, pts1):
    pts0h = np.hstack([pts0, np.ones((len(pts0), 1))])
    pts1h = np.hstack([pts1, np.ones((len(pts1), 1))])
    Fp0   = (F  @ pts0h.T).T
    FTp1  = (F.T @ pts1h.T).T
    num   = (pts1h * Fp0).sum(1) ** 2
    den   = Fp0[:,0]**2 + Fp0[:,1]**2 + FTp1[:,0]**2 + FTp1[:,1]**2
    return num / (den + 1e-10)


def inlier_mask_from_F(F, pts0, pts1, thresh=SAMPSON_THRESH):
    sd = sampson_distance(F, pts0, pts1)
    return sd < thresh


def plasma_color(i, n):
    rgba = cm.plasma(i / max(n - 1, 1))
    return (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_pair(img0, img1,
              sg_pts0, sg_pts1, sg_mask,
              manual_pts0, manual_pts1,
              title='', save_path=None):
    """
    Draw one image pair with:
      - SG inlier connections: plasma coloured circles + lines
      - Manual point connections: white circles + lines
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + 4 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]     = img0
    canvas[:h1, w0+4:]   = img1
    canvas[:, w0:w0+4]   = 40

    # SG outliers: tiny grey cross
    for (x0, y0), (x1, y1) in zip(sg_pts0[~sg_mask], sg_pts1[~sg_mask]):
        for px, py in [(int(x0), int(y0)), (int(x1)+w0+4, int(y1))]:
            cv2.drawMarker(canvas, (px, py), (50, 50, 50),
                           cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)

    # SG inliers: plasma coloured
    sg_in0 = sg_pts0[sg_mask]
    sg_in1 = sg_pts1[sg_mask]
    n_in = len(sg_in0)
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(sg_in0, sg_in1)):
        c = plasma_color(i, n_in)
        p0 = (int(x0), int(y0))
        p1 = (int(x1)+w0+4, int(y1))
        cv2.line(canvas, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 5, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 5, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, (255,255,255), 1, cv2.LINE_AA)

    # Manual points: white
    for (x0, y0), (x1, y1) in zip(manual_pts0, manual_pts1):
        p0 = (int(x0), int(y0))
        p1 = (int(x1)+w0+4, int(y1))
        cv2.line(canvas, p0, p1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 6, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 6, (0, 0, 0), 1, cv2.LINE_AA)

    # Header
    n_out = (~sg_mask).sum()
    pct   = 100 * n_in / (n_in + n_out) if (n_in + n_out) > 0 else 0
    header = f'{title}  |  SG inliers: {n_in}  outliers: {n_out}  ({pct:.1f}%)  manual: {len(manual_pts0)}'
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(canvas, header, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 80), 2, cv2.LINE_AA)

    # Legend
    cv2.circle(canvas, (canvas.shape[1]-180, 20), 6, (180, 0, 255), -1)
    cv2.putText(canvas, 'SG inlier', (canvas.shape[1]-168, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(canvas, (canvas.shape[1]-80, 20), 6, (255, 255, 255), -1)
    cv2.putText(canvas, 'Manual', (canvas.shape[1]-68, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(str(save_path), canvas)

    return canvas, n_in, len(sg_pts0)


# ── per-group runners ─────────────────────────────────────────────────────────

def collect_sg_all():
    """Collect all SG matches from 20 pairs."""
    all0, all1, per_pair = [], [], {}
    for name in IMAGE_NAMES:
        stem = Path(name).stem
        npz  = OUTPUT_DIR / f'{stem}_{stem}_matches.npz'
        if not npz.exists():
            continue
        pts0, pts1 = load_superglue_matches(npz)
        all0.append(pts0)
        all1.append(pts1)
        per_pair[stem] = (pts0, pts1)
    return np.vstack(all0), np.vstack(all1), per_pair


def run_group(label, out_dir, F_global, sg_per_pair, manual_all, pairs_to_viz, show_manual=False):
    """
    Render visualizations for one group.
    F_global: the group's global F (cf-first convention NOT applied here — used internally)
    sg_per_pair: dict stem -> (pts0, pts1) of SG matches
    manual_all: dict stem -> (pts0, pts1) of manual points
    pairs_to_viz: list of stems to render
    show_manual: whether to draw manual points (only for groups B and D)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_in = total_all = 0

    for stem in pairs_to_viz:
        img0 = cv2.imread(str(DATASET_DIR / 'cf' / f'{stem}.jpg'))
        img1 = cv2.imread(str(DATASET_DIR / 'cs' / f'{stem}.jpg'))
        if img0 is None or img1 is None:
            print(f'  [SKIP] image not found: {stem}')
            continue

        sg0, sg1 = sg_per_pair.get(stem, (np.empty((0,2)), np.empty((0,2))))

        if show_manual:
            man0, man1 = manual_all.get(stem, (np.empty((0,2)), np.empty((0,2))))
        else:
            man0, man1 = np.empty((0,2)), np.empty((0,2))

        if F_global is not None and len(sg0) > 0:
            mask = inlier_mask_from_F(F_global, sg0, sg1)
        else:
            mask = np.zeros(len(sg0), dtype=bool)

        save_path = out_dir / f'{stem}.jpg'
        _, n_in, n_all = draw_pair(img0, img1, sg0, sg1, mask,
                                   man0, man1,
                                   title=f'[{label}] Pair {stem}',
                                   save_path=save_path)
        total_in  += n_in
        total_all += n_all
        print(f'  [{label}] pair {stem}: {n_in} inliers / {n_all} SG matches  -> {save_path.name}')

    if total_all > 0:
        print(f'  [{label}] Total SG inliers: {total_in} / {total_all} ({100*total_in/total_all:.1f}%)')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading data...')
    manual_all  = load_manual_points()
    sg0_all, sg1_all, sg_per_pair = collect_sg_all()

    all_manual0 = np.vstack([v[0] for v in manual_all.values()])
    all_manual1 = np.vstack([v[1] for v in manual_all.values()])

    stems_all  = [Path(n).stem for n in IMAGE_NAMES if (OUTPUT_DIR / f'{Path(n).stem}_{Path(n).stem}_matches.npz').exists()]
    stems_best = [BEST_PAIR]

    # ── Group A: SG only, all 20 pairs ────────────────
    print('\n=== Group A: SG all 20 pairs ===')
    F_A = compute_F(sg0_all, sg1_all)
    n_in_A = inlier_mask_from_F(F_A, sg0_all, sg1_all).sum() if F_A is not None else 0
    print(f'  Global F inliers: {n_in_A} / {len(sg0_all)}')
    run_group('A', DATASET_DIR / 'viz_A_sg_all', F_A, sg_per_pair, manual_all, stems_all, show_manual=False)

    # ── Group B: SG + manual, all 20 pairs ────────────
    print('\n=== Group B: SG + manual all 20 pairs ===')
    pts0_B = np.vstack([sg0_all, all_manual0])
    pts1_B = np.vstack([sg1_all, all_manual1])
    F_B = compute_F(pts0_B, pts1_B)
    n_in_B = inlier_mask_from_F(F_B, sg0_all, sg1_all).sum() if F_B is not None else 0
    print(f'  Global F inliers (SG points only): {n_in_B} / {len(sg0_all)}')
    run_group('B', DATASET_DIR / 'viz_B_sg_manual_all', F_B, sg_per_pair, manual_all, stems_all, show_manual=True)

    # ── Group C: SG only, pair 08 ─────────────────────
    print('\n=== Group C: SG pair 08 only ===')
    sg0_C, sg1_C = sg_per_pair[BEST_PAIR]
    F_C = compute_F(sg0_C, sg1_C)
    n_in_C = inlier_mask_from_F(F_C, sg0_C, sg1_C).sum() if F_C is not None else 0
    print(f'  Global F inliers: {n_in_C} / {len(sg0_C)}')
    run_group('C', DATASET_DIR / 'viz_C_sg_best', F_C, sg_per_pair, manual_all, stems_best, show_manual=False)

    # ── Group D: SG + manual, pair 08 ────────────────
    print('\n=== Group D: SG + manual pair 08 ===')
    man0_08, man1_08 = manual_all[BEST_PAIR]
    pts0_D = np.vstack([sg0_C, man0_08])
    pts1_D = np.vstack([sg1_C, man1_08])
    F_D = compute_F(pts0_D, pts1_D)
    n_in_D = inlier_mask_from_F(F_D, sg0_C, sg1_C).sum() if F_D is not None else 0
    print(f'  Global F inliers (SG points only): {n_in_D} / {len(sg0_C)}')
    run_group('D', DATASET_DIR / 'viz_D_sg_manual_best', F_D, sg_per_pair, manual_all, stems_best, show_manual=True)

    # ── Save F matrices ───────────────────────────────
    print('\n=== Saving F matrices (cf-first convention) ===')
    for label, F, n_in, n_total, tag in [
        ('A', F_A, n_in_A, len(sg0_all),  'viz_A_sg_all'),
        ('B', F_B, n_in_B, len(pts0_B),   'viz_B_sg_manual_all'),
        ('C', F_C, n_in_C, len(sg0_C),    'viz_C_sg_best'),
        ('D', F_D, n_in_D, len(pts0_D),   'viz_D_sg_manual_best'),
    ]:
        if F is None:
            print(f'  [Group {label}] F is None, skip.')
            continue
        out_dir  = DATASET_DIR / tag
        out_path = out_dir / 'F_matrix.txt'
        F_out = F.T  # transpose to cf-first convention
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f'Convention: p_cf^T F p_cs = 0 (cf first)\n')
            f.write(f'Group {label}: {tag}\n')
            f.write(f'Total points used: {n_total}\n')
            f.write(f'RANSAC inliers (SG): {n_in} / {len(sg0_all) if label in "AB" else len(sg0_C)}\n\n')
            f.write('Fundamental Matrix F:\n')
            for row in F_out:
                f.write(f' [{row[0]:.8e}  {row[1]:.8e}  {row[2]:.8e}]\n')
        print(f'  [Group {label}] F saved -> {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
