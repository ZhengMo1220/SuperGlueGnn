"""
visualize_inliers.py

三種模式：
  1. 預設：只畫 RANSAC inlier 連線（inlier-only，清楚看對應關係）
  2. --epipolar：畫 inlier 連線 + epipolar line（幾何驗證，確認點真的在線上）
  3. --all_in_one：輸出 4xN 總覽圖

Usage:
  .venv/Scripts/python.exe visualize_inliers.py                   # 全部 20 對，只畫 inlier
  .venv/Scripts/python.exe visualize_inliers.py --pair 01         # 只看第 01 對
  .venv/Scripts/python.exe visualize_inliers.py --epipolar        # 加 epipolar line
  .venv/Scripts/python.exe visualize_inliers.py --all_in_one      # 總覽圖
"""

import argparse
import numpy as np
import cv2
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm


DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
OUTPUT_DIR    = DATASET_DIR / 'output_1920'
VIZ_DIR       = DATASET_DIR / 'inlier_viz'
IMAGE_NAMES   = [f'{i:02d}.jpg' for i in range(1, 21)]
RANSAC_THRESH = 3.0


# ── helpers ───────────────────────────────────────────────────────────────────

def load_superglue_matches(npz_path):
    data    = np.load(npz_path)
    kpts0   = data['keypoints0']
    kpts1   = data['keypoints1']
    matches = data['matches']
    conf    = data['match_confidence']
    valid   = matches > -1
    return kpts0[valid], kpts1[matches[valid]], conf[valid]


def compute_F_and_inliers(mkpts0, mkpts1):
    if len(mkpts0) < 8:
        return None, np.zeros(len(mkpts0), dtype=bool)
    F, mask = cv2.findFundamentalMat(
        mkpts0, mkpts1,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=RANSAC_THRESH,
        confidence=0.99
    )
    if mask is None:
        return None, np.zeros(len(mkpts0), dtype=bool)
    return F, mask.ravel().astype(bool)


def jet_color(i, n):
    """Return BGR colour from plasma colormap, i-th out of n."""
    rgba = cm.plasma(i / max(n - 1, 1))
    return (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))


# ── drawing ───────────────────────────────────────────────────────────────────

def draw_inliers_only(img0, img1, pts0, pts1, inlier_mask, title='', save_path=None):
    """
    Left = cf, Right = cs.
    Only RANSAC inliers are drawn (coloured lines + dots).
    A small red cross marks rejected outlier positions so you can see what was filtered.
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1 + 4, 3), dtype=np.uint8)
    canvas[:h0, :w0]       = img0
    canvas[:h1, w0+4:]     = img1

    n_in  = inlier_mask.sum()
    n_out = (~inlier_mask).sum()

    # outlier positions — tiny grey cross, no line
    for (x0, y0), (x1, y1) in zip(pts0[~inlier_mask], pts1[~inlier_mask]):
        for px, py in [(int(x0), int(y0)), (int(x1)+w0+4, int(y1))]:
            cv2.drawMarker(canvas, (px, py), (60, 60, 60),
                           cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)

    # inlier connections — coloured, thick
    for i, ((x0, y0), (x1, y1)) in enumerate(zip(pts0[inlier_mask], pts1[inlier_mask])):
        c = jet_color(i, n_in)
        p0 = (int(x0),        int(y0))
        p1 = (int(x1)+w0+4,   int(y1))
        cv2.line(canvas, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 5, c, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, c, -1, cv2.LINE_AA)
        # white border on dot for contrast
        cv2.circle(canvas, p0, 5, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 5, (255,255,255), 1, cv2.LINE_AA)

    # divider
    canvas[:, w0:w0+4] = 40

    # header
    pct = 100 * n_in / (n_in + n_out) if (n_in + n_out) > 0 else 0
    header = f'{title}  |  INLIERS: {n_in}  outliers: {n_out}  ({pct:.1f}%)'
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(canvas, header, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 80), 2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(str(save_path), canvas)
        print(f'  Saved: {save_path.name}  ({n_in} inliers / {n_in+n_out} matches, {pct:.1f}%)')

    return canvas


def draw_epipolar(img0, img1, pts0, pts1, F, inlier_mask, title='', save_path=None):
    """
    Draw inlier matches AND the epipolar lines for geometric verification.
    For each inlier point in img0, compute its epipolar line in img1 and draw it.
    A good inlier's corresponding point in img1 should lie exactly on that line.
    """
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)

    left  = img0.copy()
    right = img1.copy()

    ipts0 = pts0[inlier_mask]
    ipts1 = pts1[inlier_mask]
    n_in  = len(ipts0)

    # compute epipolar lines in both images
    lines1 = cv2.computeCorrespondEpilines(ipts0.reshape(-1,1,2), 1, F).reshape(-1, 3)
    lines0 = cv2.computeCorrespondEpilines(ipts1.reshape(-1,1,2), 2, F).reshape(-1, 3)

    for i in range(n_in):
        c = jet_color(i, n_in)

        # epipolar line in right image (from left point)
        a, b, cc = lines1[i]
        x0r, y0r = 0, int(-cc / b) if abs(b) > 1e-6 else 0
        x1r, y1r = w1, int(-(cc + a*w1) / b) if abs(b) > 1e-6 else h1
        cv2.line(right, (x0r, y0r), (x1r, y1r), c, 1, cv2.LINE_AA)

        # epipolar line in left image (from right point)
        a, b, cc = lines0[i]
        x0l, y0l = 0, int(-cc / b) if abs(b) > 1e-6 else 0
        x1l, y1l = w0, int(-(cc + a*w0) / b) if abs(b) > 1e-6 else h0
        cv2.line(left, (x0l, y0l), (x1l, y1l), c, 1, cv2.LINE_AA)

        # keypoints
        cv2.circle(left,  (int(ipts0[i,0]), int(ipts0[i,1])), 6, c, -1, cv2.LINE_AA)
        cv2.circle(right, (int(ipts1[i,0]), int(ipts1[i,1])), 6, c, -1, cv2.LINE_AA)
        cv2.circle(left,  (int(ipts0[i,0]), int(ipts0[i,1])), 6, (255,255,255), 1, cv2.LINE_AA)
        cv2.circle(right, (int(ipts1[i,0]), int(ipts1[i,1])), 6, (255,255,255), 1, cv2.LINE_AA)

    canvas = np.zeros((H, w0 + 4 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0]     = left
    canvas[:h1, w0+4:]   = right
    canvas[:, w0:w0+4]   = 40

    header = f'{title}  |  Epipolar verification  ({n_in} inliers)  — coloured dot should lie ON its coloured line'
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (20, 20, 20), -1)
    cv2.putText(canvas, header, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2, cv2.LINE_AA)

    if save_path:
        cv2.imwrite(str(save_path), canvas)
        print(f'  Saved: {save_path.name}')

    return canvas


# ── main runners ──────────────────────────────────────────────────────────────

def process_pair(stem, epipolar=False, save_dir=None):
    npz_path = OUTPUT_DIR / f'{stem}_{stem}_matches.npz'
    if not npz_path.exists():
        print(f'  [SKIP] {npz_path.name} not found')
        return None, 0, 0

    img0 = cv2.imread(str(DATASET_DIR / 'cf' / f'{stem}.jpg'))
    img1 = cv2.imread(str(DATASET_DIR / 'cs' / f'{stem}.jpg'))
    if img0 is None or img1 is None:
        print(f'  [SKIP] image not found for {stem}')
        return None, 0, 0

    mkpts0, mkpts1, _ = load_superglue_matches(npz_path)
    F, mask = compute_F_and_inliers(mkpts0, mkpts1)

    tag   = 'epipolar' if epipolar else 'inliers'
    spath = (save_dir / f'{tag}_{stem}.jpg') if save_dir else None

    if epipolar and F is not None:
        frame = draw_epipolar(img0, img1, mkpts0, mkpts1, F, mask,
                              title=f'Pair {stem}', save_path=spath)
    else:
        frame = draw_inliers_only(img0, img1, mkpts0, mkpts1, mask,
                                  title=f'Pair {stem}', save_path=spath)

    return frame, int(mask.sum()), len(mask)


def run_single(pair_name, epipolar=False):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    stem = pair_name.zfill(2) if '.' not in pair_name else Path(pair_name).stem
    process_pair(stem, epipolar=epipolar, save_dir=VIZ_DIR)


def run_all(epipolar=False, all_in_one=False):
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    total_in = total_all = 0

    for name in IMAGE_NAMES:
        stem = Path(name).stem
        save_dir = None if all_in_one else VIZ_DIR
        frame, n_in, n_all = process_pair(stem, epipolar=epipolar, save_dir=save_dir)
        if frame is not None:
            frames.append((stem, frame))
            total_in  += n_in
            total_all += n_all

    print(f'\nTotal: {total_in} inliers / {total_all} matches ({100*total_in/total_all:.1f}%)')

    if all_in_one and frames:
        cols  = 4
        rows  = (len(frames) + cols - 1) // cols
        h, w  = frames[0][1].shape[:2]
        scale = 0.32
        th, tw = int(h * scale), int(w * scale)

        grid = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
        for idx, (stem, frame) in enumerate(frames):
            r, c = divmod(idx, cols)
            grid[r*th:(r+1)*th, c*tw:(c+1)*tw] = cv2.resize(frame, (tw, th))

        tag      = 'epipolar' if epipolar else 'inliers'
        out_path = VIZ_DIR / f'{tag}_all.jpg'
        cv2.imwrite(str(out_path), grid)
        print(f'  Saved overview: {out_path}')


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair',       type=str,           help='只處理單一對，例如 --pair 01')
    parser.add_argument('--epipolar',   action='store_true', help='畫 epipolar line 幾何驗證圖')
    parser.add_argument('--all_in_one', action='store_true', help='輸出 4xN 總覽圖')
    opt = parser.parse_args()

    if opt.pair:
        run_single(opt.pair, epipolar=opt.epipolar)
    else:
        run_all(epipolar=opt.epipolar, all_in_one=opt.all_in_one)
