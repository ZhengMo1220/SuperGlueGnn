"""
export_pose_matrices.py

輸出 RoMa V1 B_hires 和 LoFTR 的 E matrix、R、t 供開發人員使用。
同時輸出人工點基準的 E, R, t 作對照。

輸出檔案：
  pose_roma_bhires.txt   - RoMa V1 upsample_res=1024 的結果（最佳穩定性）
  pose_loftr.txt         - LoFTR conf>=0.5 的結果（Pitch 最接近基準）
  pose_manual.txt        - 人工點 Direct E 基準

Usage:
  .venv/Scripts/python.exe export_pose_matrices.py
"""

import numpy as np
import cv2
import json
import torch
from pathlib import Path

DATASET_DIR   = Path('Bullpen_Calibration/TSG_Bullpen')
INTRINSIC_DIR = Path('Bullpen_Calibration/Intrinsic')
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR       = Path('pose_output')
OUT_DIR.mkdir(exist_ok=True)


def load_K(path):
    text = path.read_text(encoding='utf-8')
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('['):
            continue
        clean = line.replace('[','').replace(']','').replace(',',' ')
        vals = [float(x) for x in clean.split()]
        if vals:
            rows.append(vals)
    return np.array(rows[:3], dtype=np.float64)


def rot_to_euler(R):
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    x  = np.degrees(np.arctan2( R[2,1],  R[2,2]))
    y  = np.degrees(np.arctan2(-R[2,0],  sy))
    z  = np.degrees(np.arctan2( R[1,0],  R[0,0]))
    return x, y, z


def compute_direct_E(pts_cf, pts_cs, K_cf, K_cs):
    """Direct findEssentialMat（normalize 後直接算 E）"""
    pts_cf_n = cv2.undistortPoints(pts_cf.reshape(-1,1,2), K_cf, None).reshape(-1,2)
    pts_cs_n = cv2.undistortPoints(pts_cs.reshape(-1,1,2), K_cs, None).reshape(-1,2)
    E, mask = cv2.findEssentialMat(pts_cf_n, pts_cs_n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    if E is None:
        return None
    if E.shape[0] > 3:
        E = E[:3]
    mask = mask.ravel().astype(bool)
    n_in, R, t, _ = cv2.recoverPose(E, pts_cf_n[mask], pts_cs_n[mask], np.eye(3))
    u, s, vt = np.linalg.svd(E)
    sv_ratio = s[0]/s[1] if s[1] > 1e-8 else 999
    rx, ry, rz = rot_to_euler(R)
    return {'E': E, 'R': R, 't': t, 'n_inliers': int(mask.sum()),
            'sv_ratio': sv_ratio, 'roll': rx, 'pitch': ry, 'yaw': rz}


def write_pose_file(path, title, results_list, K_cf, K_cs):
    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  {title}")
    lines.append(f"{'='*60}")
    lines.append(f"  世界座標原點：cs（投手區側面相機）")
    lines.append(f"  E matrix 定義：p_cs_norm^T · E · p_cf_norm = 0")
    lines.append(f"  R, t 定義：cf 的 Projection Matrix = K_cf · [R | t]")
    lines.append(f"             cs 的 Projection Matrix = K_cs · [I | 0]")
    lines.append(f"  triangulatePoints 呼叫方式：")
    lines.append(f"    cv2.triangulatePoints(K_cs·[I|0], K_cf·[R|t], p_cs_2d, p_cf_2d)")
    lines.append(f"")
    lines.append(f"  K_cf（cf 內參，1920×1084 座標系）：")
    for row in K_cf:
        lines.append(f"    {row[0]:12.4f}  {row[1]:12.4f}  {row[2]:12.4f}")
    lines.append(f"")
    lines.append(f"  K_cs（cs 內參，1920×1084 座標系）：")
    for row in K_cs:
        lines.append(f"    {row[0]:12.4f}  {row[1]:12.4f}  {row[2]:12.4f}")
    lines.append(f"")

    for entry in results_list:
        label = entry['label']
        res   = entry['result']
        lines.append(f"  {'─'*50}")
        lines.append(f"  {label}")
        lines.append(f"  {'─'*50}")
        if res is None:
            lines.append(f"  [失敗]")
            continue
        lines.append(f"  E-inliers : {res['n_inliers']}")
        lines.append(f"  sv_ratio  : {res['sv_ratio']:.6f}  (理想值 = 1.0)")
        lines.append(f"  Roll      : {res['roll']:8.2f}°")
        lines.append(f"  Pitch     : {res['pitch']:8.2f}°  (預期 ≈ −82°)")
        lines.append(f"  Yaw       : {res['yaw']:8.2f}°")
        lines.append(f"")
        lines.append(f"  E matrix (3×3)：")
        E = res['E']
        for row in E:
            lines.append(f"    {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")
        lines.append(f"")
        lines.append(f"  R matrix (3×3)：")
        R = res['R']
        for row in R:
            lines.append(f"    {row[0]:14.8f}  {row[1]:14.8f}  {row[2]:14.8f}")
        lines.append(f"")
        lines.append(f"  t vector (3×1)：")
        t = res['t'].ravel()
        lines.append(f"    {t[0]:14.8f}")
        lines.append(f"    {t[1]:14.8f}")
        lines.append(f"    {t[2]:14.8f}")
        lines.append(f"")

        # Projection matrices
        I0 = np.hstack([np.eye(3), np.zeros((3,1))])
        Rt = np.hstack([R, t.reshape(3,1)])
        PS = K_cs @ I0
        PF = K_cf @ Rt
        lines.append(f"  Projection Matrix PS = K_cs·[I|0]  (cs, 世界原點)：")
        for row in PS:
            lines.append(f"    {row[0]:14.4f}  {row[1]:14.4f}  {row[2]:14.4f}  {row[3]:14.4f}")
        lines.append(f"")
        lines.append(f"  Projection Matrix PF = K_cf·[R|t]  (cf)：")
        for row in PF:
            lines.append(f"    {row[0]:14.4f}  {row[1]:14.4f}  {row[2]:14.4f}  {row[3]:14.4f}")
        lines.append(f"")

    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  → 已寫入 {path}")


def main():
    print(f'Device: {DEVICE}')
    K_cf = load_K(INTRINSIC_DIR / 'Cf_Intrinsic.txt')
    K_cs = load_K(INTRINSIC_DIR / 'Cs_Intrinsic.txt')

    # ── 1. 人工點基準 ──────────────────────────────────────────
    print('\n[1/3] 人工點 Direct E...')
    with open(DATASET_DIR / 'selected_points_cf.json') as f:
        cf_dict = json.load(f)
    with open(DATASET_DIR / 'selected_points_cs.json') as f:
        cs_dict = json.load(f)
    # 合併所有 pair 的點
    all_cf, all_cs = [], []
    for key in sorted(cf_dict.keys()):
        if key in cs_dict:
            all_cf.extend(cf_dict[key])
            all_cs.extend(cs_dict[key])
    pts_cf_manual = np.array(all_cf, dtype=np.float64)
    pts_cs_manual = np.array(all_cs, dtype=np.float64)
    print(f"  人工點總數：cf={len(pts_cf_manual)}, cs={len(pts_cs_manual)}")
    res_manual = compute_direct_E(pts_cf_manual, pts_cs_manual, K_cf, K_cs)
    if res_manual:
        print(f"  人工點：Pitch={res_manual['pitch']:.2f}°  E-inliers={res_manual['n_inliers']}  sv_ratio={res_manual['sv_ratio']:.4f}")

    write_pose_file(
        OUT_DIR / 'pose_manual.txt',
        '人工標記點 Direct E（基準）',
        [{'label': '人工點（全 20 對，120pts）', 'result': res_manual}],
        K_cf, K_cs
    )

    # ── 2. RoMa V1 B_hires（upsample_res=1024）──────────────────
    print('\n[2/3] RoMa V1 B_hires（upsample_res=1024）全 20 對...')
    import os, ssl
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')
    ssl._create_default_https_context = ssl._create_unverified_context
    from romatch import roma_outdoor
    model = roma_outdoor(device=DEVICE, upsample_res=(1024, 1024))
    ssl._create_default_https_context = ssl._create_default_https_context

    roma_results = []
    pitches = []
    for i in range(1, 21):
        pair = f'{i:02d}'
        img_cf = str(DATASET_DIR / 'cf' / f'{pair}.jpg')
        img_cs = str(DATASET_DIR / 'cs' / f'{pair}.jpg')
        if not Path(img_cf).exists():
            continue
        img = cv2.imread(img_cf); H0, W0 = img.shape[:2]
        img = cv2.imread(img_cs); H1, W1 = img.shape[:2]
        with torch.no_grad():
            warp, certainty = model.match(img_cf, img_cs, device=DEVICE)
        matches, conf = model.sample(warp, certainty, num=5000)
        kp0, kp1 = model.to_pixel_coordinates(matches, H0, W0, H1, W1)
        kp0 = kp0.cpu().numpy(); kp1 = kp1.cpu().numpy(); conf = conf.cpu().numpy()
        mask = conf >= 0.5
        pts0, pts1 = kp0[mask], kp1[mask]
        res = compute_direct_E(pts0, pts1, K_cf, K_cs)
        label = f'Pair {pair}'
        if res:
            print(f"  [{pair}] Pitch={res['pitch']:7.2f}°  E-inliers={res['n_inliers']:4d}  sv_ratio={res['sv_ratio']:.4f}")
            pitches.append(res['pitch'])
        else:
            print(f"  [{pair}] 失敗")
        roma_results.append({'label': label, 'result': res})

    if pitches:
        print(f"  Pitch 統計：mean={np.mean(pitches):.1f}°  std={np.std(pitches):.1f}°  min={np.min(pitches):.1f}°  max={np.max(pitches):.1f}°")

    # 另外輸出所有對的中位數 E（取 pitch 最接近中位數的那對）
    if pitches:
        median_pitch = np.median(pitches)
        best_idx = np.argmin([abs(r['result']['pitch'] - median_pitch)
                              for r in roma_results if r['result'] is not None])
        valid_results = [r for r in roma_results if r['result'] is not None]
        best_pair_result = valid_results[best_idx]
        print(f"  中位數 Pitch={median_pitch:.1f}°，最接近的對：{best_pair_result['label']}")

    write_pose_file(
        OUT_DIR / 'pose_roma_bhires.txt',
        'RoMa V1 B_hires（upsample_res=1024，全 20 對）',
        roma_results,
        K_cf, K_cs
    )

    # ── 3. LoFTR conf>=0.3（640px，搭配 640px 對應的 K 縮放）────────
    print('\n[3/3] LoFTR conf>=0.3（pair 01~20，640×480）...')
    import kornia.feature as KF
    from kornia.color import rgb_to_grayscale
    import torchvision.transforms as T
    matcher = KF.LoFTR(pretrained='outdoor').to(DEVICE).eval()
    transform = T.Compose([T.ToTensor()])

    # LoFTR 在 640×480 下算，對應的 K 需按比例縮放
    LOFTR_W, LOFTR_H = 640, 480
    def scale_K(K, orig_w, orig_h, new_w, new_h):
        K2 = K.copy()
        K2[0,0] *= new_w / orig_w   # fx
        K2[1,1] *= new_h / orig_h   # fy
        K2[0,2] *= new_w / orig_w   # cx
        K2[1,2] *= new_h / orig_h   # cy
        return K2

    loftr_results = []
    for i in range(1, 21):
        pair = f'{i:02d}'
        img_cf_path = DATASET_DIR / 'cf' / f'{pair}.jpg'
        img_cs_path = DATASET_DIR / 'cs' / f'{pair}.jpg'
        if not img_cf_path.exists():
            continue
        img_cf_bgr = cv2.imread(str(img_cf_path))
        img_cs_bgr = cv2.imread(str(img_cs_path))
        H0_orig, W0_orig = img_cf_bgr.shape[:2]
        H1_orig, W1_orig = img_cs_bgr.shape[:2]

        img_cf_r = cv2.resize(img_cf_bgr, (LOFTR_W, LOFTR_H))
        img_cs_r = cv2.resize(img_cs_bgr, (LOFTR_W, LOFTR_H))

        t0 = transform(cv2.cvtColor(img_cf_r, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        t1 = transform(cv2.cvtColor(img_cs_r, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        t0 = rgb_to_grayscale(t0); t1 = rgb_to_grayscale(t1)

        with torch.no_grad():
            out = matcher({'image0': t0, 'image1': t1})
        kp0  = out['keypoints0'].cpu().numpy()
        kp1  = out['keypoints1'].cpu().numpy()
        conf = out['confidence'].cpu().numpy()

        # 用 640px 對應的縮放 K
        K_cf_s = scale_K(K_cf, W0_orig, H0_orig, LOFTR_W, LOFTR_H)
        K_cs_s = scale_K(K_cs, W1_orig, H1_orig, LOFTR_W, LOFTR_H)

        for thresh, label_suffix in [(0.5, 'conf>=0.5'), (0.3, 'conf>=0.3')]:
            mask = conf >= thresh
            if mask.sum() < 8:
                continue
            pts0 = kp0[mask]; pts1 = kp1[mask]
            res = compute_direct_E(pts0, pts1, K_cf_s, K_cs_s)
            if res:
                print(f"  [{pair}] {label_suffix}: Pitch={res['pitch']:7.2f}°  E-inliers={res['n_inliers']:4d}  pts={mask.sum()}")
                # 只記錄第一個成功的（優先 0.5）
                loftr_results.append({'label': f'Pair {pair} ({label_suffix})', 'result': res})
                break
        else:
            print(f"  [{pair}] 點數不足，跳過")
            loftr_results.append({'label': f'Pair {pair}', 'result': None})

    write_pose_file(
        OUT_DIR / 'pose_loftr.txt',
        'LoFTR outdoor conf>=0.5（全 20 對，640px）',
        loftr_results,
        K_cf, K_cs
    )

    print(f'\n全部完成。輸出檔案在 {OUT_DIR}/')
    print(f'  pose_manual.txt    - 人工點基準')
    print(f'  pose_roma_bhires.txt - RoMa V1 B_hires 全20對')
    print(f'  pose_loftr.txt     - LoFTR 全20對')


if __name__ == '__main__':
    main()
