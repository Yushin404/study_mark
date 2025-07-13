import cv2
import numpy as np
from pathlib import Path
import sys

SUBJECTS = ["国語", "数学", "英語", "理科", "他"]

def mm2px(mm, dpi=300):
    return int(round(mm / 25.4 * dpi))

def detect_and_warp(image, dpi=300):
    """ArUcoマーカーによるA4補正（300dpi）"""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        raise RuntimeError("ArUcoマーカーが4つ検出できません")

    ids = ids.flatten()
    id_map = {id_: corner[0] for id_, corner in zip(ids, corners)}
    try:
        tl = id_map[0][0]
        tr = id_map[1][1]
        br = id_map[2][2]
        bl = id_map[3][3]
    except KeyError:
        raise RuntimeError("マーカーID 0〜3 が揃っていません")

    w_px = mm2px(210, dpi)
    h_px = mm2px(297, dpi)
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [w_px-1, 0], [w_px-1, h_px-1], [0, h_px-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w_px, h_px))

def get_fixed_positions(dpi=300):
    """正確なX位置と、Yは1.5cm間隔で14行に並んだマスの座標群を返す"""
    rows = 14
    block_mm = 10.5
    row_spacing_mm = 16.5  # 1.5cm間隔
    top_margin_mm = 55.0

    # 中心X位置（mm）→ px（指定値）
    x_centers_mm = [29.0, 66.0, 105.0, 142.5, 180.0]
    x_centers_px = [mm2px(x, dpi) for x in x_centers_mm]
    block_px = mm2px(block_mm, dpi)

    # 各マスの上端Y座標
    y_tops_px = [mm2px(top_margin_mm + i * row_spacing_mm - 27.5 + i*0.2, dpi) for i in range(rows)]

    # 各教科列（x） × 各行（y） → マス位置リスト
    positions = []
    for x_center in x_centers_px:
        col = []
        for y in y_tops_px:
            x = x_center - block_px // 2
            col.append((x, y, block_px, block_px))
        positions.append(col)
    return positions

def analyze_boxes(image, boxes):
    """各マスの塗り判定とデバッグ描画"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    debug = image.copy()
    results = {}

    for col_idx, subject in enumerate(SUBJECTS):
        filled = 0
        for (x, y, w, h) in boxes[col_idx]:
            roi = bin_img[y:y+h, x:x+w]
            fill_ratio = np.sum(roi == 255) / (w * h)
            if fill_ratio > 0.3:
                filled += 1
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.rectangle(debug, (x, y), (x+w, y+h), (180, 180, 180), 1)
        results[subject] = filled * 15
    return results, debug

def main():
    if len(sys.argv) < 2:
        path_str = input("🖼️ 画像ファイルのパスを入力してください: ").strip()
    else:
        path_str = sys.argv[1]
    dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    img_path = Path(path_str)
    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ 画像を読み込めません:", img_path)
        return

    try:
        warped = detect_and_warp(img, dpi)
        boxes = get_fixed_positions(dpi)
        result, debug = analyze_boxes(warped, boxes)
    except Exception as e:
        print("⚠️ エラー:", e)
        return

    print("\n=== 📊 学習結果 ===")
    total = 0
    for subj in SUBJECTS:
        time = result[subj]
        total += time
        print(f"{subj:<4}: {time:3d} 分")
    print(f"🕒 合計 : {total:3d} 分")

    debug_path = img_path.with_stem(img_path.stem + "_debug")
    cv2.imwrite(str(debug_path), debug)
    print("💾 デバッグ画像を保存:", debug_path)

if __name__ == "__main__":
    main()
