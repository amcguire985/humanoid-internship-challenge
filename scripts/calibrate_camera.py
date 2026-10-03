"""Calibrate from chessboard photos; square sizes and translations use metres."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, default=root / "chessboard_calibration")
    p.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait", help="Calibrate one stored image orientation at a time")
    p.add_argument("--columns", type=int, default=6, help="Inner corners across board")
    p.add_argument("--rows", type=int, default=9, help="Inner corners down board")
    p.add_argument("--square-size", type=float, default=0.03, help="Square side in metres")
    p.add_argument("--output", type=Path, default=None)
    a = p.parse_args()
    if a.square_size <= 0 or min(a.columns, a.rows) < 2:
        p.error("Positive square size and at least two corners per axis required")
    if a.output is None:
        a.output = root / "results" / ("camera_calibration_" + a.orientation)
    pattern = (a.columns, a.rows)
    obj = np.zeros((a.columns * a.rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:a.columns, 0:a.rows].T.reshape(-1, 2)
    obj *= a.square_size
    paths = sorted(f for f in a.images.iterdir() if f.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff"})
    objects, points, names, skipped = [], [], [], []
    size = None
    a.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if frame is None:
            skipped.append({"image": path.name, "reason": "Unreadable image"})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = (gray.shape[1], gray.shape[0])
        if (current[1] > current[0]) != (a.orientation == "portrait"):
            skipped.append({"image": path.name, "reason": "Other image orientation"})
            continue
        if size is None:
            size = current
        elif current != size:
            raise ValueError(f"{path.name}: resolution {current} differs from {size}")
        found, corners = cv2.findChessboardCornersSB(gray, pattern, cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not found:
            skipped.append({"image": path.name, "reason": "Board not detected"})
            print(f"Skipped {path.name}", flush=True)
            continue
        objects.append(obj.copy())
        points.append(corners)
        names.append(path.name)
        cv2.drawChessboardCorners(frame, pattern, corners, found)
        if not cv2.imwrite(str(a.output / f"{path.stem}_corners.jpg"), frame):
            raise IOError("Could not save corner overlay")
        print(f"Detected {path.name}", flush=True)
    if len(names) < 3:
        raise RuntimeError(f"Only {len(names)} usable views; need at least 3 varied poses")
    rms, matrix, dist, rvecs, tvecs, std, _, errors = cv2.calibrateCameraExtended(objects, points, size, None, None)
    result = {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "distortion_order": ["k1", "k2", "p1", "p2", "k3"],
        "image_size": list(size),
        "image_orientation": "Stored pixels; EXIF orientation ignored",
        "inner_corners": list(pattern),
        "square_size_m": a.square_size,
        "rms_reprojection_error_px": float(rms),
        "intrinsic_standard_deviations": std.ravel().tolist(),
        "per_image": [
            {"image": name, "rms_reprojection_error_px": float(e[0]),
             "rotation_vector": r.ravel().tolist(), "translation_m": t.ravel().tolist()}
            for name, e, r, t in zip(names, errors, rvecs, tvecs)
        ],
        "skipped_images": skipped,
    }
    (a.output / "calibration.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez(a.output / "calibration.npz", camera_matrix=matrix, distortion_coefficients=dist,
             image_size=np.array(size), square_size_m=a.square_size,
             rotation_vectors=np.array(rvecs), translation_vectors=np.array(tvecs))
    print(f"Used {len(names)}/{len(paths)} images; RMS error: {rms:.4f} px")
    print("Camera matrix:\n", matrix)
    print("Distortion:", dist.ravel())

if __name__ == "__main__":
    main()

