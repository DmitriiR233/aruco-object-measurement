"""Генерирует синтетические «фотографии» с известными размерами для самопроверки кода.

Сцена строится в миллиметрах на плоскости (маркер 50 мм + объект), затем «снимается»
камерой с заданным наклоном. Это НЕ замена реальному набору фотографий — только способ
убедиться, что код работает, и проверить гомографию на заведомо известной геометрии.

python tools/make_synthetic.py --out dataset_synthetic
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PX_PER_MM = 8
PLANE_W, PLANE_H = 360, 260  # мм


def draw_object(canvas, kind, center_mm, angle_deg, color):
    """Рисует объект на плоскости; возвращает (длина, ширина) в мм."""
    sizes = {"card": (86, 54), "pen": (142, 10), "plate": (65, 32), "box": (70, 45)}
    length, width = sizes[kind]
    c = (center_mm[0] * PX_PER_MM, center_mm[1] * PX_PER_MM)
    rect = (c, (length * PX_PER_MM, width * PX_PER_MM), angle_deg)
    cv2.fillConvexPoly(canvas, np.int32(cv2.boxPoints(rect)), color)
    return length, width


def make_plane(bg, light, kind, angle, rng):
    w, h = PLANE_W * PX_PER_MM, PLANE_H * PX_PER_MM
    colors = {"light": (225, 228, 230), "dark": (45, 40, 38), "colored": (60, 150, 190),
              "textured": (170, 175, 180)}
    canvas = np.full((h, w, 3), colors[bg], np.uint8)
    if bg == "textured":
        noise = rng.normal(0, 22, (h // 8, w // 8)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        canvas = np.clip(canvas.astype(np.float32) + noise[..., None], 0, 255).astype(np.uint8)

    # маркер 50 мм с белым полем 8 мм в точке (40, 40) мм
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    side = 50 * PX_PER_MM
    marker = cv2.aruco.generateImageMarker(dictionary, 70, side)  # id 70 = маркер по умолчанию в config.py
    pad = 8 * PX_PER_MM
    marker = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    x0, y0 = (40 - 8) * PX_PER_MM, (40 - 8) * PX_PER_MM
    canvas[y0:y0 + marker.shape[0], x0:x0 + marker.shape[1]] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

    obj_color = (35, 35, 35) if bg in ("light", "textured") else (235, 235, 235)
    length, width = draw_object(canvas, kind, (230, 140), angle, obj_color)

    if light == "shadow":      # мягкий градиент освещённости
        grad = np.linspace(1.0, 0.45, w, dtype=np.float32)[None, :, None]
        canvas = (canvas * grad).astype(np.uint8)
    elif light == "dim":
        canvas = (canvas * 0.45).astype(np.uint8)
    return canvas, length, width


def photograph(plane, tilt_deg, yaw_deg, out_size=(1600, 1200)):
    """Проецирует плоскость камерой с наклоном tilt (вокруг оси X) и поворотом yaw."""
    w, h = out_size
    f = 1500.0
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    t, y = np.radians(tilt_deg), np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rx @ Rz
    centre = np.array([PLANE_W / 2, PLANE_H / 2, 0.0])
    T = np.array([0, 0, 520.0]) - R @ centre                  # камера ~52 см от центра сцены
    H_mm = K @ np.column_stack([R[:, 0], R[:, 1], T])         # мм -> пиксели фото
    S = np.diag([1 / PX_PER_MM, 1 / PX_PER_MM, 1.0])          # пиксели плоскости -> мм
    # за краем «стола» фон продолжается (replicate), чтобы кадр целиком был занят фоном
    return cv2.warpPerspective(plane, H_mm @ S, out_size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("dataset_synthetic"))
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    (a.out / "images").mkdir(parents=True, exist_ok=True)

    scenarios = []
    for i, (kind, bg, light, tilt) in enumerate([
        ("card", "light", "even", 0), ("pen", "light", "even", 0), ("plate", "light", "even", 0),
        ("card", "light", "even", 20), ("pen", "light", "even", 35), ("box", "light", "even", 45),
        ("card", "dark", "even", 20), ("plate", "colored", "even", 20), ("pen", "textured", "even", 20),
        ("card", "light", "shadow", 20), ("plate", "light", "dim", 20), ("box", "textured", "shadow", 35),
    ]):
        scenarios.append((f"synth_{i + 1:02d}.jpg", kind, bg, light, tilt))

    rows = []
    for name, kind, bg, light, tilt in scenarios:
        plane, length, width = make_plane(bg, light, kind, float(rng.uniform(0, 90)), rng)
        photo = photograph(plane, tilt, float(rng.uniform(-10, 10)))
        photo = np.clip(photo.astype(np.float32) + rng.normal(0, 3, photo.shape), 0, 255).astype(np.uint8)
        photo = cv2.GaussianBlur(photo, (3, 3), 0)
        cv2.imwrite(str(a.out / "images" / name), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(dict(file=name, object=kind, length_mm=length, width_mm=width,
                         lighting=light, background=bg, tilt=f"{tilt} deg",
                         split="final" if len(rows) % 2 else "debug", expect_fail=0))

    # проблемный пример: маркер отсутствует -> программа должна отказаться от измерения
    plane, length, width = make_plane("light", "even", "card", 30.0, rng)
    plane[0:600, 0:600] = 225
    cv2.imwrite(str(a.out / "images" / "synth_13.jpg"), photograph(plane, 20, 0))
    rows.append(dict(file="synth_13.jpg", object="card", length_mm=length, width_mm=width,
                     lighting="even", background="light", tilt="20 deg", split="final", expect_fail=1))

    pd.DataFrame(rows).to_csv(a.out / "ground_truth.csv", index=False, encoding="utf-8-sig")
    print(f"Создано {len(rows)} изображений в {a.out}/images")


if __name__ == "__main__":
    main()
