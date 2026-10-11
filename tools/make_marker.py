"""Создаёт PNG с ArUco-маркером для печати.

python tools/make_marker.py --id 0 --out marker_id0.png
Печатайте в масштабе 100% (без «вписать в страницу») и линейкой проверьте, что
чёрный квадрат имеет сторону 50 мм (значение --marker-mm / config.marker_size_mm).
"""
import argparse

import cv2


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--dict", default="DICT_5X5_100")
    p.add_argument("--side-px", type=int, default=800, help="сторона чёрного квадрата, px")
    p.add_argument("--out", default="marker.png")
    a = p.parse_args()

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict))
    marker = cv2.aruco.generateImageMarker(dictionary, a.id, a.side_px)
    margin = a.side_px // 5                      # белое поле вокруг маркера обязательно
    marker = cv2.copyMakeBorder(marker, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(a.out, marker)
    print(f"Сохранено: {a.out}")


if __name__ == "__main__":
    main()
