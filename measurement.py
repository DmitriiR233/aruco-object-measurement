"""Коррекция перспективы, вычисление масштаба и оценка размеров."""
from dataclasses import dataclass

import cv2
import numpy as np

from config import Params


@dataclass
class PlaneView:
    """Кадр, в котором считаются размеры (выровненный или исходный)."""

    image: np.ndarray
    scale_px_per_mm: float     # масштаб ИМЕННО этого кадра (пиксели на миллиметр)
    exclude_mask: np.ndarray   # 255 — игнорировать (маркер + пустые края)
    invalid_mask: np.ndarray   # 255 — область вне исходного кадра (только после warp)
    homography: np.ndarray     # исходный кадр -> этот кадр (3x3)
    marker_quad: np.ndarray    # положение маркера в этом кадре, (4, 2)


def native_scale(marker_corners: np.ndarray, params: Params) -> float:
    """k = P_r / L_r: пикселей на мм по средней стороне маркера в исходном кадре."""
    sides = np.linalg.norm(marker_corners - np.roll(marker_corners, -1, axis=0), axis=1)
    return float(sides.mean() / params.marker_size_mm)


def _marker_exclusion(shape, quad: np.ndarray, margin_px: float) -> np.ndarray:
    """Маска маркера, расширенная на margin_px (захватывает белую рамку вокруг маркера)."""
    center = quad.mean(axis=0)
    vec = quad - center
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    grown = center + vec * (1 + margin_px / norm)
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(grown).astype(np.int32), 255)
    return mask


def build_original_view(image: np.ndarray, marker_corners: np.ndarray, params: Params) -> PlaneView:
    """Вариант БЕЗ коррекции перспективы: масштаб по маркеру в исходном кадре."""
    k = native_scale(marker_corners, params)
    exclude = _marker_exclusion(image.shape, marker_corners, params.marker_mask_margin_mm * k)
    return PlaneView(image, k, exclude, np.zeros(image.shape[:2], np.uint8),
                     np.eye(3), marker_corners.copy())


def build_warped_view(image: np.ndarray, marker_corners: np.ndarray, params: Params) -> PlaneView:
    """Приводит плоскость маркера к виду сверху (гомография + warpPerspective).

    В плоскостных координатах (мм) маркер — квадрат со стороной marker_size_mm
    в точке (0, 0). После преобразования 1 мм = scale пикселей, причём scale
    известен точно и хранится отдельно от масштаба исходного кадра.
    """
    m = params.marker_size_mm
    plane_pts = np.array([[0, 0], [m, 0], [m, m], [0, m]], np.float32)
    h_mm = cv2.getPerspectiveTransform(marker_corners.astype(np.float32), plane_pts)  # px -> мм

    # какую часть плоскости видно камере (если какая-то точка «за горизонтом», берём окно целиком)
    ih, iw = image.shape[:2]
    frame = np.array([[0, 0], [iw, 0], [iw, ih], [0, ih]], np.float32).reshape(-1, 1, 2)
    w_vals = (h_mm @ np.c_[frame.reshape(-1, 2), np.ones(4)].T)[2]
    c, r = m / 2, max(params.warp_radius_mm, 8 * m)   # окно растёт вместе с размером маркера
    window = np.array([c - r, c - r, c + r, c + r])
    if (w_vals > 1e-6).all():
        foot = cv2.perspectiveTransform(frame, h_mm).reshape(-1, 2)
        box = np.array([*foot.min(axis=0), *foot.max(axis=0)])
        box = np.array([max(box[0], window[0]), max(box[1], window[1]),
                        min(box[2], window[2]), min(box[3], window[3])])
    else:
        box = window
    box[0], box[1] = min(box[0], 0), min(box[1], 0)           # маркер всегда в кадре
    box[2], box[3] = max(box[2], m), max(box[3], m)

    scale = min(native_scale(marker_corners, params),
                params.max_warp_side_px / float(max(box[2] - box[0], box[3] - box[1])))
    shift = np.array([[scale, 0, -box[0] * scale], [0, scale, -box[1] * scale], [0, 0, 1]])
    homography = shift @ h_mm
    out_w = int(np.ceil((box[2] - box[0]) * scale))
    out_h = int(np.ceil((box[3] - box[1]) * scale))

    warped = cv2.warpPerspective(image, homography, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    inside = cv2.warpPerspective(np.full((ih, iw), 255, np.uint8), homography, (out_w, out_h),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
    inside = cv2.erode(inside, np.ones((5, 5), np.uint8))      # край интерполяции ненадёжен
    invalid = cv2.bitwise_not(inside)

    quad = cv2.perspectiveTransform(marker_corners.reshape(-1, 1, 2).astype(np.float32),
                                    homography).reshape(-1, 2)
    exclude = cv2.bitwise_or(_marker_exclusion(warped.shape, quad,
                                               params.marker_mask_margin_mm * scale), invalid)
    return PlaneView(warped, float(scale), exclude, invalid, homography, quad)


def measure_contour(contour: np.ndarray, scale_px_per_mm: float):
    """Длина и ширина (мм) по минимальному повёрнутому прямоугольнику контура."""
    rect = cv2.minAreaRect(contour)
    side_a, side_b = rect[1]
    length_px, width_px = max(side_a, side_b), min(side_a, side_b)
    box = cv2.boxPoints(rect)
    return length_px / scale_px_per_mm, width_px / scale_px_per_mm, box
