"""Обнаружение эталона ArUco и проверка результата."""
from dataclasses import dataclass

import cv2
import numpy as np

from config import Params


# Стандартные словари, которые перебираются при aruco_dict="auto" (самые частые сначала).
AUTO_DICTS = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]
_last_dict: list[str] = []   # словарь, найденный на предыдущем фото, пробуем первым (ускоряет пакетную обработку)


class MarkerNotFoundError(Exception):
    """Маркер не найден или найден некорректно."""


@dataclass
class MarkerDetection:
    corners: np.ndarray   # (4, 2) float32, порядок: верх-лево, верх-право, низ-право, низ-лево
    marker_id: int
    side_px: float        # средняя длина стороны маркера в пикселях исходного кадра
    dict_name: str = ""   # словарь, в котором маркер найден


def _get_detector(params: Params, dict_name: str) -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    det_params = cv2.aruco.DetectorParameters()
    methods = {"subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
               "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
               "contour": cv2.aruco.CORNER_REFINE_CONTOUR}
    det_params.cornerRefinementMethod = methods[params.corner_refinement]
    return cv2.aruco.ArucoDetector(dictionary, det_params)


def _side_lengths(corners: np.ndarray) -> np.ndarray:
    return np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)


def _is_convex(corners: np.ndarray) -> bool:
    return bool(cv2.isContourConvex(corners.reshape(-1, 1, 2).astype(np.float32)))


def _detect_once(gray: np.ndarray, params: Params, dict_name: str, use_clahe: bool = True):
    """Один проход детектора с выравниванием контраста как запасным вариантом."""
    detector = _get_detector(params, dict_name)
    attempts = [gray]
    if use_clahe:
        attempts.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
    for attempt in attempts:
        found_corners, found_ids, _ = detector.detectMarkers(attempt)
        if found_ids is not None and len(found_ids) > 0:
            return list(found_corners), found_ids.flatten()
    return [], None


def _best_in_dict(gray: np.ndarray, params: Params, dict_name: str, use_clahe: bool = True,
                  min_side_px: float | None = None) -> MarkerDetection | None:
    min_side = params.min_marker_side_px if min_side_px is None else min_side_px
    candidates, ids = _detect_once(gray, params, dict_name, use_clahe)
    if ids is None:
        return None
    best = None
    for corners, marker_id in zip(candidates, ids):
        if params.marker_id is not None and int(marker_id) != params.marker_id:
            continue
        pts = corners.reshape(4, 2).astype(np.float32)
        side = float(_side_lengths(pts).mean())
        if side < min_side or not _is_convex(pts):
            continue                      # мелкие ложные срабатывания не принимаем
        if best is None or side > best.side_px:
            best = MarkerDetection(pts, int(marker_id), side, dict_name)
    return best


def detect_marker(image_bgr: np.ndarray, params: Params) -> MarkerDetection:
    """Находит ArUco-маркер. Бросает MarkerNotFoundError, если надёжно найти не удалось.

    При aruco_dict="auto" перебираются стандартные словари — задавать словарь и id вручную не нужно.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if params.aruco_dict == "auto":
        names = _last_dict + [n for n in AUTO_DICTS if n not in _last_dict]
    else:
        names = [params.aruco_dict]

    best = None
    if params.aruco_dict == "auto" and not _last_dict:
        # первый кадр: смотрим все словари на уменьшенной копии (быстро) и берём самый крупный
        # маркер — реальный маркер крупнее случайных ложных срабатываний
        k = min(1.0, 1200.0 / max(gray.shape))
        small = cv2.resize(gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else gray
        scan_best = None
        for name in names:
            found = _best_in_dict(small, params, name, use_clahe=False,
                                  min_side_px=params.min_marker_side_px * k)
            if found is not None and (scan_best is None or found.side_px > scan_best.side_px):
                scan_best = found
        if scan_best is not None:
            best = _best_in_dict(gray, params, scan_best.dict_name)   # точный расчёт в полном размере
    else:
        for name in names:
            best = _best_in_dict(gray, params, name)
            if best is not None:
                break
    if best is None:
        raise MarkerNotFoundError("ArUco-маркер не найден или слишком мал"
                                  + (f" (id={params.marker_id})" if params.marker_id is not None else ""))
    if params.aruco_dict == "auto":
        _last_dict[:] = [best.dict_name]

    if best.side_px < params.min_marker_side_px:
        raise MarkerNotFoundError(
            f"маркер слишком мал ({best.side_px:.0f} px < {params.min_marker_side_px:.0f} px)"
        )
    if not _is_convex(best.corners):
        raise MarkerNotFoundError("углы маркера образуют невыпуклый четырёхугольник")

    # Контроль уточнения углов: если «тонкий» метод сильно разошёлся с базовым SUBPIX,
    # углы ненадёжны (например, лишний контур на тёмном фоне) — используем SUBPIX.
    if params.corner_refinement != "subpix":
        from dataclasses import replace
        base_corners, base_ids = _detect_once(gray, replace(params, corner_refinement="subpix"),
                                              best.dict_name)
        if base_ids is not None:
            for corners, marker_id in zip(base_corners, base_ids):
                if int(marker_id) == best.marker_id:
                    base = corners.reshape(4, 2).astype(np.float32)
                    if np.abs(base - best.corners).max() > 0.02 * best.side_px:
                        best = MarkerDetection(base, best.marker_id,
                                               float(_side_lengths(base).mean()), best.dict_name)
                    break
    return best
