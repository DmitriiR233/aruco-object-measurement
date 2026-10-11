"""Пороговая обработка, морфология и выбор контура объекта."""
from dataclasses import dataclass

import cv2
import numpy as np

from config import Params
from image_preprocessing import AUTO_CHANNELS, preprocess

REASON_TOUCH = ("объект не выделен: слабый контраст с фоном либо объект обрезан границей кадра")


class SegmentationError(Exception):
    """Не удалось выделить объект."""


@dataclass
class Segmentation:
    mask: np.ndarray       # бинарная маска объекта после морфологии (0/255)
    contour: np.ndarray    # внешний контур выбранного объекта
    channel: str = ""      # канал, на котором объект выделен
    contrast: float = float("nan")   # цветовой контраст объекта с фоном (ΔE в Lab)


def _odd(value: int) -> int:
    return int(value) | 1


def _fill_excluded(channel: np.ndarray, exclude: np.ndarray) -> np.ndarray:
    """Закрашивает исключённые области (маркер, пустые края) типичным значением фона."""
    valid = exclude == 0
    if not valid.any():
        raise SegmentationError("нет пикселей для анализа")
    filled = channel.copy()
    filled[~valid] = int(np.median(channel[valid]))
    return filled


def _object_is_dark(channel: np.ndarray, valid: np.ndarray) -> bool:
    """Объект предполагается меньше фона: тёмный класс по Отсу — меньшинство -> объект тёмный."""
    thr, _ = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_fraction = float((channel[valid] <= thr).mean())
    return dark_fraction < 0.5


def _threshold(channel: np.ndarray, exclude: np.ndarray, params: Params) -> np.ndarray:
    valid = exclude == 0
    filled = _fill_excluded(channel, exclude)
    dark = _object_is_dark(filled, valid)

    if params.method == "otsu":
        flag = cv2.THRESH_BINARY_INV if dark else cv2.THRESH_BINARY
        _, binary = cv2.threshold(filled, 0, 255, flag + cv2.THRESH_OTSU)
    elif params.method == "adaptive":
        block = params.adaptive_block or _odd(max(31, min(filled.shape[:2]) // 8))
        src = filled if dark else 255 - filled
        binary = cv2.adaptiveThreshold(
            src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            _odd(block), params.adaptive_c,
        )
    else:
        raise ValueError(f"неизвестный метод сегментации: {params.method}")

    binary[~valid] = 0
    return binary


def _fill_ratio(mask: np.ndarray, contour: np.ndarray) -> float:
    """Доля реально закрашенных пикселей маски внутри контура.

    У пустой рамки (например, артефакт на границе исключённой области вокруг маркера)
    контур охватывает большую площадь, но закрашенных пикселей внутри почти нет.
    """
    inside = np.zeros_like(mask)
    cv2.drawContours(inside, [contour], -1, 255, thickness=cv2.FILLED)
    area = cv2.countNonZero(inside)
    return cv2.countNonZero(cv2.bitwise_and(mask, inside)) / area if area else 0.0


def _clean(binary: np.ndarray, scale_px_per_mm: float, params: Params) -> np.ndarray:
    size = max(3, _odd(round(params.morph_kernel_mm * scale_px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)    # убирает мелкие помехи
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)     # закрывает мелкие разрывы


def _pick_contour(mask, exclude_mask, invalid_mask, valid_area, scale_px_per_mm, params):
    """Выбирает контур объекта. Возвращает (Segmentation | None, причина, маска «мусора»).

    «Мусор» — области, обрезанные границей кадра (клавиатура, тетрадь у края): их можно
    исключить из статистики порога и повторить сегментацию.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = params.min_object_area_mm2 * scale_px_per_mm ** 2
    contours = [c for c in contours if cv2.contourArea(c) >= min_area
                and _fill_ratio(mask, c) >= 0.3]   # отсекает тонкие «рамки-артефакты»
    if not contours:
        return None, "все найденные области слишком малы для объекта", None

    t = params.border_touch_px
    kernel = np.ones((2 * t + 1, 2 * t + 1), np.uint8)
    clutter = np.zeros_like(mask)
    reason = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) > params.max_object_area_fraction * valid_area:
            reason = reason or "выделенная область слишком велика: вероятно, выбран фон"
            continue
        obj = np.zeros_like(mask)
        cv2.drawContours(obj, [contour], -1, 255, thickness=cv2.FILLED)
        touches = obj[:t].any() or obj[-t:].any() or obj[:, :t].any() or obj[:, -t:].any()
        if touches or cv2.bitwise_and(cv2.dilate(obj, kernel), invalid_mask).any():
            clutter = cv2.bitwise_or(clutter, cv2.dilate(obj, kernel))
            reason = reason or REASON_TOUCH
            continue
        # объект должен быть похож на плотную фигуру: у «ветвистых» пятен фона (текстура, тени)
        # площадь заметно меньше площади описанного прямоугольника
        (_, (rw, rh), _) = cv2.minAreaRect(contour)
        if rw * rh > 0 and cv2.contourArea(contour) / (rw * rh) < params.min_extent:
            reason = reason or "найденная область не похожа на предмет (рваная форма)"
            continue
        return Segmentation(mask=obj, contour=contour), None, None
    return None, reason, clutter


def segment_object(
    channel: np.ndarray,
    exclude_mask: np.ndarray,
    invalid_mask: np.ndarray,
    scale_px_per_mm: float,
    params: Params,
    max_rounds: int = 3,
) -> Segmentation:
    """Выделяет объект на одном канале.

    exclude_mask — 255 там, где пиксели нужно игнорировать (маркер + область вне кадра);
    invalid_mask — 255 только вне кадра (нужна, чтобы понять, что объект обрезан краем).

    Если найдены только области, обрезанные краем кадра (тёмная клавиатура сдвигает порог Отсу
    и «прячет» сам объект), они исключаются из расчёта порога и сегментация повторяется.
    """
    valid_area = float((exclude_mask == 0).sum())
    exclude = exclude_mask.copy()
    reason = None
    for _ in range(max_rounds):
        mask = _clean(_threshold(channel, exclude, params), scale_px_per_mm, params)
        if cv2.countNonZero(mask) == 0:
            reason = reason or "маска пуста: объект не выделен"
            break
        found, why, clutter = _pick_contour(mask, exclude, invalid_mask, valid_area,
                                            scale_px_per_mm, params)
        if found is not None:
            return found
        reason = reason or why
        if clutter is None or not cv2.bitwise_and(clutter, cv2.bitwise_not(exclude)).any():
            break
        exclude = cv2.bitwise_or(exclude, clutter)
    raise SegmentationError(reason or "объект не найден")


def _contrast_de(image_bgr: np.ndarray, obj: np.ndarray, exclude_mask: np.ndarray,
                 scale_px_per_mm: float, params: Params) -> float:
    """Цветовой контраст (ΔE в Lab) между внутренностью объекта и узким кольцом вокруг него."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    def ellipse(mm: float) -> np.ndarray:
        size = max(3, _odd(round(mm * scale_px_per_mm)))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    inner = cv2.erode(obj, ellipse(1.0))
    ring = cv2.subtract(cv2.dilate(obj, ellipse(5.0)), cv2.dilate(obj, ellipse(1.5)))
    ring[exclude_mask > 0] = 0
    if not inner.any() or not ring.any():
        return 0.0
    return float(np.linalg.norm(lab[inner > 0].mean(axis=0) - lab[ring > 0].mean(axis=0)))


def segment_with_channels(image_bgr: np.ndarray, exclude_mask: np.ndarray, invalid_mask: np.ndarray,
                          scale_px_per_mm: float, params: Params) -> Segmentation:
    """Выделяет объект, перебирая каналы (при channel="auto").

    Объект ищется на каждом канале; кандидаты с достаточным цветовым контрастом сравниваются.
    Предпочтение у яркости (gray): она берётся, если её область не заметно меньше лучшей из
    остальных (иначе на яркости мог выделиться лишь фрагмент или посторонняя мелочь, а настоящий
    объект найден по цвету). Если не получилось ни на одном канале — SegmentationError с причиной.
    """
    channels = list(AUTO_CHANNELS) if params.channel == "auto" else [params.channel]
    reasons, low_contrast, candidates = [], None, []
    for name in channels:
        try:
            gray = preprocess(image_bgr, params, name, exclude_mask)
            seg = segment_object(gray, exclude_mask, invalid_mask, scale_px_per_mm, params)
        except SegmentationError as exc:
            reasons.append(str(exc))
            continue
        de = _contrast_de(image_bgr, seg.mask, exclude_mask, scale_px_per_mm, params)
        if de < params.min_contrast_de:
            low_contrast = low_contrast or (f"слабый контраст объекта с фоном "
                                            f"(ΔE = {de:.1f} < {params.min_contrast_de:.1f})")
            continue
        seg.channel, seg.contrast = name, de
        candidates.append((cv2.contourArea(seg.contour), seg))

    if not candidates:
        raise SegmentationError(low_contrast or (reasons[0] if reasons else "объект не найден"))
    best_area = max(area for area, _ in candidates)
    for area, seg in candidates:                      # порядок каналов = порядок предпочтения
        if area >= 0.8 * best_area:
            return seg
    return candidates[0][1]
