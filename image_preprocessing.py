"""Фильтрация и выбор канала (цветового пространства) для сегментации."""
import cv2
import numpy as np

from config import Params

# Порядок перебора при channel="auto": сначала яркость, затем цветовое расстояние от фона,
# затем насыщенность. Берётся первый канал, на котором объект удалось выделить.
AUTO_CHANNELS = ("gray", "colordist", "saturation")


def _blur(image: np.ndarray, params: Params) -> np.ndarray:
    if params.blur_ksize and params.blur_ksize > 1:
        k = params.blur_ksize | 1  # ядро должно быть нечётным
        return cv2.GaussianBlur(image, (k, k), 0)
    return image


def preprocess(image_bgr: np.ndarray, params: Params, channel: str | None = None,
               exclude_mask: np.ndarray | None = None) -> np.ndarray:
    """Возвращает одноканальное сглаженное изображение для сегментации.

    gray       — яркость;
    saturation — канал S из HSV (цветные объекты на сером/белом фоне);
    colordist  — цветовое расстояние каждого пикселя до «типичного фона» в пространстве Lab
                 (объект яркий там, где его цвет отличается от фона; работает, когда яркость
                 объекта и фона совпадает, например коричневый картон на зелёном).
    exclude_mask — 255 там, где пиксели не учитываются при оценке цвета фона (маркер, пустые края).
    """
    channel = channel or params.channel
    if channel == "gray":
        return _blur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), params)
    if channel == "saturation":
        return _blur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 1], params)
    if channel == "colordist":
        lab = cv2.cvtColor(_blur(image_bgr, params), cv2.COLOR_BGR2LAB).astype(np.float32)
        valid = np.ones(lab.shape[:2], bool) if exclude_mask is None else exclude_mask == 0
        pixels = lab[valid][::7]                          # прореживание: медиане хватает
        background = np.median(pixels, axis=0) if len(pixels) else lab.reshape(-1, 3).mean(0)
        distance = np.linalg.norm(lab - background, axis=2)
        return np.clip(distance * 3.0, 0, 255).astype(np.uint8)
    raise ValueError(f"неизвестный канал: {channel}")
