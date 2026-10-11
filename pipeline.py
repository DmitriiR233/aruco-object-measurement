"""Полный конвейер: фото -> эталон -> (перспектива) -> сегментация -> размеры."""
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

import visualization as viz
from config import Params
from marker_detector import MarkerNotFoundError, detect_marker
from measurement import (build_original_view, build_warped_view, measure_contour)
from object_segmentation import SegmentationError, segment_with_channels

STATUS_OK = "ok"
STATUS_NO_MARKER = "marker_not_found"
STATUS_BAD_SEGMENTATION = "segmentation_failed"
STATUS_READ_ERROR = "read_error"


@dataclass
class MeasureResult:
    file: str
    status: str
    message: str = ""
    length_mm: float = float("nan")
    width_mm: float = float("nan")
    scale_px_per_mm: float = float("nan")
    channel: str = ""
    time_s: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    def as_dict(self) -> dict:
        return asdict(self)


def process_image(path: Path, params: Params, save_dir: Path | None = None) -> MeasureResult:
    """Обрабатывает одну фотографию. Никогда не бросает исключения по «обычным» причинам:
    ошибка превращается в статус, чтобы пакетная обработка не прерывалась."""
    path = Path(path)
    save_dir = Path(save_dir) if save_dir else None
    start = time.perf_counter()
    result = MeasureResult(file=path.name, status=STATUS_READ_ERROR)

    def finish(status: str, message: str = "") -> MeasureResult:
        result.status, result.message = status, message
        result.time_s = time.perf_counter() - start
        return result

    # 1. загрузка
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return finish(STATUS_READ_ERROR, "файл не открылся")
    stem = path.stem

    # 2. эталон — без него масштаб не считаем
    try:
        marker = detect_marker(image, params)
    except MarkerNotFoundError as exc:
        if save_dir:
            viz.save_image(save_dir / f"{stem}_failed.jpg", viz.draw_failure(image, str(exc)))
        return finish(STATUS_NO_MARKER, str(exc))
    if save_dir:
        viz.save_image(save_dir / f"{stem}_01_marker.jpg",
                       viz.draw_marker(image, marker.corners, marker.marker_id))

    # 3. кадр для измерения: выровненный (гомография) или исходный
    view = (build_warped_view if params.use_perspective else build_original_view)(
        image, marker.corners, params)
    result.scale_px_per_mm = view.scale_px_per_mm
    if save_dir and params.use_perspective:
        viz.save_image(save_dir / f"{stem}_02_warped.jpg", view.image)

    # 4-5. подготовка и сегментация
    try:
        seg = segment_with_channels(view.image, view.exclude_mask, view.invalid_mask,
                                    view.scale_px_per_mm, params)
    except SegmentationError as exc:
        if save_dir:
            viz.save_image(save_dir / f"{stem}_failed.jpg", viz.draw_failure(image, str(exc)))
        return finish(STATUS_BAD_SEGMENTATION, str(exc))

    # 6. контур -> размеры в мм (масштаб — именно того кадра, где измеряли)
    length_mm, width_mm, box = measure_contour(seg.contour, view.scale_px_per_mm)
    result.length_mm, result.width_mm, result.channel = length_mm, width_mm, seg.channel

    if save_dir:
        viz.save_image(save_dir / f"{stem}_03_mask.png", seg.mask)
        viz.save_image(save_dir / f"{stem}_04_contour.jpg",
                       viz.draw_contour_box(view.image, seg.contour, box))
        # финальная картинка рисуется на исходном фото: рамку переносим обратной гомографией
        box_orig = cv2.perspectiveTransform(box.reshape(-1, 1, 2).astype(np.float32),
                                            np.linalg.inv(view.homography)).reshape(-1, 2)
        viz.save_image(save_dir / f"{stem}_05_result.jpg",
                       viz.draw_result(image, box_orig, length_mm, width_mm))
    return finish(STATUS_OK)
