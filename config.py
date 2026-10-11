"""Параметры алгоритма и пути проекта."""
from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "dataset" / "images"
GROUND_TRUTH_CSV = BASE_DIR / "dataset" / "ground_truth.csv"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class Params:
    """Все настраиваемые параметры конвейера в одном месте."""

    # --- эталон ArUco ---
    marker_size_mm: float = 50.0          # сторона чёрного квадрата маркера по умолчанию; для отдельных
                                          # фото можно задать свою в столбце marker_mm файла ground_truth.csv
    aruco_dict: str = "auto"              # "auto" — перебрать стандартные словари; или имя, например DICT_5X5_100
    marker_id: int | None = None          # None -> любой id; число -> только маркер с этим id
    corner_refinement: str = "apriltag"   # "subpix", "apriltag" или "contour" — уточнение углов маркера
    min_marker_side_px: float = 40.0      # слишком маленький маркер -> большая ошибка масштаба

    # --- коррекция перспективы ---
    use_perspective: bool = True
    warp_radius_mm: float = 400.0         # область плоскости вокруг маркера, которую разворачиваем
    max_warp_side_px: int = 3500          # ограничение размера выровненного кадра

    # --- подготовка изображения ---
    channel: str = "auto"                 # "auto" (перебор каналов), "gray", "colordist" или "saturation"
    min_contrast_de: float = 5.0          # минимальный цветовой контраст объекта с фоном (ΔE в Lab)
    blur_ksize: int = 5                   # ядро GaussianBlur (нечётное; 0 -> без сглаживания)

    # --- сегментация ---
    method: str = "otsu"                  # "otsu" или "adaptive"
    adaptive_block: int = 0               # 0 -> подобрать автоматически от размера кадра
    adaptive_c: float = 7.0
    morph_kernel_mm: float = 0.6          # размер ядра морфологии в миллиметрах
    marker_mask_margin_mm: float = 10.0    # отступ вокруг маркера, который исключается из маски

    # --- проверка найденного объекта ---
    min_object_area_mm2: float = 100.0
    max_object_area_fraction: float = 0.6  # доля площади кадра; больше -> считаем, что выделен фон
    border_touch_px: int = 3
    min_extent: float = 0.55               # площадь контура / площадь его повёрнутого прямоугольника


DEFAULT_PARAMS = Params()
