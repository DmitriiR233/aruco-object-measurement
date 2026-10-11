"""Запуск измерения: одна фотография или целая папка.

Примеры:
    python main.py --input dataset/images/photo_01.jpg
    python main.py --input dataset/images --output results --method adaptive
    python main.py --input dataset/images --no-perspective
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from dataset_io import read_table

from config import DEFAULT_PARAMS, GROUND_TRUTH_CSV, IMAGE_EXTENSIONS, IMAGES_DIR, RESULTS_DIR
from pipeline import process_image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Измерение размеров плоского объекта по эталону ArUco")
    p.add_argument("--input", type=Path, default=IMAGES_DIR, help="файл или папка с фотографиями")
    p.add_argument("--output", type=Path, default=RESULTS_DIR, help="папка для результатов")
    p.add_argument("--method", choices=["otsu", "adaptive"], default=DEFAULT_PARAMS.method)
    p.add_argument("--channel", choices=["auto", "gray", "colordist", "saturation"],
                   default=DEFAULT_PARAMS.channel, help="канал сегментации; auto — подобрать")
    p.add_argument("--marker-mm", type=float, default=DEFAULT_PARAMS.marker_size_mm,
                   help="реальная сторона маркера, мм")
    p.add_argument("--dict", default=DEFAULT_PARAMS.aruco_dict,
                   help='словарь ArUco; "auto" (по умолчанию) — подобрать автоматически')
    p.add_argument("--marker-id", type=int, default=DEFAULT_PARAMS.marker_id,
                   help="id маркера (по умолчанию — любой)")
    p.add_argument("--sizes", type=Path, default=GROUND_TRUTH_CSV,
                   help="CSV со столбцами file и marker_mm: реальный размер маркера для каждого фото")
    p.add_argument("--no-perspective", action="store_true", help="без коррекции перспективы")
    p.add_argument("--no-steps", action="store_true", help="не сохранять промежуточные изображения")
    return p.parse_args()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)


def load_marker_sizes(csv_path: Path) -> dict[str, float]:
    """file -> реальный размер маркера, мм (столбец marker_mm в ground_truth.csv)."""
    if not csv_path or not csv_path.exists():
        print(f"Таблица {csv_path} не найдена: для всех фото используется marker_size_mm из config.py")
        return {}
    table = read_table(csv_path)
    if "marker_mm" not in table.columns or "file" not in table.columns:
        print(f"ВНИМАНИЕ: в {csv_path} нет столбцов file и marker_mm "
              f"(найдены: {', '.join(table.columns)}). Размер маркера берётся из config.py.")
        return {}
    table = table.dropna(subset=["marker_mm"])
    print(f"Размер маркера из таблицы задан для {len(table)} фото.")
    return dict(zip(table["file"], table["marker_mm"].astype(float)))


def main() -> int:
    args = parse_args()
    params = replace(DEFAULT_PARAMS, method=args.method, channel=args.channel,
                     marker_size_mm=args.marker_mm, aruco_dict=args.dict,
                     marker_id=args.marker_id, use_perspective=not args.no_perspective)

    if not args.input.exists():
        print(f"Путь не найден: {args.input}")
        return 1
    images = collect_images(args.input)
    if not images:
        print(f"В {args.input} нет изображений")
        return 1

    sizes = load_marker_sizes(args.sizes)
    steps_dir = None if args.no_steps else args.output / "steps"
    rows = []
    for image_path in images:
        photo_params = replace(params, marker_size_mm=sizes.get(image_path.name, params.marker_size_mm))
        res = process_image(image_path, photo_params, steps_dir)
        rows.append({**res.as_dict(), "marker_mm": photo_params.marker_size_mm})
        if res.ok:
            print(f"{res.file}: длина {res.length_mm:.1f} мм, ширина {res.width_mm:.1f} мм "
                  f"[маркер {photo_params.marker_size_mm:g} мм, канал {res.channel}] ({res.time_s:.2f} с)")
        else:
            print(f"{res.file}: НЕ ИЗМЕРЕНО — {res.message}")

    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.round(3).to_csv(args.output / "measurements.csv", index=False, encoding="utf-8-sig")
    print(f"\nУспешно: {int((table['status'] == 'ok').sum())} из {len(table)}. "
          f"Таблица: {args.output / 'measurements.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
