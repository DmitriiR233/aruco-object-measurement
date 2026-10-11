"""Оценка точности по dataset/ground_truth.csv.

Столбцы ground_truth.csv:
    file, object, length_mm, width_mm            — обязательные
    lighting, background, tilt                   — условия съёмки (для графиков по группам)
    split                                        — debug / final (необязательно)
    marker_mm                                    — реальная сторона чёрного квадрата маркера на этом фото,
                                                   мм (пусто -> значение marker_size_mm из config.py)
    expect_fail                                  — 1, если программа ДОЛЖНА отказаться
                                                   (маркер закрыт и т.п.), иначе 0

Пример:
    python evaluate.py --split final
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import visualization as viz
from config import DEFAULT_PARAMS, GROUND_TRUTH_CSV, IMAGES_DIR, RESULTS_DIR
from dataset_io import read_table
from pipeline import process_image

# Варианты алгоритма для сравнения (раздел 7.3 плана)
CONFIGS = {
    "otsu_persp": dict(method="otsu", use_perspective=True),
    "otsu_gray_persp": dict(method="otsu", use_perspective=True, channel="gray"),
    "otsu_noperspective": dict(method="otsu", use_perspective=False),
    "adaptive_persp": dict(method="adaptive", use_perspective=True),
    "adaptive_noperspective": dict(method="adaptive", use_perspective=False),
}
GROUP_COLUMNS = {
    "lighting": "MAE по условиям освещения",
    "background": "MAE по типу фона",
    "tilt": "MAE по углу съёмки",
    "object": "MAE по типу объекта",
}


def run_config(name: str, gt: pd.DataFrame, images_dir: Path, steps_dir: Path | None):
    base_params = replace(DEFAULT_PARAMS, **CONFIGS[name])
    rows = []
    for rec in gt.itertuples(index=False):
        params = base_params
        marker_mm = getattr(rec, "marker_mm", float("nan"))   # свой размер маркера для этого фото
        if marker_mm == marker_mm:                             # не NaN
            params = replace(base_params, marker_size_mm=float(marker_mm))
        res = process_image(images_dir / rec.file, params,
                            steps_dir / name if steps_dir else None)
        base = {c: getattr(rec, c) for c in gt.columns}
        base.update(config=name, status=res.status, message=res.message, channel=res.channel,
                    ok=res.ok, time_s=res.time_s)
        # длину и ширину считаем как два отдельных измерения
        for dim, pred, true in (("length", res.length_mm, rec.length_mm),
                                ("width", res.width_mm, rec.width_mm)):
            row = dict(base, dimension=dim, pred_mm=pred, true_mm=true)
            row["abs_err_mm"] = abs(pred - true) if res.ok else np.nan
            row["rel_err_pct"] = abs(pred - true) / true * 100 if res.ok else np.nan
            rows.append(row)
    return rows


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    # честное сравнение вариантов: MAE ещё и по фото, удачным во ВСЕХ вариантах
    normal_all = details[details["expect_fail"] == 0]
    per_file_ok = normal_all.drop_duplicates(["config", "file"]).pivot(
        index="file", columns="config", values="ok")
    common_files = set(per_file_ok.index[per_file_ok.all(axis=1)])
    out = []
    for name, d in details.groupby("config", sort=False):
        normal = d[d["expect_fail"] == 0]
        photos_normal = normal.drop_duplicates("file")
        expected_fail = d[d["expect_fail"] == 1].drop_duplicates("file")
        meas = normal[normal["ok"]]
        out.append({
            "config": name,
            "photos": len(photos_normal),
            "success_rate_pct": 100 * photos_normal["ok"].mean() if len(photos_normal) else np.nan,
            "mae_mm": meas["abs_err_mm"].mean(),
            "mae_common_mm": meas.loc[meas["file"].isin(common_files), "abs_err_mm"].mean(),
            "common_photos": len(common_files),
            "mae_length_mm": meas.loc[meas.dimension == "length", "abs_err_mm"].mean(),
            "mae_width_mm": meas.loc[meas.dimension == "width", "abs_err_mm"].mean(),
            "rel_err_pct": meas["rel_err_pct"].mean(),
            "max_abs_err_mm": meas["abs_err_mm"].max(),
            "time_s_mean": photos_normal["time_s"].mean(),
            "correct_rejection_pct": (100 * (~expected_fail["ok"]).mean()
                                      if len(expected_fail) else np.nan),
        })
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Оценка точности измерений")
    p.add_argument("--gt", type=Path, default=GROUND_TRUTH_CSV)
    p.add_argument("--images", type=Path, default=IMAGES_DIR)
    p.add_argument("--output", type=Path, default=RESULTS_DIR)
    p.add_argument("--split", default="all", help="debug / final / all")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--save-steps", action="store_true", help="сохранять промежуточные изображения")
    args = p.parse_args()

    gt = read_table(args.gt)
    if "expect_fail" not in gt.columns:
        gt["expect_fail"] = 0
    gt["expect_fail"] = pd.to_numeric(gt["expect_fail"], errors="coerce").fillna(0).astype(int)
    if "marker_mm" in gt.columns:
        print(f"Размер маркера из таблицы задан для {int(gt['marker_mm'].notna().sum())} из {len(gt)} фото.")
    else:
        print("ВНИМАНИЕ: в таблице нет столбца marker_mm — для всех фото берётся marker_size_mm из config.py.")
    if args.split != "all" and "split" in gt.columns:
        gt = gt[gt["split"] == args.split]

    missing = [f for f in gt["file"] if not (args.images / f).exists()]
    if missing:
        print(f"Нет файлов (пропущены): {', '.join(missing)}")
        gt = gt[~gt["file"].isin(missing)]
    if gt.empty:
        print("Нет фотографий для оценки: проверьте dataset/images и ground_truth.csv")
        return

    steps_dir = args.output / "steps_eval" if args.save_steps else None
    rows = []
    for name in args.configs:
        print(f"Вариант: {name}")
        rows += run_config(name, gt, args.images, steps_dir)
    details = pd.DataFrame(rows)
    summary = summarize(details)

    args.output.mkdir(parents=True, exist_ok=True)
    details.round(3).to_csv(args.output / "evaluation_details.csv", index=False, encoding="utf-8-sig")
    summary.round(3).to_csv(args.output / "evaluation_summary.csv", index=False, encoding="utf-8-sig")

    valid = details[(details["expect_fail"] == 0) & details["ok"]]
    for col in GROUP_COLUMNS:
        if col in valid.columns:
            table = valid.groupby([col, "config"])["abs_err_mm"].agg(["mean", "count"]).round(3)
            table.to_csv(args.output / f"mae_by_{col}.csv", encoding="utf-8-sig")
            viz.plot_mae_by_group(valid, col, GROUP_COLUMNS[col],
                                  args.output / "plots" / f"mae_by_{col}.png")
    viz.plot_mae_by_config(summary.dropna(subset=["mae_mm"]), args.output / "plots" / "mae_by_config.png")
    for name in args.configs:
        viz.plot_true_vs_pred(valid, name, args.output / "plots" / f"true_vs_pred_{name}.png")

    print("\n" + summary.round(2).to_string(index=False))
    print(f"\nФайлы сохранены в {args.output}")


if __name__ == "__main__":
    main()
