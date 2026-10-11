"""Рисование промежуточных результатов и графики для отчёта."""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 220), (0, 220, 255)


def _thickness(image: np.ndarray) -> int:
    return max(2, int(round(max(image.shape[:2]) / 500)))


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def draw_marker(image: np.ndarray, corners: np.ndarray, marker_id: int) -> np.ndarray:
    out, t = image.copy(), _thickness(image)
    cv2.polylines(out, [corners.astype(np.int32)], True, GREEN, t)
    for i, p in enumerate(corners):
        cv2.circle(out, tuple(np.int32(p)), t * 3, RED, -1)
        cv2.putText(out, str(i), tuple(np.int32(p) + t * 4), cv2.FONT_HERSHEY_SIMPLEX,
                    t * 0.5, YELLOW, t)
    cv2.putText(out, f"ArUco id={marker_id}", tuple(np.int32(corners[0]) - [0, t * 8]),
                cv2.FONT_HERSHEY_SIMPLEX, t * 0.5, GREEN, t)
    return out


def draw_contour_box(image: np.ndarray, contour: np.ndarray, box: np.ndarray) -> np.ndarray:
    out, t = image.copy(), _thickness(image)
    cv2.drawContours(out, [contour], -1, GREEN, t)
    cv2.polylines(out, [np.int32(box)], True, RED, t)
    return out


def draw_result(image: np.ndarray, box: np.ndarray, length_mm: float, width_mm: float) -> np.ndarray:
    """Рамка и подпись с размерами на исходном фото."""
    out, t = image.copy(), _thickness(image)
    cv2.polylines(out, [np.int32(box)], True, GREEN, t)
    text = f"L = {length_mm:.1f} mm   W = {width_mm:.1f} mm"
    org = (int(box[:, 0].min()), max(int(box[:, 1].min()) - t * 6, t * 12))
    scale = t * 0.5
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, t)
    cv2.rectangle(out, (org[0] - 4, org[1] - th - 6), (org[0] + tw + 4, org[1] + 6), (0, 0, 0), -1)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t)
    return out


def draw_failure(image: np.ndarray, message: str) -> np.ndarray:
    out, t = image.copy(), _thickness(image)
    cv2.putText(out, "FAILED", (t * 10, t * 25), cv2.FONT_HERSHEY_SIMPLEX, t * 0.9, RED, t * 2)
    cv2.putText(out, message.encode("ascii", "replace").decode(), (t * 10, t * 45),
                cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, RED, t)
    return out


# ----------------------------- графики ошибок ------------------------------

def _save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mae_by_config(summary: pd.DataFrame, path: Path) -> None:
    """Сравнение вариантов алгоритма по MAE (мм)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary["config"], summary["mae_mm"], color="#2f6f9f")
    ax.set_ylabel("MAE, мм")
    ax.set_title("Средняя абсолютная ошибка по вариантам алгоритма")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, path)


def plot_mae_by_group(details: pd.DataFrame, group_col: str, title: str, path: Path) -> None:
    """MAE в зависимости от освещения / фона / наклона для каждого варианта."""
    ok = details[details["ok"]]
    if ok.empty or group_col not in ok.columns:
        return
    table = ok.groupby([group_col, "config"])["abs_err_mm"].mean().unstack("config")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    table.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("MAE, мм")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="вариант")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, path)


def plot_true_vs_pred(details: pd.DataFrame, config: str, path: Path) -> None:
    """Диаграмма «истинный размер — измеренный размер» (идеал — диагональ)."""
    d = details[(details["config"] == config) & details["ok"]]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(d["true_mm"], d["pred_mm"], s=18, alpha=0.7, color="#2f6f9f")
    lim = [0, max(d["true_mm"].max(), d["pred_mm"].max()) * 1.05]
    ax.plot(lim, lim, "--", color="gray", label="идеал")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("истинный размер, мм")
    ax.set_ylabel("измеренный размер, мм")
    ax.set_title(f"Истинные и измеренные размеры ({config})")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_fig(fig, path)
