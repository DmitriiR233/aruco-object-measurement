# Измерение реальных размеров плоского объекта по эталону (Python + OpenCV)

Проект по дисциплине «Компьютерное зрение 1». Программа находит ArUco-маркер известного
размера, выравнивает плоскость по перспективе, выделяет объект и считает его длину и
ширину в миллиметрах.

## Установка
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Нужен именно `opencv-contrib-python` (модуль ArUco). Не ставьте рядом `opencv-python`.

## Подготовка
1. Тип маркера (словарь и id) программа определяет сама. Свой маркер можно создать так:
   `python tools/make_marker.py --id 70 --out marker.png` (словарь и id при желании задаются `--dict`, `--marker-id`).
   (масштаб 100%, **проверьте линейкой, что чёрный квадрат = 50 мм**; белое поле вокруг не обрезайте).
   Реальную сторону чёрного квадрата обязательно измерьте линейкой и задайте в `config.py` (`marker_size_mm`) или через `--marker-mm`.
   Если на разных фото маркеры разного размера, впишите размер каждого в столбец `marker_mm` файла `dataset/ground_truth.csv` (его читают и `main.py`, и `evaluate.py`).
   Если размер другой — укажите его в `config.py` (`marker_size_mm`) или `--marker-mm`.
2. Кладите маркер и объект на **одну плоскость** (стол, лист бумаги). Объект целиком в кадре.
3. Положите фото в `dataset/images/`, измеренные линейкой/штангенциркулем размеры занесите
   в `dataset/ground_truth.csv` (в файле лежат демонстрационные строки — замените своими).

## Запуск
```bash
python main.py --input dataset/images                 # вся папка, результаты в results/
python main.py --input dataset/images/photo_01.jpg    # одно фото
python main.py --input dataset/images --method adaptive
python main.py --input dataset/images --no-perspective
python evaluate.py --split final --save-steps         # MAE, ошибки, графики
```
Результаты: `results/measurements.csv`, `results/steps/` (маркер, выровненный кадр, маска,
контур, итог), `results/evaluation_*.csv`, `results/plots/*.png`.

## ground_truth.csv
`file, object, length_mm, width_mm` — обязательные; `marker_mm` — реальная сторона маркера на этом фото, мм (необязательно); `lighting, background, tilt` — условия
(по ним строятся графики); `split` — `debug` (отладка) или `final` (финальная проверка,
параметры по ней не подстраивать); `expect_fail=1` — программа должна отказаться от
измерения (маркер закрыт и т. п.).

## Структура
| Файл | Назначение |
|---|---|
| `main.py` | запуск, пакетная обработка |
| `pipeline.py` | полный конвейер для одного фото |
| `marker_detector.py` | поиск ArUco |
| `image_preprocessing.py` | фильтрация, цветовое пространство |
| `object_segmentation.py` | порог, морфология, выбор контура |
| `measurement.py` | гомография, масштаб, размеры |
| `visualization.py` | промежуточные картинки и графики |
| `evaluate.py` | метрики и сравнение вариантов |
| `config.py` | все параметры |
| `tools/make_marker.py` | печатный маркер |
| `tools/make_synthetic.py` | синтетические кадры для проверки кода (не для отчёта вместо реальных фото) |

## Каналы сегментации
`--channel auto` (по умолчанию) перебирает яркость (`gray`), цветовое расстояние до фона (`colordist`) и насыщенность (`saturation`): помогает, когда объект и фон одной яркости, но разного цвета (например, картон на зелёном). Контраст объекта с фоном должен быть не меньше ΔE = 5 (`min_contrast_de` в `config.py`), иначе программа пишет «слабый контраст» и не измеряет.

## Ограничения
Плоские объекты на одной плоскости с маркером; объёмные предметы и толщина не измеряются.
