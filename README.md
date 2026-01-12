# RecSys — MovieLens 100k (MLOps проект)

**Краткое описание:** прототип рекомендательной системы на базе MovieLens-100k с полностью воспроизводимым MLOps-пайплайном: загрузка/предобработка данных, обучение модели (MF / NCF), сохранение модели в формате, совместимом с `save_pretrained()`, тесты, линтинг и DVC-версионирование данных/моделей.

---

## 1. Цель проекта / бизнес-задача

Повысить релевантность персональных рекомендаций фильмов, что приведёт к росту CTR рекомендованных карточек и удержанию пользователей. На уровне прототипа — показать полностью воспроизводимый MLOps-поток от данных до deploy-готовой модели.

---

## 2. Датасет

**MovieLens 100k** (ml-100k).  
Формат: `user_id`, `item_id`, `rating`, `timestamp` (+ metadata: фильмы, жанры, информация о пользователях).

Скачать:

```bash
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ..
```

Или скачать архив по [ссылке](https://files.grouplens.org/datasets/movielens/ml-100k.zip) и распаковать в `./data`.

> В этом репозитории `data/ml-100k/` — сырой датасет, а `data/processed/` — выход `preprocess` (тот, что под версионированием DVC).

---

## 3. Что реализовано

* **Препроцессинг данных**: проверка структуры, типов, наличия необходимых признаков, разбиение на train/val/test.
* **Dataset / DataLoader**: `RatingDataset`, корректное формирование батчей, поддержка n_users / n_items.
* **Модели**: MF и NCF с методами `save_pretrained()` и `from_pretrained()`.
* **Обучение**: CLI runner `train.py`, чтение конфигурации, логирование через `logging`.
* **Инференс / top-N**: корректная обработка выходов модели → top-K рекомендации.
* **Тесты**: покрытие всех логических блоков пайплайна (`dataset.py`, `model.py`, `engine.py`, `train.py`, `utils.py`).
* **CI/CD**: GitHub Actions запускает тесты и линтер на каждый push.
* **Linting**: flake8 + форматирование кода.
* **Воспроизводимость**: фиксированный `random_seed`, зафиксированные версии пакетов в `requirements.txt`.
* **DVC**: пайплайн stages — `prepare`, `train`, `evaluate`, локальный remote для данных/моделей.

---

## 4. Целевые метрики

**Бизнес:**

* CTR рекомендаций — цель для продакшена: **увеличение CTR ≥ 5% vs baseline (popularity)**.

**ML / качество (оценка на валидации и тесте):**

* RMSE (rating prediction) — целевое значение: **≤ 0.95**.
* Precision@10 ≥ 0.25, Recall@10 ≥ 0.15.

**Технические SLA (прототип):**

* p95 latency (inference, single request / top-N) ≤ **200 ms**.
* Error rate ≤ **1%**.
* Memory (при инференсе) ≤ **512 MB**; CPU ≤ **1 vCPU**.

---

## 5. План экспериментов (high-level)

1. **Baseline** — popularity (most popular items), оценка Precision@K / Recall@K.
2. **Collaborative Filtering** — Matrix Factorization (SVD / PyTorch MF).
3. **Neural baseline** — Neural Collaborative Filtering (NCF) — эмбеддинги пользователей/фильмов + MLP.
4. **Evaluation** — RMSE, top-N метрики: Precision@K, Recall@K, NDCG.
5. **Deployment prep** — обёртка модели с `save_pretrained()` / `from_pretrained()`.
6. **MLOps** — pytest тесты для всех блоков + линтинг кода + GitHub Actions для автоматического запуска + DVC для data/artifacts + MLflow + Docker/TorchServe

---

## 6. Структура репозитория

```
README.md
requirements.txt
configs/
  train.yaml
src/
  preprocess.py      # загрузка, валидация данных, split
  dataset.py         # Dataset / DataLoader
  model.py           # MF / NCF с save_pretrained/from_pretrained
  train.py           # CLI runner — читает config, логирует запускает обучение
  engine.py          # функции обучения и оценки модели
  evaluate.py        # подсчет метрик на тесте (при отстутсвии - на валидации)
  predict.py         # инференс
  utils.py
  mlflow_utils.py
  metrics.py
tests/
  test_preprocess.py
  test_dataset.py
  test_model.py
  test_engine.py
  test_metrics.py
  test_train_helpers.py
.github/
  workflows/ci.yml    # GitHub Actions workflow
artifacts/
  model/            # сюда сохраняется модель (и этот каталог в DVC)
  metrics/          # evaluate сохраняет сюда metrics_eval.json (в DVC)
data/
  ml-100k/          # сырой датасет (под версией DVC)
  processed/        # output prepare (под версией DVC)
```

---

## 7. Конфиги

Основной конфиг — `configs/train.yaml`. Там задаются: пути к данным, seed, гиперпараметры обучения, модель, optimizer/scheduler, пути сохранения.

Если вы меняете расположение данных, поправьте `data.ml100k_dir` или `data.processed_dir` в `configs/train.yaml`.

---

## 8. Как запустить проект локально

> Перед запуском: работаем из корня репозитория.

1. **Создать виртуальное окружение и установить зависимости:**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. **Скачать датасет (см. раздел 2).**

3. **Препроцессинг данных**:

```bash
python src/preprocess.py --config configs/train.yaml
# или (аналогично) dvc repro (если используете DVC)
```

4. **Обучение модели**:

```bash
python -m src.train --config configs/train.yaml
```

5. **Оценка (если модель уже есть в artifacts/model)**:

```bash
python -m src.evaluate --config configs/train.yaml --model-dir artifacts/model
```

6. **Запуск тестов**:

```bash
python -m pytest -q
```

7. **Линтинг / автоформатирование**:

```bash
flake8 src/ tests/ --max-line-length=150 --ignore=W605,W503,E203  # проверка
black src/ tests/  # автоформатирование
```

---

## 9. Формат сохранения модели

Модель имеет методы `save_pretrained(save_dir)` и `from_pretrained(load_dir)`. Сохраняется:

* `pytorch_model.bin` (state_dict),
* `config.json` (копия конфигурации `train.yaml`).

Позволяет легко загружать модель для инференса или дальнейшего обучения.

---

## 10. CI / GitHub Actions

* Автоматический запуск тестов и линтера на каждый push.
* Проверяет: чтение/валидацию данных, корректность split, shape/тип батчей, корректность преобразования output → top-N.
* Все результаты логируются в Actions.
* Автоматически завершает коммит, если тесты или линтер не пройдены.

---

## 11. Воспроизводимость

* Зафиксирован `random_seed` для `random`, `numpy` и `torch`.
* Версии пакетов зафиксированы в `requirements.txt`.
* Инструкции по воспроизведению пайплайна описаны в разделе 8.

---

## 12. DVC — версия данных и моделей (локальный remote)

В этом проекте DVC настроен и используется для хранения больших бинарных артефактов (сырой датасет, обработанные данные, обученная модель, метрики оценки). Пайплайн описан в `dvc.yaml` со стадиями:

```yaml
stages:
  prepare:
    cmd: python -m src.preprocess --config configs/train.yaml
    outs:
      - data/processed

  train:
    cmd: python -m src.train --config configs/train.yaml
    outs:
      - artifacts/model

  evaluate:
    cmd: python -m src.evaluate --config configs/train.yaml --model-dir artifacts/model
    metrics:
      - artifacts/metrics/metrics_eval.json
```

**Локальный remote (пример)**
На моей машине я использую локальную папку как remote DVC:

```
/Users/v.p.zykov/mmp/MLOps/dvc-storage
```

Команды для настройки (пример, если надо повторить):

```bash
# добавить локальный remote (пример: абсолютный путь)
dvc remote add -d dvc-local "/Users/v.p.zykov/mmp/MLOps/dvc-storage"

# пушим текущий кэш в remote
dvc push

# проверить состояние относительно remote
dvc status -c
```

**Как восстановить проект на другой машине**
Если вы клонируете репозиторий на другую машину:

```bash
git clone <repo_url>
cd <repo>
# если remote в .dvc/config указывает на локальный путь, и у вас нет доступа к нему — назначьте свой remote:
dvc remote add -d dvc-local <your-remote-path-or-s3-url>
# затем:
dvc pull      # подтянет data/processed и artifacts/model из remote
dvc repro     # (опционально) пересоберёт pipeline, если нужно
```

> ВАЖНО: если remote — абсолютный локальный путь (как у меня), другой пользователь не сможет dvc pull пока не перенастроит remote на доступный путь.

**Если outputs уже получены локально, но DVC их не «видит» (warnings):**

*Рекомендация:* запускать `dvc repro` чтобы DVC воспроизвёл стадии и добавил outputs в кэш. Если `repro` невозможен (долгое обучение), можно использовать `dvc commit <stage>` для ассоциации текущих файлов с stage в кэше (быстрый, но менее предпочтительный путь).

---

## 13. MLflow — трекинг экспериментов

В проекте добавлен MLflow для трекинга экспериментов. Каждый запуск `python -m src.train --config configs/train.yaml` теперь создаёт отдельный MLflow run с логированием:

* параметров обучения (взятых из конфига и CLI-override),
* метрик по эпохам (train_loss, rmse, precision_at_k и т. п. — имена метрик автоматически «санитизируются» для совместимости с MLflow),
* артефактов: сохранённой модели (`artifacts/model/pytorch_model.bin`, `config.json`), `dvc.lock` (если есть), `artifacts/metrics/metrics_eval.json` и др.

### Что добавлено в коде

* В `src/train.py` обёртка запуска использует MLflow (создаётся эксперимент/run).
* Утилиты в `src/mlflow_utils.py`:

  * `mlflow_log_epoch_metrics(epoch, metrics)` — санитизирует имена метрик и логирует их шагом `epoch`.
  * Логирование артефактов и тегов (например `dvc_lock_md5` или `dvc_data_hash`) — см. примеры ниже.


### Быстрый запуск локального MLflow UI

1. Запустите тренировку (она автоматически создаст run и запишет артефакты в локальный `mlruns/` по умолчанию):

```bash
python -m src.train --config configs/train.yaml
```

2. Запустите MLflow UI (порт 5000):

```bash
mlflow ui -p 5000
```

Откройте: `http://127.0.0.1:5000` — там вы увидите все run-ы, параметры, графики метрик, артефакты.

---

