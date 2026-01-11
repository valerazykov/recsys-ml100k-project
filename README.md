# RecSys — MovieLens 100k (MLOps проект)

**Краткое описание:** прототип рекомендательной системы на базе MovieLens-100k с полностью воспроизводимым пайплайном: загрузка/предобработка данных, обучение модели (MF / NCF), сохранение модели в формате, совместимом с `save_pretrained()`, тесты, линтинг кода и CI/CD.

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
6. **MLOps** — pytest тесты для всех блоков + линтинг кода + GitHub Actions для автоматического запуска.

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
  utils.py
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
  model/               # сюда сохраняется модель после обучения
```

---

## 7. Пример `configs/train.yaml`

```yaml
data:
  ml100k_dir: "data/ml-100k"
  ratings_file: "data/ml-100k/u.data"
seed: 42

train:
  batch_size: 512
  epochs: 20
  lr: 1e-3
  val_split: 0.1
  test_split: 0.1

model:
  type: "mf"            # "mf" | "ncf"
  embedding_dim: 64
  hidden_dims: [128, 64]  # для NCF

save:
  dir: "artifacts/model"

metrics:
  topk: 10

slo:
  p95_latency_ms: 200
  max_memory_mb: 512
  error_rate_percent: 1.0
```

---

## 8. Как запустить проект локально

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
```

4. **Обучение модели**:

```bash
python -m src.train --config configs/train.yaml
```

* **Запустить оценку (Optional)** (предполагается, что artifacts/model уже существует и в нём файлы pytorch_model.bin и config.json):
```bash
python -m src.evaluate --config configs/train.yaml --model-dir artifacts/model
```

5. **Инференс / top-N рекомендации**:

```bash
<Пока не реализовано>
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

