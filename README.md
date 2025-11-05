# RecSys — MovieLens 100k (MLOps проект)

**Краткое описание:** прототип рекомендательной системы на базе MovieLens-100k с полностью воспроизводимым пайплайном: загрузка/предобработка данных, обучение модели (MF / NCF), сохранение модели в формате, совместимом с `save_pretrained()`-подходом, тесты и CI.

---

## 1. Цель проекта / бизнес-задача

Повысить релевантность персональных рекомендаций фильмов, что приведёт к росту CTR рекомендованных карточек и удержанию пользователей. На уровне прототипа — показать воспроизводимый MLOps-поток от данных до deploy-готовой модели.

---

## 2. Датасет

**MovieLens 100k** (ml-100k).
Формат: `user_id`, `item_id`, `rating`, `timestamp` (+ metadata: фильмы, жанры, информация о пользователях). Небольшой и удобный для быстрого прототипирования.

Скачать:

```bash
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ..
```

Можно также просто скачать архив по [ссылке](https://files.grouplens.org/datasets/movielens/ml-100k.zip) и распаковать в папку `./data`.

---

## 3. Целевые метрики

**Бизнес:**

* CTR рекомендаций — цель для продакшена: **увеличение CTR ≥ 5% vs baseline (popularity)**.

**ML / качество (оценка на валидации и тесте):**

* RMSE (rating prediction) — целевое значение: **≤ 0.95** (ориентировочно для прототипа).
* Precision@10 ≥ 0.25, Recall@10 ≥ 0.15.

**Технические SLA (прототип):**

* p95 latency (inference, single request / top-N) ≤ **200 ms**.
* Error rate ≤ **1%**.
* Memory (при инференсе) ≤ **512 MB**; CPU ≤ **1 vCPU** (ориентировочно).

---

## 4. План экспериментов (high-level)

1. **Baseline** — popularity (most popular items), оценить Precision@K / Recall@K.
2. **Collaborative Filtering** — Matrix Factorization (SVD / PyTorch MF).
3. **Neural baseline** — Neural Collaborative Filtering (NCF) — эмбеддинги пользователей/фильмов + MLP.
4. **Evaluation** — RMSE (если предсказываем рейтинг) + top-N метрики: Precision@K, Recall@K, NDCG.
5. **Deployment prep** — обёртка модели с `save_pretrained()` / `from_pretrained()` (сохранение state_dict + config).
6. **MLOps** — pytest тесты для предобработки/датасета/инференса + GitHub Actions для автоматического запуска тестов на push.

---

## 5. Структура репозитория

```
README.md
requirements.txt
configs/
  train.yaml
src/
  preprocess.py      # загрузка, валидация данных, split
  dataset.py         # Dataset / DataLoader
  model.py           # MF / NCF с методами save_pretrained/from_pretrained
  train.py           # CLI runner — читает config, логирует, запускает обучение
  predict.py         # wrapper для inference (top-N)
  logger.py
tests/
  test_preprocess.py
  test_dataset.py
  test_inference.py
.github/
  workflows/ci.yml    # workflow для запуска тестов на push
artifacts/
  model/               # сюда сохраняется модель после обучения
```

---

## 6. Пример `configs/train.yaml`

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

## 7. Как запустить (локально)

1. Создать виртуальное окружение и установить зависимости:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Скачать датасет (см. раздел 2).

3. Запустить обучение:

```bash
python src/train.py --config configs/train.yaml
```

4. Запустить тесты:

```bash
pytest -q
```

---

## 8. Формат сохранения модели

Модель должна иметь методы `save_pretrained(save_dir)` и `from_pretrained(load_dir)`; внутри можно сохранять:

* `pytorch_model.bin` (state_dict),
* `config.json` (или копию `configs/train.yaml`).

Это позволит легко загружать модель в будущем и использовать единый интерфейс для deploy.

---

## 9. Точки контроля воспроизводимости

* `random_seed` фиксируется в конфиге и применяется к `random`, `numpy`, `torch` (если используется).
* В `requirements.txt` зафиксировать версии пакетов (чтобы повторить окружение).
* В README и CI указать способ восстановления окружения и набор команд для запуска.

---

## 10. CI / тесты

* GitHub Actions: установить окружение, `pip install -r requirements.txt`, `pytest`.
* Тесты покрывают: чтение/валидация данных, корректность split, shape/тип батчей, корректность преобразования output→top-N.

---