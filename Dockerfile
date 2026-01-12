# Use a slim base
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Copy only requirements for docker (runtime)
COPY requirements_docker.txt /app/requirements_docker.txt

# Install CPU-only PyTorch first (explicit pinned version).
# Use the official PyTorch cpu wheel index:
RUN pip install --no-cache-dir "torch" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /app/requirements_docker.txt \
 && rm -rf /root/.cache/pip

# Copy minimal project files needed for prediction:
COPY src/predict.py /app/src/predict.py
COPY src/model.py /app/src/model.py

# Copy model artifacts (if you want model baked into image). Otherwise use dvc pull at runtime.
COPY artifacts/model/pytorch_model.bin /app/artifacts/model/pytorch_model.bin
COPY artifacts/model/config.json /app/artifacts/model/config.json

# App entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]
