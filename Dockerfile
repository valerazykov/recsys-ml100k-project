# Dockerfile
FROM python:3.10-slim

# set noninteractive
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# copy only requirements first to leverage layer cache
COPY requirements.txt /app/requirements.txt

# install system deps for torch if needed (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# install python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project code
COPY . /app

# ensure model artifacts are present (we expect artifacts/model in repo before build)
# if you prefer to pull via DVC inside image, you can add DVC and run dvc pull
# make entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]
CMD ["--help"]
