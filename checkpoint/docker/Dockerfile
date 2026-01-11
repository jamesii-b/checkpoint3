FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

RUN cd src/native && make clean && make

ENV PYTHONPATH=/app
ENV LD_LIBRARY_PATH=/app/native:$LD_LIBRARY_PATH

CMD ["python3", "-c", "print('Container ready')"]
