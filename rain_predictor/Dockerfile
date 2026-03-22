FROM python:3.12-slim

# Versions pinned for reproducibility
RUN pip install --no-cache-dir \
    onnxruntime==1.22.0 \
    numpy==2.2.3 \
    aiohttp==3.11.18

WORKDIR /app

COPY rain_model.onnx .
COPY rain_model.onnx.data .
COPY scaler_params.json .
COPY run.py .

CMD ["python", "-u", "run.py"]
