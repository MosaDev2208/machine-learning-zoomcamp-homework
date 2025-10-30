FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.120.2 \
    uvicorn==0.38.0 \
    pydantic==2.12.3

COPY main.py .
COPY pipeline_v1.bin .

EXPOSE 8000

CMD ["python", "main.py"]
