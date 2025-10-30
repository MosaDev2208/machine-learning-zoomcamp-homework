FROM agrigorev/zoomcamp-model:2025

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.120.2 \
    uvicorn==0.38.0 \
    pydantic==2.12.3

COPY main.py .
COPY pipeline_v1.bin .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/', timeout=5)"

CMD ["python", "main.py"]
