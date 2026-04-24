FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir numpy fastapi uvicorn pydantic || pip install --no-cache-dir -r requirements.txt

COPY tetrahedron_memory/ ./tetrahedron_memory/
COPY start_api_v2.py .
COPY ui/ ./static/
COPY ui/dashboard.html ./static/dashboard.html

RUN sed -i "s/CHANGE_ME/\${TETRAMEM_PASSWORD:-CHANGE_ME}/g" ./static/index.html

RUN mkdir -p /data/tetramem_data_v2 && \
    echo '{"metadata":{"version":"6.5.0"},"tetrahedra":{}}' > /data/tetramem_data_v2/mesh_index.json

ENV TETRAMEM_STORAGE=/data/tetramem_data_v2
ENV TETRAMEM_UI_PASSWORD=CHANGE_ME

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

CMD ["python", "-m", "uvicorn", "start_api_v2:app", "--host", "0.0.0.0", "--port", "8000"]
