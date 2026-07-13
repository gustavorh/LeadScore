# Imagen de la API: python:3.11-slim (igual que el venv de desarrollo).
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

# Dependencias primero (capa cacheable). torch CPU vía índice en requirements.
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Código + artefactos entrenados (imagen autocontenida; sin paso de entrenamiento).
COPY src/ ./src/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

EXPOSE 8000
CMD ["uvicorn", "api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
