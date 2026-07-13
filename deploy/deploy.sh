#!/usr/bin/env bash
# Despliegue reproducible en un VPS con Docker. Ejecutar desde la raíz del repo.
set -euo pipefail

COMPOSE_FILE="docker-compose.prod.yml"

echo "==> Construyendo y levantando el stack (frontend :80 + api)…"
docker compose -f "$COMPOSE_FILE" up -d --build

echo "==> Esperando a la API…"
for i in $(seq 1 30); do
  if curl -fs http://localhost/api/health >/dev/null 2>&1; then
    echo "==> OK: $(curl -s http://localhost/api/health)"
    echo "==> App disponible en http://<IP_DEL_VPS>/"
    exit 0
  fi
  sleep 2
done

echo "!! La API no respondió a tiempo. Revisar: docker compose -f $COMPOSE_FILE logs" >&2
exit 1
