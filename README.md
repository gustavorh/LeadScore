# LeadScore — Predictor de conversión de ventas en e-commerce

Aplicación web que predice la probabilidad de que una sesión de e-commerce
termine en compra, a partir de la secuencia de eventos de navegación. Compara un
baseline supervisado, una GRU y un encoder Transformer, y los combina en un
**ensamble híbrido**. Incluye segmentación no supervisada (K-means) de sesiones.

> 🚧 **En construcción.** Este README se completa en M7 (arquitectura,
> tabla de métricas, instrucciones de despliegue y URL pública). Ver `SPEC.md`
> para la especificación completa.

## Estado por milestone

- [x] **M1** — Esqueleto + baseline UCI.
- [ ] M2 — Sesionización RetailRocket + EDA.
- [ ] M3 — Modelos secuenciales (GRU, Transformer).
- [ ] M4 — Híbrido + K-means.
- [ ] M5 — API (FastAPI).
- [ ] M6 — Frontend + docker-compose.
- [ ] M7 — Deploy (docker-compose en VPS) + README final.

## Desarrollo local

Requiere **Python 3.11** (igual que la imagen Docker de la API).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### M1 — baseline sobre UCI

```bash
python -m src.train --dataset uci   # descarga UCI, entrena baseline, guarda artefactos
python -m src.evaluate              # métricas del test set → results/metrics.json
pytest                              # tests en verde
```

## Estructura

```
src/            config, datos (download/sessionize/features), modelos, train, evaluate
tests/          tests unitarios (features, sessionize, api)
artifacts/      modelos entrenados que consume la API (versionados, <50MB)
results/        metrics.json + matrices de confusión
api/            FastAPI (M5)
frontend/       estáticos + nginx (M6)
```

## Datos

- **UCI Online Shoppers Purchasing Intention** (§4.2): descarga automática sin
  credenciales vía `ucimlrepo`. Es el dataset de M1.
- **RetailRocket** (§4.1): descarga vía Kaggle API (M2+).
