# LeadScore — Predictor de conversión de ventas en e-commerce

Aplicación web que predice, en vivo, la **probabilidad de que una sesión de
e-commerce termine en compra** a partir de su secuencia de eventos de navegación
(`view`, `addtocart`). Compara un baseline supervisado, una red recurrente (GRU)
y un encoder Transformer con self-attention, y los combina en un **ensamble
híbrido** que entrega el score final. Incluye segmentación no supervisada
(K-means) de sesiones.

> **URL pública:** _(pendiente — desplegar en un VPS siguiendo [`deploy/README.md`](deploy/README.md) y pegar aquí la URL)_
>
> 📋 **[Apunte para la presentación](docs/PRESENTACION.md)** — resumen del proyecto,
> cómo se conecta con la materia (secuencias, GRU, atención, Transformers), lectura de
> los resultados y preguntas frecuentes.

Proyecto final MSI608. Cumple los requisitos duros de la pauta: Docker con
docker-compose (2 contenedores), Git con historial por milestone, despliegue en
la nube, problemática de negocio con espacios publicitarios, y metodología
visible (datos → limpieza → EDA → comparación de modelos → híbrido).

## 1. Problemática de negocio

En e-commerce, la mayoría de las sesiones no convierten: en RetailRocket solo
~4 % termina en compra. Gastar el mismo esfuerzo comercial (cupones, contacto,
retargeting) en todos los visitantes es ineficiente. LeadScore **prioriza leads**
según su intención de compra estimada, y recomienda una acción por banda:

| Banda | p_final | Acción |
|-------|---------|--------|
| 🔥 caliente | ≥ 0.70 | Priorizar contacto comercial; no quemar descuento |
| 🌡️ tibio | 0.40–0.70 | Ofrecer descuento o recordatorio de carrito |
| ❄️ frío | < 0.40 | No invertir; retargeting de bajo costo |

La monetización de la propia app se contempla con espacios publicitarios
(`div.ad-slot`, 728×90 y 300×250) listos para Google AdSense / Bidvertiser en
ambas páginas.

## 2. Arquitectura

Dos contenedores orquestados con docker-compose. El frontend (nginx) sirve los
estáticos y hace proxy de `/api` al backend (FastAPI), que solo lee de
`artifacts/`. La misma imagen se despliega en local y en producción.

```
                 docker-compose
  ┌─────────────────────────────────────────────────────┐
  │                                                       │
  │   navegador ──▶  frontend (nginx:80)                  │
  │                    ├── sirve  /  → static/            │
  │                    └── proxy  /api/ ─▶ api (uvicorn:8000)
  │                                          │  FastAPI    │
  │                                          ▼             │
  │                                   artifacts/*.pt,.joblib,
  │                                   ensemble.json, vocab.json
  └─────────────────────────────────────────────────────┘

  Pipeline (offline):
    events.csv ─▶ sessionize ─▶ sessions.parquet ─▶ features
                                                       ├─▶ baseline (LogReg/RF)
                                                       ├─▶ GRU
                                                       ├─▶ Transformer
                                                       ├─▶ K-means (segmentos)
                                                       └─▶ stacking ─▶ ensemble
```

## 3. Metodología

1. **Datos.** UCI Online Shoppers (tabular, para el pipeline de M1) y RetailRocket
   (secuencias de eventos, dataset principal).
2. **Limpieza / sesionización.** Eventos → sesiones cortando por gaps > 30 min.
   **Anti-leakage:** la etiqueta `converted` se calcula sobre la sesión completa,
   pero de la secuencia de entrada se elimina el primer `transaction` y todo lo
   posterior (verificado por test). Filtro < 3 eventos, truncado a 50.
3. **EDA** (`notebooks/01_eda.ipynb`): embudo de conversión, distribución de
   largos, desbalance.
4. **Modelos** (`notebooks/02_models.ipynb`): baseline, GRU, Transformer, todos
   sobre el **mismo split** estratificado 70/15/15 **por visitante** (sin fuga
   entre train y test).
5. **Híbrido.** Stacking logístico de las 3 probabilidades (aprendido en
   validación), umbral calibrado por F1, y segmentación K-means.

## 4. Resultados (test set, n = 25.032)

| Modelo | Accuracy | Precision | Recall | F1 | AUC |
|--------|:--------:|:---------:|:------:|:------:|:------:|
| Baseline (LogReg) | 0.871 | 0.236 | 0.913 | 0.375 | 0.916 |
| GRU | 0.885 | 0.263 | 0.944 | 0.411 | 0.945 |
| Transformer | 0.883 | 0.260 | 0.945 | 0.407 | 0.943 |
| **Híbrido** | **0.942** | 0.397 | 0.680 | **0.501** | **0.945** |

Dataset desbalanceado (~4 % de conversión), por eso se prioriza F1/AUC sobre
accuracy. Los modelos secuenciales superan al baseline en AUC (la secuencia
aporta señal), y el **híbrido domina en F1 y accuracy** gracias al stacking y al
umbral calibrado (0.891). Matrices de confusión y heatmap de atención en
`results/`.

**Segmentos (K-means, k=4):** `decidido`, `comparador`, `explorador`,
`carrito abandonado` — nombrados por heurística sobre los centroides.

## 5. Cómo correr

### Local con docker-compose (recomendado)

Los artefactos entrenados están versionados, así que no hay que entrenar:

```bash
docker compose up --build       # http://localhost:8080
```

- Simulador: `http://localhost:8080/`
- Dashboard de leads: `http://localhost:8080/dashboard.html` — se puede probar subiendo
  [`data/ejemplo_leads.csv`](data/ejemplo_leads.csv) (30 leads reales del test set).

### Entorno de desarrollo (Python 3.11)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                                        # suite de tests

# Reproducir el pipeline (opcional; requiere Kaggle API para RetailRocket)
python -m src.train --dataset retailrocket    # entrena y guarda artefactos
python -m src.evaluate                         # métricas → results/metrics.json
```

### API (contrato)

- `GET /api/health` → `{status, models_loaded, version}`
- `POST /api/score` → score de una sesión (secuencia de eventos)
- `POST /api/score/batch` → CSV de leads (baseline tabular + segmento), ordenado
  por probabilidad

## 6. Despliegue en la nube

Producción corre **el mismo docker-compose de 2 contenedores** en un VPS
(frontend en el puerto 80). Ver [`deploy/README.md`](deploy/README.md):

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

## 7. Estructura

```
src/            config, datos (download/sessionize/features/sequence),
                modelos (baseline/gru/transformer/clustering/ensemble),
                train, evaluate, inference
api/            FastAPI (Dockerfile, requirements, app/)
frontend/       estáticos + nginx (Dockerfile)
artifacts/      modelos entrenados que consume la API (versionados, <1MB)
results/        metrics.json, matrices de confusión, heatmap de atención
notebooks/      01_eda.ipynb, 02_models.ipynb (ejecutados)
tests/          features, sessionize (anti-leakage), modelos, clustering,
                ensemble, api
deploy/         compose de producción + guía de VPS
```

## 8. Decisiones de diseño

- **Sin fuga de datos:** test set nunca se usa para entrenar ni calibrar; el
  umbral y el stacking se ajustan en validación; ningún feature contiene
  información posterior a la primera transacción (test automatizado).
- **Categorías diferidas a v1:** los modelos secuenciales usan tipo de evento +
  tiempos; la API acepta `item_category` mapeándolo a OOV.
- **Reproducibilidad:** seed 42 en numpy/torch/sklearn.
- **Entrenamiento en CPU:** modelos pequeños (< 30 min); la imagen de la API usa
  torch CPU, coincidiendo con el entorno de entrenamiento.

---

**URL pública del despliegue:** _(pegar aquí tras desplegar en el VPS)_
