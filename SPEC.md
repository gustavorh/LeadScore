# SPEC — LeadScore: Predictor de conversión de ventas en e-commerce

> Proyecto final MSI608 (coeficiente 2). Este documento es la especificación completa
> para implementar el proyecto de principio a fin. Trabajar por milestones en orden;
> cada milestone debe dejar el repositorio en un estado funcional y commiteado.

## 1. Resumen

Aplicación web que predice la probabilidad de que una sesión de e-commerce termine
en una compra, a partir de la secuencia de eventos de navegación (view, addtocart,
transaction). Compara un baseline supervisado clásico, una red recurrente (GRU) y un
encoder Transformer con self-attention, y los combina en un **ensamble híbrido** que
entrega el score final. Incluye segmentación no supervisada (K-means) de sesiones.

La app se compone de **2 contenedores** (API FastAPI + frontend Nginx) orquestados
con **docker-compose**, versionada en **Git**, y desplegada en la **nube**
(Hugging Face Spaces con Docker SDK; alternativa: Render).

**Requisitos duros de la pauta del curso (no negociables):**
1. Docker con docker-compose (más de 1 contenedor).
2. Repositorio Git con historial de commits real (commit por milestone como mínimo).
3. Despliegue en la nube con URL pública.
4. Problemática de negocio: conversión de ventas / priorización de leads; el frontend
   debe reservar espacios publicitarios (placeholders para Google AdSense / Bidvertiser).
5. Metodología visible: datos → limpieza → EDA → comparación de modelos → híbrido.

## 2. Stack tecnológico

- **Python 3.11** — pipeline de datos, entrenamiento e inferencia.
- **PyTorch** — GRU y encoder Transformer (modelos pequeños, entrenables en CPU/MPS).
- **scikit-learn** — baseline (LogisticRegression, RandomForest), K-means, métricas.
- **pandas / pyarrow** — procesamiento de datos (usar parquet como formato intermedio).
- **FastAPI + uvicorn** — API de inferencia.
- **Frontend**: HTML + CSS + JS vanilla (sin build step) servido por Nginx.
  Mantenerlo simple; nada de React ni bundlers.
- **Docker + docker-compose**.
- Notebooks Jupyter para EDA y experimentos (guardar con outputs ejecutados).
- Entorno local: MacBook (usar `mps` si está disponible, si no CPU). Los modelos
  deben ser lo bastante pequeños para entrenar en <30 min en CPU.

## 3. Estructura del repositorio

```
leadscore/
├── README.md                  # descripción, arquitectura, cómo correr, URL deploy
├── SPEC.md                    # este archivo
├── docker-compose.yml
├── .gitignore                 # ignorar data/raw, *.pt grandes solo si >100MB, venv
├── data/
│   ├── raw/                   # datasets descargados (gitignored)
│   └── processed/             # parquet de sesiones (gitignored, regenerable)
├── notebooks/
│   ├── 01_eda.ipynb           # EDA de RetailRocket y UCI
│   └── 02_models.ipynb        # comparación de modelos + tabla final de métricas
├── src/
│   ├── config.py              # rutas, hiperparámetros, seed global (42)
│   ├── data/
│   │   ├── download.py        # descarga de datasets
│   │   ├── sessionize.py      # eventos → sesiones etiquetadas
│   │   └── features.py        # features tabulares por sesión
│   ├── models/
│   │   ├── baseline.py        # LogReg + RandomForest
│   │   ├── gru.py             # GRUClassifier (PyTorch)
│   │   ├── transformer.py     # TransformerClassifier (PyTorch)
│   │   ├── clustering.py      # K-means + etiquetas de segmento
│   │   └── ensemble.py        # HybridEnsemble
│   ├── train.py               # CLI: entrena todo y guarda artefactos
│   └── evaluate.py            # CLI: métricas comparativas → results/metrics.json
├── artifacts/                 # modelos entrenados que usa la API (committeados si <50MB)
├── results/                   # metrics.json, matrices de confusión (png)
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py            # FastAPI
│       └── schemas.py         # Pydantic models
├── frontend/
│   ├── Dockerfile             # nginx:alpine + copia de static/
│   ├── nginx.conf             # sirve static y hace proxy /api → contenedor api
│   └── static/
│       ├── index.html         # landing + simulador
│       ├── dashboard.html     # tabla de leads con scoring por CSV
│       ├── app.js
│       └── styles.css
└── tests/
    ├── test_sessionize.py
    ├── test_features.py
    └── test_api.py
```

## 4. Datos

### 4.1 Dataset principal: RetailRocket (Kaggle)
- `events.csv`: columnas `timestamp, visitorid, event, itemid, transactionid`.
  Eventos: `view`, `addtocart`, `transaction`.
- Descarga: requiere Kaggle API (`kaggle datasets download -d retailrocket/ecommerce-dataset`).
  `download.py` debe: (a) intentar con kaggle CLI si hay credenciales, (b) si no,
  imprimir instrucciones claras para descarga manual a `data/raw/retailrocket/`.

### 4.2 Dataset complementario / plan B: UCI Online Shoppers Purchasing Intention
- 12.330 filas, 17 features tabulares, target `Revenue` (bool).
- Descarga directa por URL desde el repositorio UCI (sin credenciales) o vía
  `ucimlrepo`. Usarlo en Milestone 1 para tener el pipeline completo funcionando
  antes de pelear con la sesionización de RetailRocket.

### 4.3 Sesionización (RetailRocket) — `sessionize.py`
- Ordenar eventos por `visitorid, timestamp`.
- Cortar sesión cuando pasan **>30 minutos** entre eventos consecutivos del mismo visitante.
- Etiqueta `converted = 1` si la sesión contiene al menos un evento `transaction`.
- **Importante (anti-leakage):** eliminar de la secuencia de entrada los eventos
  `transaction` y todo lo posterior al primer `transaction`. El modelo predice
  conversión a partir del comportamiento previo, no de la compra misma.
- Filtrar sesiones con menos de 3 eventos (sin señal) y truncar a máx. 50 eventos.
- Guardar en `data/processed/sessions.parquet` con:
  `session_id, visitor_id, event_types (list[int]), item_cats (list[int]),
  deltas_t (list[float], segundos log-normalizados), length, converted`.
- Codificación de eventos: vocabulario pequeño {PAD=0, view=1, addtocart=2}.
  Categoría de ítem: mapear con `category_tree.csv` a las top-N categorías (N=200)
  + token OOV; si el join resulta muy pesado, es aceptable omitir categorías en v1.

### 4.4 Features tabulares — `features.py`
Por sesión: `n_events, n_views, n_addtocart, n_unique_items, duration_sec,
mean_delta_t, hour_of_day, day_of_week, addtocart_rate`. Estas alimentan el
baseline y K-means.

### 4.5 Split
- Split estratificado 70/15/15 (train/val/test) **por visitante** (un visitante no
  puede estar en train y test a la vez). Seed 42 en todo.

## 5. Modelos

Todos consumen el mismo split y reportan sobre el mismo test set.

### 5.1 Baseline — `baseline.py`
- `LogisticRegression(class_weight='balanced', max_iter=1000)` y
  `RandomForestClassifier(n_estimators=300, class_weight='balanced')` sobre features
  tabulares estandarizadas. Reportar ambos, elegir el mejor por F1 como "baseline".
- Guardar importancia de variables (para la sección de interpretación).

### 5.2 GRU — `gru.py`
- `Embedding(event_vocab, 32)` (+ embedding de categoría 32 si está disponible,
  concatenado con delta_t proyectado) → `GRU(hidden=64, layers=1, batch_first=True)`
  → último estado → `Linear(64→1)`.
- Padding con longitud empaquetada (`pack_padded_sequence`) o máscara simple.
- Loss: `BCEWithLogitsLoss(pos_weight=ratio_neg/pos)`. Optimizer AdamW lr=1e-3.
- Early stopping por AUC de validación, máx. 10 épocas. Si el dataset completo es
  lento, submuestrear negativos de train a ratio 10:1 (no tocar val/test).

### 5.3 Transformer — `transformer.py`
- Mismos embeddings de entrada + **positional encoding** sinusoidal.
- `nn.TransformerEncoder` con 2 capas, `d_model=64, nhead=4, dim_feedforward=128,
  dropout=0.1`, con máscara de padding (`src_key_padding_mask`).
- Pooling: mean pooling enmascarado (o token CLS prepend, a elección) → `Linear(64→1)`.
- Mismo esquema de entrenamiento que la GRU.
- Exponer los pesos de atención de la última capa para 2-3 ejemplos del test
  (guardar un heatmap png en `results/`) — se usa en la sección de interpretación.

### 5.4 K-means — `clustering.py`
- K-means (k=4) sobre features tabulares estandarizadas. Nombrar clusters según sus
  centroides (ej.: "explorador", "comparador", "carrito abandonado", "decidido").
  El nombre se asigna con una heurística documentada, no a mano por índice.

### 5.5 Ensamble híbrido — `ensemble.py`
- `p_final = w1*p_baseline + w2*p_gru + w3*p_transformer`, pesos aprendidos con una
  regresión logística sobre las probabilidades en **validación** (stacking simple),
  con fallback a promedio simple si algo falla.
- Umbral de decisión calibrado en validación maximizando F1; guardar el umbral.
- Acción recomendada por reglas sobre (p_final, segmento):
  - p ≥ 0.7 → "caliente": priorizar contacto comercial / no gastar cupón.
  - 0.4–0.7 → "tibio": ofrecer descuento o recordatorio de carrito.
  - < 0.4 → "frío": no invertir; retargeting de bajo costo.

### 5.6 Artefactos
`train.py` guarda en `artifacts/`: `baseline.joblib, scaler.joblib, kmeans.joblib,
gru.pt, transformer.pt, ensemble.json (pesos+umbral), vocab.json, metadata.json
(fechas, tamaños, métricas)`. La API solo lee de `artifacts/`.

`evaluate.py` genera `results/metrics.json` con accuracy, precision, recall, F1 y AUC
de cada modelo + híbrido, y matrices de confusión en PNG.

## 6. API — contrato

`api/app/main.py`, FastAPI, prefijo `/api`.

### GET /api/health
```json
{ "status": "ok", "models_loaded": true, "version": "1.0.0" }
```

### POST /api/score — una sesión
Request:
```json
{
  "events": [
    { "type": "view", "item_category": "electronics", "seconds_since_prev": 0 },
    { "type": "view", "item_category": "electronics", "seconds_since_prev": 45 },
    { "type": "addtocart", "item_category": "electronics", "seconds_since_prev": 30 }
  ],
  "hour_of_day": 21,
  "day_of_week": 4
}
```
Response:
```json
{
  "conversion_probability": 0.73,
  "label": "caliente",
  "recommended_action": "Priorizar contacto comercial; alta intención, no quemar descuento.",
  "segment": "decidido",
  "model_breakdown": { "baseline": 0.61, "gru": 0.77, "transformer": 0.79, "ensemble": 0.73 },
  "threshold": 0.42
}
```
Categorías desconocidas → token OOV; validación con Pydantic; errores 422 con mensaje claro.

### POST /api/score/batch — CSV de leads
- `multipart/form-data` con un CSV: una fila por sesión, columnas
  `lead_id, n_views, n_addtocart, n_unique_items, duration_sec, hour_of_day, day_of_week`.
  (Batch usa solo el baseline tabular + segmento; documentarlo en el README.)
- Response: lista de `{lead_id, probability, label, segment, recommended_action}`
  ordenada por probabilidad descendente. Límite 5.000 filas.

## 7. Frontend

Dos páginas estáticas, estética limpia tipo dashboard SaaS (CSS propio, tipografía
del sistema, un color de acento; sin frameworks).

1. **index.html — Simulador**: botones para agregar eventos ("Vio producto",
   "Agregó al carrito", selector de categoría y de tiempo transcurrido), timeline
   visual de la sesión construida, y un gauge/score grande que se actualiza llamando
   a `/api/score` en cada cambio. Muestra el desglose por modelo (barras) y la
   acción recomendada. Esta página es la demo de la defensa: debe verse bien.
2. **dashboard.html — Leads**: upload de CSV → tabla ordenada por score con semáforo
   🔥/🌡/❄, filtros por segmento, y botón para descargar el CSV scoreado.
3. **Publicidad**: en ambas páginas, 2 bloques `div.ad-slot` (banner superior 728x90
   y lateral 300x250) con placeholder visible "Espacio publicitario — Google AdSense
   / Bidvertiser" y el snippet de AdSense comentado en el HTML listo para pegar el
   client-id. Esto responde al requisito de monetización de la pauta.

`nginx.conf`: sirve `/` desde static y proxy_pass de `/api/` al servicio `api:8000`.
El JS llama rutas relativas (`/api/score`) — nunca hardcodear localhost.

## 8. Docker y despliegue

### docker-compose.yml
```yaml
services:
  api:
    build: ./api
    # monta artifacts en build (COPY) — la imagen debe ser autocontenida
    expose: ["8000"]
  frontend:
    build: ./frontend
    ports: ["8080:80"]
    depends_on: [api]
```
- `api/Dockerfile`: `python:3.11-slim`, instalar requirements con torch CPU
  (`--index-url https://download.pytorch.org/whl/cpu`) para imagen liviana,
  copiar `src/`, `api/app/` y `artifacts/`, correr uvicorn.
- `docker compose up --build` debe dejar todo funcionando en `http://localhost:8080`.

### Despliegue en la nube
- **Objetivo primario: Hugging Face Spaces (Docker SDK, gratis).** Spaces corre un
  solo contenedor, así que crear un `Dockerfile.spaces` alternativo (multi-stage o
  single image) que sirva FastAPI y monte los estáticos con `StaticFiles` en la
  misma app, puerto 7860. Documentar en README ambos modos:
  local = docker-compose (2 contenedores, cumple la pauta),
  cloud = imagen única (limitación de la plataforma, explicada).
- **Alternativa si se prefiere compose real en cloud:** Render (dos servicios) —
  documentar como opción.
- El README debe terminar con la URL pública del deploy.

## 9. Milestones (commitear al final de cada uno)

1. **M1 — Esqueleto + baseline UCI**: estructura del repo, config, descarga UCI,
   features, baseline entrenado, `evaluate.py` con métricas, tests de features.
   *Criterio: `python -m src.train --dataset uci && python -m src.evaluate` corre.*
2. **M2 — Sesionización RetailRocket + EDA**: `sessionize.py` con tests
   (anti-leakage verificado por test), `01_eda.ipynb` ejecutado con embudo,
   distribución de largos y desbalance.
3. **M3 — Modelos secuenciales**: GRU y Transformer entrenados, métricas comparativas
   en `results/metrics.json`, heatmap de atención guardado.
4. **M4 — Híbrido + K-means**: ensemble con stacking, umbral calibrado, segmentos.
   `02_models.ipynb` con la tabla comparativa final.
5. **M5 — API**: FastAPI completa con tests (`test_api.py` con TestClient),
   carga de artefactos, validaciones.
6. **M6 — Frontend + compose**: simulador, dashboard, nginx, ad-slots;
   `docker compose up --build` funcional end-to-end.
7. **M7 — Deploy + README**: Space/Render arriba, README completo (problemática,
   arquitectura con diagrama ASCII o imagen, tabla de métricas, instrucciones, URL).

## 10. Criterios de aceptación globales

- `docker compose up --build` levanta todo sin pasos manuales (asumiendo
  `artifacts/` presentes en el repo).
- El test set nunca se usa para entrenar ni calibrar (umbral y stacking se calibran
  en validación). Ningún feature de entrada contiene información posterior a la
  primera transacción (test automatizado lo verifica).
- Todas las métricas de todos los modelos provienen del mismo test set y quedan en
  `results/metrics.json` + tabla en README.
- Seed fijo (42) en numpy, torch y sklearn para reproducibilidad razonable.
- Código con type hints y docstrings breves en español; nombres en inglés.
- `pytest` en verde en cada milestone.
- Commits descriptivos en español, uno o más por milestone (nada de un commit único
  gigante al final).

## 11. Fuera de alcance (no implementar)

- Autenticación de usuarios, base de datos, tracking real de eventos en vivo,
  reentrenamiento automático, A/B testing. Si sobra tiempo, priorizar pulir la demo
  del simulador y la calidad del análisis de errores en el notebook 02.
