"""API FastAPI de LeadScore (§6). Prefijo /api. Solo lee de artifacts/."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile

from src import config
from src.inference import Artifacts, load_artifacts, score_batch, score_session

from api.app.schemas import BatchRow, HealthResponse, ScoreRequest, ScoreResponse

_BATCH_COLUMNS = [
    "lead_id", "n_views", "n_addtocart", "n_unique_items",
    "duration_sec", "hour_of_day", "day_of_week",
]
_BATCH_LIMIT = 5000

# Los artefactos se cargan una vez al importar el módulo (imagen autocontenida).
try:
    ARTIFACTS: Artifacts | None = load_artifacts()
except Exception:  # pragma: no cover - la API arranca aunque falten artefactos
    ARTIFACTS = None

app = FastAPI(title="LeadScore API", version=config.VERSION)
router = APIRouter(prefix="/api")


def _require_models() -> Artifacts:
    if ARTIFACTS is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados.")
    return ARTIFACTS


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok", models_loaded=ARTIFACTS is not None, version=config.VERSION
    )


@router.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    art = _require_models()
    result = score_session(
        art,
        [e.model_dump() for e in req.events],
        req.hour_of_day,
        req.day_of_week,
    )
    return ScoreResponse(**result)


@router.post("/score/batch", response_model=list[BatchRow])
async def score_batch_endpoint(file: UploadFile = File(...)) -> list[BatchRow]:
    art = _require_models()
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # CSV inválido
        raise HTTPException(status_code=422, detail=f"CSV inválido: {exc}") from exc

    missing = [c for c in _BATCH_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan columnas: {missing}")
    if len(df) > _BATCH_LIMIT:
        raise HTTPException(
            status_code=422, detail=f"Máximo {_BATCH_LIMIT} filas (recibidas {len(df)})."
        )
    return [BatchRow(**row) for row in score_batch(art, df)]


app.include_router(router)
