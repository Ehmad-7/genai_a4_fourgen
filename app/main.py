"""FastAPI app: one page with four tabs (MAE, VAE, Diffusion, RAG), all running the trained models.
Run from the repo root:  uvicorn app.main:app --port 8000   then open http://127.0.0.1:8000"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import loader

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="GenAI A4: four generative models")


@app.exception_handler(loader.MissingWeights)
def _missing_weights(request, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(ValueError)
def _bad_value(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# endpoints are plain `def`, so FastAPI runs the heavy work in a worker thread
@app.get("/api/status")
def status():
    return loader.status()


@app.get("/api/mae/tiles")
def mae_tiles(n: int = 24, seed: int = 0):
    return loader.mae_tiles(min(max(n, 1), 48), seed)


@app.get("/api/mae/run")
def mae_run(index: int, seed: int = 0):
    return loader.mae_run(index, seed)


@app.get("/api/vae/classes")
def vae_classes():
    return loader.class_names()


@app.get("/api/vae/random")
def vae_random(n: int = 16, seed: int = 0):
    return loader.vae_random(min(max(n, 1), 64), seed)


@app.get("/api/vae/pair")
def vae_pair(a: str, b: str, seed: int = 0):
    return loader.vae_pair(a, b, seed)


class DiffusionRequest(BaseModel):
    character: str
    n: int = 8
    seed: int = 0


@app.post("/api/diffusion/sample")
def diffusion_sample(req: DiffusionRequest):
    return loader.diffusion_sample(req.character, min(max(req.n, 1), 16), req.seed)


class RagRequest(BaseModel):
    question: str


@app.post("/api/rag")
def rag(req: RagRequest):
    q = req.question.strip()
    if not q or len(q) > 500:
        raise HTTPException(status_code=400, detail="Write a question of at most 500 characters.")
    return loader.rag_answer(q)


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
