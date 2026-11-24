"""Loads the trained models lazily for the web app (one bundle per tab, loaded on first use)."""
import base64, functools, io, sys, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = ROOT / "weights"
LOCK = threading.Lock()                       # one heavy job (diffusion / generation) at a time
UHAT_ROOT = str(ROOT / "starter" / "data" / "uhat")

MEDICAL_NOTICE = ("Research prototype, not medical advice. The answer is written by a small language model from a few "
                  "PubMed abstracts and can be wrong, incomplete or unsupported by the passages. Do not use it for "
                  "diagnosis or treatment decisions; ask a qualified health professional.")


class MissingWeights(Exception):
    pass


def _need(name):
    p = WEIGHTS / name
    if not p.exists():
        raise MissingWeights(f"weights/{name} not found. Put the trained weights in weights/ "
                             f"(python scripts/download_weights.py).")
    return p


def to_png(img, scale=4):
    """tensor (C,H,W) or (H,W) with values in [0,1] -> data URI (PNG, nearest-neighbour upscaled)."""
    a = img.detach().cpu().float().clamp(0, 1)
    if a.dim() == 3:
        a = a.permute(1, 2, 0)
    a = (a.numpy() * 255).round().astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    im = Image.fromarray(a).resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def status():
    files = ["mae.pt", "mae_classifier.pt", "vae.pt", "ddpm.pt", "rag_index.npy", "rag_chunk_doc_ids.npy"]
    return {"device": DEVICE, "weights": {f: (WEIGHTS / f).exists() for f in files}}


# ------------------------------------------------------------------ MAE tab
@functools.lru_cache(maxsize=1)
def _mae():
    from eurosat import load_eurosat_split, CLASS_NAMES
    from part_a_mae.model import MAE
    model = MAE()
    model.load_state_dict(torch.load(_need("mae.pt"), map_location="cpu"))
    head = nn.Linear(192, 10)
    head.load_state_dict(torch.load(_need("mae_classifier.pt"), map_location="cpu"))
    x, y = load_eurosat_split("test")
    return {"model": model.eval(), "head": head.eval(), "x": x, "y": y, "names": CLASS_NAMES}


def mae_tiles(n=24, seed=0):
    b = _mae()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randperm(len(b["x"]), generator=g)[:n].tolist()
    return [{"id": i, "label": b["names"][int(b["y"][i])], "img": to_png(b["x"][i], 2)} for i in ids]


def mae_run(index, seed=0):
    from eurosat import patchify, unpatchify
    b = _mae()
    if not 0 <= index < len(b["x"]):
        raise ValueError("tile index out of range")
    model, head = b["model"], b["head"]
    p = patchify(b["x"][index : index + 1])                              # (1, 64, 192)
    g = torch.Generator().manual_seed(seed)
    ids_keep = torch.rand(1, 64, generator=g).argsort(dim=1)[:, : model.num_keep]
    mask = torch.ones(1, 64)
    mask.scatter_(1, ids_keep, 0.0)                                      # 1 = masked
    with torch.no_grad():
        pred = model.decode(model.encode(p, ids_keep), ids_keep)
        probs = torch.softmax(head(model.features(p)), dim=1)[0]         # class from the full tile
    m = mask.unsqueeze(-1)
    masked_in = p * (1 - m) + 0.5 * m
    recon = p * (1 - m) + pred.clamp(0, 1) * m
    top = probs.topk(3)
    return {"masked": to_png(unpatchify(masked_in)[0], 4), "reconstruction": to_png(unpatchify(recon)[0], 4),
            "original": to_png(unpatchify(p)[0], 4), "true_label": b["names"][int(b["y"][index])],
            "predicted": b["names"][int(top.indices[0])],
            "top3": [{"label": b["names"][int(i)], "prob": float(v)} for v, i in zip(top.values, top.indices)],
            "visible_patches": int(model.num_keep), "total_patches": 64}


# ------------------------------------------------------------------ VAE / diffusion tabs
@functools.lru_cache(maxsize=1)
def class_names():
    from uhat import _find_split_dir
    return sorted(c.name.lower() for c in _find_split_dir(UHAT_ROOT, "char", "train").iterdir() if c.is_dir())


@functools.lru_cache(maxsize=1)
def _vae():
    from uhat import load_uhat_characters
    from part_b_vae.model import VAE
    names = class_names()
    x, y, _ = load_uhat_characters(root=UHAT_ROOT, split="test", class_names=names)
    model = VAE()
    model.load_state_dict(torch.load(_need("vae.pt"), map_location="cpu"))
    by_class = {n: (y == i).nonzero().flatten() for i, n in enumerate(names)}
    return {"model": model.eval(), "x": x, "by_class": by_class}


def vae_random(n=16, seed=0):
    model = _vae()["model"]
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        imgs = torch.sigmoid(model.decode(torch.randn(n, model.latent_dim, generator=g)))
    return [to_png(i, 4) for i in imgs]


def vae_pair(class_a, class_b, seed=0, steps=21):
    b = _vae()
    for c in (class_a, class_b):
        if c not in b["by_class"] or len(b["by_class"][c]) == 0:
            raise ValueError(f"no test images for character '{c}'")
    g = torch.Generator().manual_seed(seed)
    pick = lambda c: int(b["by_class"][c][torch.randint(len(b["by_class"][c]), (1,), generator=g)])
    ia, ib = pick(class_a), pick(class_b)
    xa, xb = b["x"][ia : ia + 1], b["x"][ib : ib + 1]
    with torch.no_grad():
        mu_a, _ = b["model"].encode(xa)
        mu_b, _ = b["model"].encode(xb)
        ts = torch.linspace(0, 1, steps).unsqueeze(1)
        frames = torch.sigmoid(b["model"].decode((1 - ts) * mu_a + ts * mu_b))
    return {"a": to_png(xa[0], 4), "b": to_png(xb[0], 4), "frames": [to_png(f, 4) for f in frames],
            "ts": ts.flatten().tolist()}


@functools.lru_cache(maxsize=1)
def _ddpm():
    from part_c_diffusion.model import build_unet
    model = build_unet()
    model.load_state_dict(torch.load(_need("ddpm.pt"), map_location="cpu"))
    return model.to(DEVICE).eval()


def diffusion_sample(character, n=8, seed=0):
    from part_c_diffusion.sampler import p_sample_loop, to_img
    names = class_names()
    if character not in names:
        raise ValueError(f"unknown character '{character}'")
    model = _ddpm()
    labels = torch.full((n,), names.index(character), device=DEVICE, dtype=torch.long)
    with LOCK:
        x, _ = p_sample_loop(model, labels, DEVICE, seed=seed)         # 1000 denoising steps
    return [to_png(to_img(x[i]), 4) for i in range(n)]


# ------------------------------------------------------------------ RAG tab
@functools.lru_cache(maxsize=1)
def _rag():
    from bioasq import load_corpus
    from part_d_rag.chunking import build_chunks
    from part_d_rag.index import Embedder
    from part_d_rag.search import Index
    emb = np.load(_need("rag_index.npy"))
    cids = np.load(_need("rag_chunk_doc_ids.npy"))
    texts, ids = build_chunks(load_corpus())
    if len(texts) != len(emb) or not np.array_equal(cids, np.array(ids)):
        raise RuntimeError("the saved index does not match the chunks of the corpus")
    return {"texts": texts, "ids": cids, "index": Index(emb, cids, DEVICE), "embedder": Embedder(DEVICE)}


@functools.lru_cache(maxsize=1)
def _generator():
    from part_d_rag.generate import Generator
    return Generator(DEVICE)


def rag_answer(question):
    from part_d_rag.generate import build_messages, parse_citations, is_idk
    b = _rag()
    q = b["embedder"].encode([question])
    idx, scores = b["index"].search_chunks(q, k=5)
    passages = [b["texts"][i] for i in idx[0]]
    with LOCK:
        answer = _generator().generate(build_messages(question, passages))
    cited = [c for c in parse_citations(answer) if 1 <= c <= 5]
    return {"answer": answer, "idk": is_idk(answer), "cited": cited, "notice": MEDICAL_NOTICE,
            "passages": [{"n": j + 1, "abstract_id": int(b["ids"][i]), "score": float(s), "text": b["texts"][i],
                          "cited": (j + 1) in cited} for j, (i, s) in enumerate(zip(idx[0], scores[0]))]}
