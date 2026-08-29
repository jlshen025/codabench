"""Inference-window resolution shared by the Phase-2 prediction-rebuild stages.

Every default reproduces the original hardcoded behaviour (the phase2 `inference/`
tree, windows 1..8 in ascending order), so an UNSET environment leaves each stage
byte-for-byte unchanged.

Env:
  SEAWINDS_INFER_DIR   inference tree holding window_<id>/ (default $SEAWINDS_PHASE2_DIR/inference)
  SEAWINDS_WINDOW_IDS  explicit comma list of on-disk window ids (escape hatch; default = enumerate)
"""
from __future__ import annotations
import json, os, re

REQUIRED_FILES = ("metadata.json", "context_hres_north_sea.parquet",
                  "context_reanalysis_north_sea.parquet")


def infer_dir() -> str:
    return os.environ.get("SEAWINDS_INFER_DIR") or f"{os.environ['SEAWINDS_PHASE2_DIR']}/inference"


def window_ids(inf: str | None = None) -> list[int]:
    """The <id> of each window_<id>/ directory on disk, ascending."""
    inf = inf or infer_dir()
    env = os.environ.get("SEAWINDS_WINDOW_IDS", "").strip()
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    assert os.path.isdir(inf), f"inference dir not found: {inf}"
    ids = sorted(int(m.group(1)) for m in
                 (re.fullmatch(r"window_(\d+)", d) for d in os.listdir(inf)) if m)
    assert ids, f"no window_<id>/ directories under {inf}"
    return ids


def metadata(wid: int, inf: str | None = None) -> dict:
    inf = inf or infer_dir()
    return json.load(open(f"{inf}/window_{wid}/metadata.json"))


def windows(inf: str | None = None) -> list[tuple[int, int]]:
    """[(on-disk window id, submission `window` column value = metadata id - 1)],
    ordered by the window column so the CSV stays window-major ascending."""
    inf = inf or infer_dir()
    out = sorted(((w, int(metadata(w, inf)["id"]) - 1) for w in window_ids(inf)),
                 key=lambda t: t[1])
    idx = [t[1] for t in out]
    assert idx == list(range(len(idx))), \
        f"metadata ids do not form a 1..n set in {inf}: window column would be {idx}"
    return out
