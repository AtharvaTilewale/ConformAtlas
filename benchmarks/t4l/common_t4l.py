"""Shared helpers for the T4 lysozyme benchmark (benchmarks/CRITERIA_T4L_HIV.md).

Inputs are the 100 ps copies written by ``prepare_t4l.sh`` under $CA_BENCH_DATA/t4l/<set>/prepared/.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
if os.environ.get("CA_SRC"):
    sys.path.insert(0, os.environ["CA_SRC"])

DATA = Path(os.environ.get("CA_BENCH_DATA", Path(__file__).resolve().parents[1] / "data"))
SETS = {
    "ff99SBstar": (
        "ff99SBstar-ILDN_TIP4P2005_5x1us",
        ["md1us4", "md1us5", "md1us6", "md1us7", "md1us8"],
    ),
    "ff15ipq": ("ff15ipq_SPCEb_3x1us", ["sim1", "sim2", "sim3"]),
}
SELECTIONS = {"calpha": "name CA", "heavy": "not name H*"}
TEMPERATURE = 300.0  # ref-t of both run files (V-rescale)
KB_SI = 1.380649e-23 * 6.02214076e23 / 1000.0
N_MODES = 10
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def tpr(force_field: str) -> Path:
    return DATA / "t4l" / SETS[force_field][0] / "prepared" / "protein.tpr"


def xtc(force_field: str, rep: str) -> Path:
    return DATA / "t4l" / SETS[force_field][0] / "prepared" / f"{rep}_100ps_whole.xtc"


def replicates(force_field: str) -> list[str]:
    return list(SETS[force_field][1])


def universe(force_field: str, rep: str):
    import MDAnalysis as mda

    return mda.Universe(str(tpr(force_field)), str(xtc(force_field, rep)))


def raw(force_field: str, rep: str, selection: str) -> np.ndarray:
    u = universe(force_field, rep)
    return u.trajectory.timeseries(
        u.select_atoms(SELECTIONS.get(selection, selection)), order="fac"
    ).astype(np.float64)


def kabsch_batch(mobile_fit: np.ndarray, ref_fit: np.ndarray, mobile_all: np.ndarray) -> np.ndarray:
    """Superpose every frame of ``mobile_all`` using the rotation that fits ``mobile_fit`` onto ``ref_fit``."""
    mc = mobile_fit.mean(axis=1, keepdims=True)
    rc = ref_fit.mean(axis=0)
    h = np.einsum("fai,aj->fij", mobile_fit - mc, ref_fit - rc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum("fij,fjk->fik", u, vt)))
    u[:, :, 2] *= d[:, None]
    rot = np.einsum("fij,fjk->fik", u, vt)
    return np.einsum("fai,fij->faj", mobile_all - mc, rot) + rc


_REF = None


def reference_calpha() -> np.ndarray:
    global _REF
    if _REF is None:
        _REF = raw("ff99SBstar", "md1us4", "calpha")[0]
    return _REF


def fitted(force_field: str, rep: str, selection: str) -> np.ndarray:
    """Coordinates of ``selection`` after Cα Kabsch fit to the first frame of md1us4 (Å, float64)."""
    ca = raw(force_field, rep, "calpha")
    sel = ca if selection == "calpha" else raw(force_field, rep, selection)
    return kabsch_batch(ca, reference_calpha(), sel)


def pooled(force_field: str, selection: str) -> tuple[np.ndarray, list[int]]:
    """Pooled fitted coordinates, rounded once to float32 and returned as float64, so that every tool (including
    those that store float32 internally, e.g. MDAnalysis' MemoryReader) receives exactly the same numbers."""
    parts = [fitted(force_field, r, selection) for r in replicates(force_field)]
    return np.concatenate(parts).astype(np.float32).astype(np.float64), [len(p) for p in parts]


def save_json(name: str, obj) -> Path:
    p = OUT / name
    p.write_text(json.dumps(obj, indent=2, default=float))
    return p
