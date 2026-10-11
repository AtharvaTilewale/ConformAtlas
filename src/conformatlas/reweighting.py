"""Enhanced-sampling readers and reweighting.

This module turns the output of biased simulations into per-frame statistical
weights for the unbiased (Boltzmann) ensemble, in the format consumed by the
``--weights-file`` pathway of ``conformatlas analyze`` (one weight per analysed
frame, last column of a whitespace-separated text file).

Supported inputs
----------------
* PLUMED ``COLVAR`` files (``#! FIELDS`` header; columns addressed by name).
* PLUMED ``HILLS`` files written by ``METAD`` (``multivariate false``; periodic
  CVs declared through ``#! SET min_<cv>`` / ``#! SET max_<cv>``).
* Umbrella-sampling windows with harmonic restraints ``V_i(x) = 1/2 kappa (x - x_i)^2``
  (the PLUMED ``RESTRAINT`` and ``gmx wham`` convention).

Methods
-------
Metadynamics (Laio & Parrinello, PNAS 2002; well-tempered: Barducci, Bussi &
Parrinello, PRL 2008):

* ``final-bias``: ``w_t = exp(beta V(s_t, t_final))`` -- the bias at the end of
  the run is treated as a static bias. Exact for a converged (quasi-static)
  bias; frames from the filling transient are over-weighted toward the
  starting basin, so ``skip_time`` should be used to drop them.
* ``tiwary-parrinello``: ``w_t = exp(beta [V(s_t, t) - c(t)])`` with the
  time-dependent offset of Tiwary & Parrinello (J. Phys. Chem. B 2015, 119, 736),

      c(t) = (1/beta) ln [ sum_s exp(gamma beta V(s,t)/(gamma-1)) / sum_s exp(beta V(s,t)/(gamma-1)) ]

  evaluated on a regular CV grid (the integrals of eq. 12 become grid sums;
  the grid must cover the sampled CV range) and piecewise-constant between
  hill depositions. For standard (untempered) metadynamics ``gamma -> inf`` and
  ``c(t) = (1/beta) ln [ mean_s exp(beta V(s,t)) ]``. This is the same quantity
  PLUMED reports as ``metad.rct`` (``CALC_RCT``).

PLUMED conventions reproduced here: hills are evaluated as
``h * (A exp(-d^2/2sigma^2) + B)`` truncated at ``d^2/2sigma^2 = 6.25`` with
``A = 1/(1-e^-6.25)``, ``B = 1 - A`` (the "stretched" Gaussian of PLUMED >= 2.5;
``hill_shape="gaussian"`` gives the untruncated Gaussian); in well-tempered runs
the ``height`` column of HILLS is the deposited height multiplied by
``gamma/(gamma-1)`` and is divided back before the bias is reconstructed.

Validation status: everything above was checked on 1-D synthetic systems
(pymbar, PLUMED 2.10 ``sum_hills``, analytical potentials; see
validation/checks_enhanced.py). UNVERIFIED: multi-CV metadynamics (the code is
dimension-general but only 1-D runs were compared with a known potential; the
``c(t)`` grid resolution is then a user choice), HILLS written by PLUMED < 2.5
(use ``hill_shape="gaussian"``), and ``multivariate true`` hills (rejected).

Umbrella sampling:

* ``wham``: Kumar et al., J. Comput. Chem. 1992, 13, 1011 -- histogram WHAM
  iterated to a free-energy tolerance; per-frame weights are then obtained
  from the window free energies with the binless formula below.
* ``mbar``: Shirts & Chodera, J. Chem. Phys. 2008, 129, 124105 -- self-consistent
  iteration followed by Newton-Raphson on the MBAR objective; the unbiased
  weight of sample ``n`` is ``w_n = 1 / sum_k N_k exp(f_k - u_k(x_n))``.

Units: kJ/mol, ps, and whatever units the CV carries. ``R = 0.008314462618
kJ/(mol K)``. All free energies returned in kJ/mol unless marked ``reduced``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from conformatlas.utils import R_GAS_CONSTANT

logger = logging.getLogger("conformatlas.reweighting")

#: PLUMED METAD Gaussian cutoff in units of d^2 / (2 sigma^2).
DP2CUTOFF = 6.25
_STRETCH_A = 1.0 / (1.0 - np.exp(-DP2CUTOFF))
_STRETCH_B = 1.0 - _STRETCH_A


# =============================================================================
# PLUMED file readers
# =============================================================================


@dataclass
class PlumedTable:
    """Numeric table from a PLUMED output file (COLVAR, HILLS, ...)."""

    fields: list[str]
    data: np.ndarray
    constants: dict[str, str] = field(default_factory=dict)
    path: str = ""

    def column(self, name: str) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(f"{self.path}: column '{name}' not in FIELDS {self.fields}")
        return self.data[:, self.fields.index(name)]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data, columns=self.fields)

    def __len__(self) -> int:
        return int(self.data.shape[0])


def read_plumed_file(path: str | Path) -> PlumedTable:
    """Read a PLUMED file with a ``#! FIELDS`` header.

    ``#! SET key value`` lines are collected in ``constants``. When a run was
    restarted and the file contains several ``FIELDS`` headers, the column names
    must agree; rows are concatenated. Lines starting with ``#`` are ignored.
    """
    path = Path(path)
    fields: list[str] | None = None
    constants: dict[str, str] = {}
    rows: list[list[float]] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            s = line.strip()
            if not s:
                continue
            if s.startswith("#!"):
                toks = s[2:].split()
                if not toks:
                    continue
                if toks[0] == "FIELDS":
                    new_fields = toks[1:]
                    if fields is not None and new_fields != fields:
                        raise ValueError(
                            f"{path}:{ln}: FIELDS changed from {fields} to {new_fields}"
                        )
                    fields = new_fields
                elif toks[0] == "SET" and len(toks) >= 3:
                    constants[toks[1]] = " ".join(toks[2:])
                continue
            if s.startswith("#"):
                continue
            if fields is None:
                raise ValueError(f"{path}: numeric data before any '#! FIELDS' header.")
            parts = s.split()
            if len(parts) < len(fields):
                raise ValueError(
                    f"{path}:{ln}: expected {len(fields)} columns ({fields}), got {len(parts)}"
                )
            try:
                rows.append([float(p) for p in parts[: len(fields)]])
            except ValueError as exc:
                raise ValueError(f"{path}:{ln}: non-numeric entry in data row.") from exc
    if fields is None:
        raise ValueError(f"{path}: no '#! FIELDS' header found (not a PLUMED file?).")
    data = np.array(rows, dtype=np.float64).reshape(len(rows), len(fields))
    return PlumedTable(fields=fields, data=data, constants=constants, path=str(path))


def read_colvar(
    path: str | Path,
    columns: list[str] | tuple[str, ...] | None = None,
    time_column: str = "time",
) -> tuple[np.ndarray, np.ndarray | None, PlumedTable]:
    """Read CV columns (by name) from a PLUMED COLVAR file.

    Returns ``(values (n, len(columns)), times (n,) or None, table)``. With
    ``columns=None`` every column except ``time_column`` is returned.
    """
    table = read_plumed_file(path)
    if columns is None:
        columns = [f for f in table.fields if f != time_column]
    values = np.column_stack([table.column(c) for c in columns])
    times = table.column(time_column) if time_column in table.fields else None
    return values, times, table


@dataclass
class Hills:
    """Gaussian hills from a PLUMED HILLS file.

    ``height`` is the column as stored in the file; ``deposited_height`` undoes
    the well-tempered ``gamma/(gamma-1)`` rescaling PLUMED applies on output.
    ``periodic[j]`` is ``(min, max)`` for periodic CVs and ``None`` otherwise.
    """

    time: np.ndarray
    centers: np.ndarray
    sigma: np.ndarray
    height: np.ndarray
    biasf: np.ndarray
    cv_names: list[str]
    periodic: list[tuple[float, float] | None]
    heights_rescaled: bool
    path: str = ""

    @property
    def n_hills(self) -> int:
        return int(self.time.shape[0])

    @property
    def n_cv(self) -> int:
        return int(self.centers.shape[1])

    @property
    def bias_factor(self) -> float | None:
        """Well-tempered bias factor gamma, or None for standard metadynamics."""
        bf = self.biasf[np.isfinite(self.biasf)]
        if bf.size == 0 or np.all(bf <= 1.0):
            return None
        return float(bf[-1])

    @property
    def deposited_height(self) -> np.ndarray:
        h = self.height.copy()
        if self.heights_rescaled:
            wt = self.biasf > 1.0
            h[wt] *= (self.biasf[wt] - 1.0) / self.biasf[wt]
        return h


def _parse_pi(s: str) -> float:
    s = s.strip().lower()
    if s in ("pi", "+pi"):
        return float(np.pi)
    if s == "-pi":
        return float(-np.pi)
    if s.endswith("pi"):
        return float(s[:-2]) * np.pi
    return float(s)


def read_hills(path: str | Path, heights_rescaled: bool | None = None) -> Hills:
    """Read a PLUMED HILLS file (``multivariate false`` only).

    Parameters
    ----------
    heights_rescaled : bool, optional
        Whether the ``height`` column was multiplied by ``gamma/(gamma-1)``
        (PLUMED writes it that way in well-tempered runs). ``None`` applies the
        PLUMED convention automatically whenever ``biasf > 1``.
    """
    table = read_plumed_file(path)
    f = table.fields
    if "time" not in f:
        raise ValueError(f"{path}: HILLS file without 'time' column.")
    mv = table.constants.get("multivariate", "false").lower()
    if mv == "true":
        raise ValueError(
            f"{path}: multivariate (adaptive-sigma) hills are not supported; "
            "deposit hills with fixed SIGMA or convert them first."
        )
    sigma_fields = [c for c in f if c.startswith("sigma_")]
    cv_names = [c[len("sigma_") :] for c in sigma_fields]
    if not cv_names or any(c not in f for c in cv_names):
        raise ValueError(f"{path}: could not pair CV columns with sigma_<cv> columns in {f}.")
    if "height" not in f:
        raise ValueError(f"{path}: HILLS file without 'height' column.")
    centers = np.column_stack([table.column(c) for c in cv_names])
    sigma = np.column_stack([table.column(s) for s in sigma_fields])
    if np.any(sigma <= 0):
        raise ValueError(f"{path}: non-positive sigma found.")
    height = table.column("height")
    biasf = table.column("biasf") if "biasf" in f else np.full(len(table), -1.0)
    periodic: list[tuple[float, float] | None] = []
    for c in cv_names:
        lo, hi = table.constants.get(f"min_{c}"), table.constants.get(f"max_{c}")
        periodic.append(
            (_parse_pi(lo), _parse_pi(hi)) if lo is not None and hi is not None else None
        )
    if heights_rescaled is None:
        heights_rescaled = bool(np.any(biasf > 1.0))
    return Hills(
        time=table.column("time"),
        centers=centers,
        sigma=sigma,
        height=height,
        biasf=biasf,
        cv_names=cv_names,
        periodic=periodic,
        heights_rescaled=heights_rescaled,
        path=str(path),
    )


# =============================================================================
# Bias reconstruction
# =============================================================================


def _hill_kernel(dp2: np.ndarray, hill_shape: str) -> np.ndarray:
    if hill_shape == "plumed":
        k = _STRETCH_A * np.exp(-dp2) + _STRETCH_B
        k[dp2 >= DP2CUTOFF] = 0.0
        return k
    if hill_shape == "gaussian":
        return np.exp(-dp2)
    raise ValueError("hill_shape must be 'plumed' or 'gaussian'.")


def _wrapped_diff(d: np.ndarray, periodic: tuple[float, float] | None) -> np.ndarray:
    if periodic is None:
        return d
    period = periodic[1] - periodic[0]
    return d - period * np.round(d / period)


def hills_bias(
    hills: Hills,
    points: np.ndarray,
    t_max: float | None = None,
    hill_shape: str = "plumed",
    chunk: int = 2000,
) -> np.ndarray:
    """Bias ``V(s)`` accumulated from all hills with ``time <= t_max`` at ``points``.

    ``points`` has shape ``(m, n_cv)`` (or ``(m,)`` for one CV). Returns kJ/mol
    (whatever unit the HILLS heights carry).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[:, None]
    if pts.shape[1] != hills.n_cv:
        raise ValueError(f"points have {pts.shape[1]} CVs; HILLS has {hills.n_cv}.")
    sel = np.ones(hills.n_hills, dtype=bool) if t_max is None else hills.time <= t_max
    h = hills.deposited_height[sel]
    c = hills.centers[sel]
    s = hills.sigma[sel]
    v = np.zeros(pts.shape[0])
    for i0 in range(0, h.size, chunk):
        i1 = min(i0 + chunk, h.size)
        dp2 = np.zeros((pts.shape[0], i1 - i0))
        for j in range(hills.n_cv):
            d = _wrapped_diff(pts[:, j][:, None] - c[i0:i1, j][None, :], hills.periodic[j])
            dp2 += d * d / (2.0 * s[i0:i1, j][None, :] ** 2)
        v += _hill_kernel(dp2, hill_shape) @ h[i0:i1]
    return v


def hills_bias_at_times(
    hills: Hills,
    points: np.ndarray,
    times: np.ndarray,
    hill_shape: str = "plumed",
    chunk: int = 2000,
) -> np.ndarray:
    """Time-dependent bias ``V(s_t, t)``: hills deposited strictly before ``t``.

    PLUMED computes ``metad.bias`` of a step before the hill of that step is
    added, so a hill whose time equals the frame time is excluded. With this
    convention the ``metad.bias`` column of a COLVAR written by the same run is
    reproduced exactly (up to the hill-shape convention, see module docstring).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[:, None]
    times = np.asarray(times, dtype=np.float64)
    if pts.shape[0] != times.shape[0]:
        raise ValueError("points and times must have the same length.")
    order = np.argsort(hills.time, kind="stable")
    ht, hc, hs = hills.time[order], hills.centers[order], hills.sigma[order]
    hh = hills.deposited_height[order]
    # number of hills deposited strictly before each frame time
    n_before = np.searchsorted(ht, times, side="left")
    v = np.zeros(pts.shape[0])
    for i0 in range(0, hills.n_hills, chunk):
        i1 = min(i0 + chunk, hills.n_hills)
        dp2 = np.zeros((pts.shape[0], i1 - i0))
        for j in range(hills.n_cv):
            d = _wrapped_diff(pts[:, j][:, None] - hc[i0:i1, j][None, :], hills.periodic[j])
            dp2 += d * d / (2.0 * hs[i0:i1, j][None, :] ** 2)
        k = _hill_kernel(dp2, hill_shape)
        # mask hills deposited after the frame time
        k[np.arange(i0, i1)[None, :] >= n_before[:, None]] = 0.0
        v += k @ hh[i0:i1]
    return v


def default_grid(hills: Hills, bins: int = 200, pad_sigma: float = 4.0) -> list[np.ndarray]:
    """Regular CV grid covering the hill centres (+/- ``pad_sigma`` widths, or the
    declared period for periodic CVs)."""
    axes = []
    for j in range(hills.n_cv):
        per = hills.periodic[j]
        if per is not None:
            axes.append(np.linspace(per[0], per[1], bins, endpoint=False))
        else:
            lo = hills.centers[:, j].min() - pad_sigma * hills.sigma[:, j].max()
            hi = hills.centers[:, j].max() + pad_sigma * hills.sigma[:, j].max()
            axes.append(np.linspace(lo, hi, bins))
    return axes


def metad_rct(
    hills: Hills,
    temperature: float,
    grid_axes: list[np.ndarray] | None = None,
    bias_factor: float | None = None,
    hill_shape: str = "plumed",
    bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-dependent offset ``c(t)`` of Tiwary & Parrinello (2015), eq. 12.

    Returns ``(hill_times, c)`` with ``c[i]`` the value *after* hill ``i`` has
    been deposited (kJ/mol), i.e. the offset valid for frames with
    ``hill_times[i] < t <= hill_times[i+1]``. The CV integrals are evaluated as sums over the
    regular grid ``grid_axes`` (``default_grid`` when None); the grid must span
    the region where the bias is non-negligible. For standard metadynamics
    (``bias_factor`` None and ``biasf <= 1`` in the file) the untempered limit
    ``c(t) = kT ln <exp(beta V)>_grid`` is used.
    """
    kT = R_GAS_CONSTANT * temperature
    beta = 1.0 / kT
    gamma = bias_factor if bias_factor is not None else hills.bias_factor
    axes = grid_axes if grid_axes is not None else default_grid(hills, bins)
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.column_stack([m.ravel() for m in mesh])
    order = np.argsort(hills.time, kind="stable")
    h = hills.deposited_height[order]
    c = hills.centers[order]
    s = hills.sigma[order]
    v = np.zeros(pts.shape[0])
    rct = np.empty(hills.n_hills)
    log_n = np.log(pts.shape[0])
    for i in range(hills.n_hills):
        dp2 = np.zeros(pts.shape[0])
        for j in range(hills.n_cv):
            d = _wrapped_diff(pts[:, j] - c[i, j], hills.periodic[j])
            dp2 += d * d / (2.0 * s[i, j] ** 2)
        v += h[i] * _hill_kernel(dp2, hill_shape)
        if gamma is None:
            rct[i] = kT * (logsumexp(beta * v) - log_n)
        else:
            num = logsumexp(gamma / (gamma - 1.0) * beta * v)
            den = logsumexp(1.0 / (gamma - 1.0) * beta * v)
            rct[i] = kT * (num - den)
    return hills.time[order], rct


@dataclass
class MetadReweightResult:
    """Per-frame metadynamics reweighting."""

    times: np.ndarray
    bias: np.ndarray  # V used in the exponent (kJ/mol)
    rct: np.ndarray  # c(t) per frame (zeros for final-bias)
    log_weights: np.ndarray  # beta (V - c), before normalisation
    weights: np.ndarray  # normalised to mean 1 over kept frames (0 for skipped)
    kept: np.ndarray  # boolean mask of frames with time >= skip_time
    method: str
    bias_factor: float | None
    hill_times: np.ndarray | None = None
    rct_history: np.ndarray | None = None


def metad_weights(
    cv_values: np.ndarray,
    cv_times: np.ndarray,
    hills: Hills,
    temperature: float,
    method: str = "tiwary-parrinello",
    skip_time: float = 0.0,
    grid_axes: list[np.ndarray] | None = None,
    bins: int = 200,
    bias_factor: float | None = None,
    hill_shape: str = "plumed",
) -> MetadReweightResult:
    """Unbiased-ensemble weights for frames of a metadynamics run.

    Parameters
    ----------
    cv_values : (n, n_cv) CV values of the frames (COLVAR columns in HILLS order)
    cv_times : (n,) frame times (same unit as HILLS ``time``)
    method : "final-bias" or "tiwary-parrinello"
    skip_time : frames with ``time < skip_time`` receive weight 0 (filling transient)
    """
    kT = R_GAS_CONSTANT * temperature
    cv = np.asarray(cv_values, dtype=np.float64)
    if cv.ndim == 1:
        cv = cv[:, None]
    t = np.asarray(cv_times, dtype=np.float64)
    kept = t >= skip_time
    gamma = bias_factor if bias_factor is not None else hills.bias_factor
    hill_times = rct_hist = None
    if method == "final-bias":
        bias = hills_bias(hills, cv, hill_shape=hill_shape)
        rct = np.zeros_like(bias)
    elif method == "tiwary-parrinello":
        bias = hills_bias_at_times(hills, cv, t, hill_shape=hill_shape)
        hill_times, rct_hist = metad_rct(
            hills, temperature, grid_axes, gamma, hill_shape=hill_shape, bins=bins
        )
        idx = np.searchsorted(hill_times, t, side="left") - 1  # hills strictly before t
        rct = np.where(idx >= 0, rct_hist[np.clip(idx, 0, None)], 0.0)
    else:
        raise ValueError("method must be 'final-bias' or 'tiwary-parrinello'.")
    logw = (bias - rct) / kT
    w = np.zeros_like(logw)
    if kept.any():
        lw = logw[kept]
        w[kept] = np.exp(lw - logsumexp(lw)) * kept.sum()
    return MetadReweightResult(
        times=t,
        bias=bias,
        rct=rct,
        log_weights=logw,
        weights=w,
        kept=kept,
        method=method,
        bias_factor=gamma,
        hill_times=hill_times,
        rct_history=rct_hist,
    )


# =============================================================================
# Frame <-> COLVAR alignment and output
# =============================================================================


def match_times(
    frame_times: np.ndarray, cv_times: np.ndarray, tolerance: float | None = None
) -> np.ndarray:
    """Index of the COLVAR row closest in time to each trajectory frame.

    Raises if any frame is further than ``tolerance`` from its nearest COLVAR
    time (default tolerance: half the smallest COLVAR spacing).
    """
    ft = np.asarray(frame_times, dtype=np.float64)
    ct = np.asarray(cv_times, dtype=np.float64)
    order = np.argsort(ct, kind="stable")
    cts = ct[order]
    pos = np.clip(np.searchsorted(cts, ft), 1, len(cts) - 1)
    left, right = cts[pos - 1], cts[pos]
    choose_right = np.abs(right - ft) < np.abs(ft - left)
    idx = np.where(choose_right, pos, pos - 1)
    if tolerance is None:
        dts = np.diff(cts)
        tolerance = 0.5 * float(dts[dts > 0].min()) if np.any(dts > 0) else 0.0
    err = np.abs(cts[idx] - ft)
    if np.any(err > tolerance + 1e-9):
        worst = int(np.argmax(err))
        raise ValueError(
            f"Frame time {ft[worst]} has no COLVAR row within {tolerance} (nearest {cts[idx[worst]]})."
        )
    return order[idx]


def write_weights_file(
    path: str | Path, times: np.ndarray, weights: np.ndarray, header: str = ""
) -> Path:
    """Write ``time weight`` rows readable by ``conformatlas.features.read_weights_file``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for line in header.splitlines():
            fh.write(f"# {line}\n")
        fh.write("# time weight\n")
        for t, w in zip(np.asarray(times), np.asarray(weights)):
            fh.write(f"{t:.6f} {w:.10e}\n")
    return path


def pmf_1d(
    values: np.ndarray,
    weights: np.ndarray,
    temperature: float,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Weighted 1-D PMF ``-kT ln p(x)`` (minimum set to zero).

    Columns: ``center, pmf_kJ_mol, probability, weight_sum, count``; empty bins
    have NaN PMF.
    """
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    hist, edges = np.histogram(v, bins=bins, range=value_range, weights=w)
    counts, _ = np.histogram(v, bins=edges)
    p = hist / hist.sum() if hist.sum() > 0 else hist
    pmf = np.full(len(p), np.nan)
    m = p > 0
    pmf[m] = -R_GAS_CONSTANT * temperature * np.log(p[m])
    pmf -= np.nanmin(pmf)
    return pd.DataFrame(
        {
            "center": 0.5 * (edges[1:] + edges[:-1]),
            "pmf_kJ_mol": pmf,
            "probability": p,
            "weight_sum": hist,
            "count": counts,
        }
    )


# =============================================================================
# Umbrella sampling: WHAM and MBAR
# =============================================================================


@dataclass
class UmbrellaWindow:
    """One umbrella window: samples of the biased coordinate and its harmonic restraint
    ``V(x) = 1/2 kappa (x - center)^2`` (kappa in kJ/mol per CV unit squared)."""

    samples: np.ndarray
    center: float
    kappa: float
    name: str = ""
    period: float | None = None  # period of the CV for angular coordinates

    def bias(self, x: np.ndarray) -> np.ndarray:
        d = np.asarray(x, dtype=np.float64) - self.center
        if self.period is not None:
            d = d - self.period * np.round(d / self.period)
        return 0.5 * self.kappa * d * d


@dataclass
class UmbrellaResult:
    """Output of WHAM or MBAR on umbrella windows."""

    method: str
    window_free_energies: np.ndarray  # kJ/mol, first window = 0
    weights: list[np.ndarray]  # unbiased weights per window (sum over all = 1)
    pmf: pd.DataFrame
    n_iter: int
    converged: bool
    reduced_free_energies: np.ndarray  # f_k = F_k / kT


def _window_arrays(windows: list[UmbrellaWindow]):
    x = np.concatenate([np.asarray(w.samples, dtype=np.float64).ravel() for w in windows])
    n_k = np.array([np.asarray(w.samples).size for w in windows], dtype=np.int64)
    return x, n_k


def _unbiased_weights_from_f(
    windows: list[UmbrellaWindow], f_reduced: np.ndarray, kT: float, g: np.ndarray | None
) -> list[np.ndarray]:
    """Binless weights ``w_n = 1 / sum_k (N_k/g_k) exp(f_k - beta V_k(x_n))``."""
    x, n_k = _window_arrays(windows)
    n_eff = n_k.astype(float) if g is None else n_k / np.asarray(g, dtype=float)
    log_den = np.full(x.size, -np.inf)
    terms = np.empty((len(windows), x.size))
    for k, w in enumerate(windows):
        terms[k] = np.log(n_eff[k]) + f_reduced[k] - w.bias(x) / kT
    log_den = logsumexp(terms, axis=0)
    logw = -log_den
    w_all = np.exp(logw - logsumexp(logw))
    out, i0 = [], 0
    for n in n_k:
        out.append(w_all[i0 : i0 + n])
        i0 += n
    return out


def wham(
    windows: list[UmbrellaWindow],
    temperature: float,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
    tol: float = 1e-8,
    max_iter: int = 100_000,
    g: np.ndarray | None = None,
) -> UmbrellaResult:
    """Histogram WHAM (Kumar et al. 1992) for 1-D umbrella windows.

    Iterates ``P(b) = sum_i n_ib / sum_i (N_i/g_i) exp(f_i - beta V_i(x_b))`` and
    ``f_i = -ln sum_b P(b) exp(-beta V_i(x_b))`` until the largest change in any
    ``f_i`` (reduced units) is below ``tol``. ``g`` are optional per-window
    statistical inefficiencies (Kumar's ``g_i``; default 1). The bias of a bin is
    evaluated at its centre. Per-frame weights are derived from the converged
    ``f_i`` with the binless estimator, so the PMF returned is the histogram of
    those weights (identical to the bin-based WHAM profile for the same bins).
    """
    kT = R_GAS_CONSTANT * temperature
    x, n_k = _window_arrays(windows)
    if value_range is None:
        value_range = (float(x.min()), float(x.max()))
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    n_eff = n_k.astype(float) if g is None else n_k / np.asarray(g, dtype=float)
    hist = np.zeros((len(windows), bins))
    for k, w in enumerate(windows):
        hk, _ = np.histogram(np.asarray(w.samples, dtype=float), bins=edges)
        hist[k] = hk * (n_eff[k] / n_k[k])  # scale counts by 1/g_k
    n_tot_b = hist.sum(axis=0)
    bias_kb = np.array([w.bias(centers) / kT for w in windows])  # (K, B) reduced
    f = np.zeros(len(windows))
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        log_den = logsumexp(np.log(n_eff)[:, None] + f[:, None] - bias_kb, axis=0)
        with np.errstate(divide="ignore"):
            log_p = np.where(n_tot_b > 0, np.log(n_tot_b) - log_den, -np.inf)
        f_new = -logsumexp(log_p[None, :] - bias_kb, axis=1)
        f_new -= f_new[0]
        delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if delta < tol:
            converged = True
            break
    if not converged:
        logger.warning(f"WHAM did not converge in {max_iter} iterations (last change {delta:.2e}).")
    weights = _unbiased_weights_from_f(windows, f, kT, g)
    pmf = pmf_1d(x, np.concatenate(weights), temperature, bins=bins, value_range=value_range)
    return UmbrellaResult(
        method="wham",
        window_free_energies=f * kT,
        weights=weights,
        pmf=pmf,
        n_iter=it,
        converged=converged,
        reduced_free_energies=f,
    )


def mbar_solve(
    u_kn: np.ndarray,
    n_k: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 10_000,
    n_self_consistent: int = 10,
    initial_f: np.ndarray | None = None,
) -> tuple[np.ndarray, int, bool]:
    """Solve the MBAR equations for reduced free energies ``f_k`` (``f_0 = 0``).

    ``u_kn[k, n]`` is the reduced potential of sample ``n`` in state ``k``;
    ``n_k[k]`` the number of samples drawn from state ``k`` (states with
    ``n_k = 0`` are allowed and their ``f_k`` is evaluated at the end).
    Self-consistent iterations (eq. 11 of Shirts & Chodera) are followed by
    Newton-Raphson on the convex objective ``sum_n ln sum_k N_k e^{f_k - u_kn}
    - sum_k N_k f_k`` (gradient ``N_i (sum_n W_in - 1)``, Hessian
    ``delta_ij N_i sum_n W_in - N_i N_j sum_n W_in W_jn``) until the largest
    change in ``f`` is below ``tol``. Returns ``(f, n_iterations, converged)``.
    """
    u = np.asarray(u_kn, dtype=np.float64)
    n = np.asarray(n_k, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != n.shape[0]:
        raise ValueError("u_kn must be (K, N) and n_k (K,).")
    if n.sum() != u.shape[1]:
        raise ValueError(f"sum(n_k)={int(n.sum())} must equal the number of samples {u.shape[1]}.")
    sampled = n > 0
    us, ns = u[sampled], n[sampled]
    log_n = np.log(ns)
    f = np.zeros(us.shape[0]) if initial_f is None else np.asarray(initial_f, float)[sampled].copy()
    f = f - f[0]

    def log_den(fv):
        return logsumexp(log_n[:, None] + fv[:, None] - us, axis=0)

    def sc_update(fv):
        fn = -logsumexp(-us - log_den(fv)[None, :], axis=1)
        return fn - fn[0]

    def grad_hess(fv):
        w = np.exp(fv[:, None] - us - log_den(fv)[None, :])  # W_in
        grad = ns * (w.sum(axis=1) - 1.0)
        hess = np.diag(ns * w.sum(axis=1)) - (ns[:, None] * w) @ (ns[:, None] * w).T
        return grad, hess

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        if it <= n_self_consistent or us.shape[0] == 1:
            f_new = sc_update(f)
        else:
            grad, hess = grad_hess(f)
            try:
                step = np.linalg.solve(hess[1:, 1:], grad[1:])
                f_new = f.copy()
                f_new[1:] -= step
                if not np.all(np.isfinite(f_new)):
                    raise np.linalg.LinAlgError
                # accept only if the gradient norm does not grow (else fall back)
                g_after, _ = grad_hess(f_new)
                if np.linalg.norm(g_after) > np.linalg.norm(grad):
                    f_new = sc_update(f)
            except np.linalg.LinAlgError:
                f_new = sc_update(f)
        delta = float(np.max(np.abs(f_new - f)))
        f = f_new
        if delta < tol:
            converged = True
            break
    if not converged:
        logger.warning(f"MBAR did not converge in {max_iter} iterations (last change {delta:.2e}).")
    # free energies of unsampled states
    ld = log_den(f)
    f_all = np.empty(u.shape[0])
    f_all[sampled] = f
    f_all[~sampled] = -logsumexp(-u[~sampled] - ld[None, :], axis=1)
    return f_all, it, converged


def mbar_weights(
    u_kn: np.ndarray, n_k: np.ndarray, f_k: np.ndarray, u_target: np.ndarray | None = None
):
    """Normalised MBAR weights of every sample in a target state (``u_target`` reduced
    potential per sample; zeros = the unbiased state when the biases are the only
    difference between states)."""
    u = np.asarray(u_kn, dtype=np.float64)
    n = np.asarray(n_k, dtype=np.float64)
    m = n > 0
    ld = logsumexp(np.log(n[m])[:, None] + np.asarray(f_k)[m][:, None] - u[m], axis=0)
    ut = np.zeros(u.shape[1]) if u_target is None else np.asarray(u_target, dtype=np.float64)
    logw = -ut - ld
    return np.exp(logw - logsumexp(logw))


def mbar(
    windows: list[UmbrellaWindow],
    temperature: float,
    bins: int = 100,
    value_range: tuple[float, float] | None = None,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> UmbrellaResult:
    """MBAR (Shirts & Chodera 2008) for 1-D harmonic umbrella windows.

    The reduced potential of sample ``n`` in window ``k`` is ``beta V_k(x_n)``
    (the unbiased Hamiltonian is common to all windows and cancels). Returns the
    window free energies, the unbiased per-frame weights and the PMF histogram
    of those weights.
    """
    kT = R_GAS_CONSTANT * temperature
    x, n_k = _window_arrays(windows)
    u_kn = np.array([w.bias(x) / kT for w in windows])
    f, n_iter, conv = mbar_solve(u_kn, n_k, tol=tol, max_iter=max_iter)
    w_all = mbar_weights(u_kn, n_k, f)
    weights, i0 = [], 0
    for n in n_k:
        weights.append(w_all[i0 : i0 + n])
        i0 += n
    if value_range is None:
        value_range = (float(x.min()), float(x.max()))
    pmf = pmf_1d(x, w_all, temperature, bins=bins, value_range=value_range)
    return UmbrellaResult(
        method="mbar",
        window_free_energies=f * kT,
        weights=weights,
        pmf=pmf,
        n_iter=n_iter,
        converged=conv,
        reduced_free_energies=f,
    )


def umbrella_windows_from_colvars(
    paths: list[str | Path],
    cv_column: str,
    centers: list[float],
    kappas: list[float] | float,
    time_column: str = "time",
    skip_time: float = 0.0,
    period: float | None = None,
) -> tuple[list[UmbrellaWindow], list[np.ndarray]]:
    """Build umbrella windows from one COLVAR per window.

    Returns ``(windows, times)`` where ``times[k]`` are the retained frame times
    of window ``k`` (needed to write per-trajectory weight files).
    """
    if np.isscalar(kappas):
        kappas = [float(kappas)] * len(paths)
    if not (len(paths) == len(centers) == len(kappas)):
        raise ValueError("paths, centers and kappas must have the same length.")
    windows, times = [], []
    for p, c, k in zip(paths, centers, kappas):
        vals, t, _ = read_colvar(p, [cv_column], time_column)
        keep = np.ones(vals.shape[0], dtype=bool) if t is None else t >= skip_time
        windows.append(
            UmbrellaWindow(
                samples=vals[keep, 0],
                center=float(c),
                kappa=float(k),
                name=Path(p).name,
                period=period,
            )
        )
        times.append(np.arange(keep.sum()) if t is None else t[keep])
    return windows, times


def write_umbrella_outputs(
    result: UmbrellaResult,
    out_dir: str | Path,
    times: list[np.ndarray] | None = None,
    window_names: list[str] | None = None,
) -> dict[str, Path]:
    """Write ``pmf.csv``, ``window_free_energies.csv`` and one ``weights_<window>.dat``
    per window (weights normalised to mean 1 over *all* frames, so the files
    can be concatenated across trajectories without rescaling)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    pmf_path = out_dir / "pmf.csv"
    result.pmf.to_csv(pmf_path, index=False)
    written["pmf"] = pmf_path
    n_total = sum(w.size for w in result.weights)
    names = window_names or [f"window{k}" for k in range(len(result.weights))]
    fe = pd.DataFrame(
        {
            "window": names,
            "n_samples": [w.size for w in result.weights],
            "free_energy_kJ_mol": result.window_free_energies,
        }
    )
    fe_path = out_dir / "window_free_energies.csv"
    fe.to_csv(fe_path, index=False)
    written["window_free_energies"] = fe_path
    for k, w in enumerate(result.weights):
        t = times[k] if times is not None else np.arange(w.size)
        p = write_weights_file(
            out_dir / f"weights_{names[k]}.dat",
            t,
            w * n_total,
            header=f"{result.method} unbiased weights, window {names[k]} (mean 1 over all frames)",
        )
        written[f"weights_{names[k]}"] = p
    return written
