"""Markov state models on the retained-component space.

Workflow (Prinz et al., J. Chem. Phys. 2011, 134, 174105):

1. discretise the frames in the retained PC / TICA / dihedral-PC space into
   microstates with k-means (Lloyd iterations, k-means++ seeding, deterministic
   seed) or regular-space clustering;
2. count transitions at lag ``tau`` frames with a sliding window *within each
   replicate* (trajectory junctions are never counted as transitions);
3. estimate a transition matrix on the largest strongly connected set of the
   count matrix, either row-normalised (maximum likelihood without constraints)
   or with the detailed-balance constraint (reversible maximum-likelihood
   estimator, iterative fixed point of Bowman et al. 2009 / Prinz et al. 2011);
4. stationary distribution, eigen-decomposition, implied timescales
   ``t_i = -tau / ln lambda_i`` scanned over a list of lags, with a plateau
   report (the lag is *reported*, never chosen silently);
5. Chapman-Kolmogorov test comparing the model prediction ``T(tau)^k`` with the
   transition matrix estimated directly at ``k tau``;
6. PCCA+ metastable memberships (Deuflhard & Weber 2005; Roeblitz & Weber 2013)
   and crisp assignment; mean first-passage times between metastable sets;
7. overlap table between the kinetic (PCCA+) sets and the watershed basins of
   the free-energy landscape.

Assumptions stated once here and in every docstring that depends on them: the
discretised process is assumed Markovian at the chosen lag (check the implied
timescales and the CK test); the reversible estimator assumes detailed balance
(equilibrium sampling); replicates are assumed to sample the same equilibrium
ensemble and are pooled with equal weight per transition; frame weights
(reweighted ensembles) are not supported for kinetics. Times are converted with
``dt_ps`` (frame spacing in ps); lags are in frames.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger("conformatlas.msm")

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


@dataclass
class ClusteringResult:
    """Microstate discretisation of the frames."""

    method: str  # 'kmeans' | 'regular-space'
    centers: np.ndarray  # (k, d)
    labels: np.ndarray  # (n,) microstate index of every frame
    inertia: float  # sum of squared distances to the assigned centre
    n_iter: int = 0
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def n_clusters(self) -> int:
        return int(self.centers.shape[0])


def _sq_dist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (n, k) computed without forming the (n, k, d) tensor."""
    xx = np.einsum("ij,ij->i", X, X)[:, None]
    cc = np.einsum("ij,ij->i", C, C)[None, :]
    d2 = xx + cc - 2.0 * X @ C.T
    np.maximum(d2, 0.0, out=d2)
    return d2


def assign_to_centers(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-centre (Voronoi) assignment of each row of ``X``."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    centers = np.atleast_2d(np.asarray(centers, dtype=np.float64))
    return np.argmin(_sq_dist(X, centers), axis=1)


def _kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]))
    centers[0] = X[rng.integers(n)]
    d2 = _sq_dist(X, centers[:1]).ravel()
    for j in range(1, k):
        total = d2.sum()
        if total <= 0:  # all remaining points coincide with existing centres
            centers[j:] = X[rng.integers(n, size=k - j)]
            break
        idx = rng.choice(n, p=d2 / total)
        centers[j] = X[idx]
        d2 = np.minimum(d2, _sq_dist(X, centers[j : j + 1]).ravel())
    return centers


def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-8,
    n_init: int = 5,
    init: np.ndarray | None = None,
) -> ClusteringResult:
    """k-means (Lloyd's algorithm) with k-means++ seeding and a fixed random seed.

    The objective is the within-cluster sum of squares (the same objective as
    ``scipy.cluster.vq.kmeans2``). ``n_init`` independent seedings are run and
    the lowest-inertia solution is returned. Empty clusters are re-seeded at the
    point farthest from its centre. ``init`` (k, d) bypasses the seeding (useful
    for cross-checks against other implementations).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.ndim != 2:
        raise ValueError("X must be 2-D (frames x components)")
    n, d = X.shape
    k = int(n_clusters)
    if k < 1 or k > n:
        raise ValueError(f"n_clusters must be in [1, {n}] (got {k})")
    rng = np.random.default_rng(seed)
    best: ClusteringResult | None = None
    inits = [np.asarray(init, dtype=np.float64)] if init is not None else None
    n_runs = 1 if inits is not None else max(1, int(n_init))
    for run in range(n_runs):
        centers = inits[0].copy() if inits is not None else _kmeans_plus_plus(X, k, rng)
        if centers.shape != (k, d):
            raise ValueError(f"init must have shape ({k}, {d})")
        labels = assign_to_centers(X, centers)
        inertia = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            new_centers = centers.copy()
            counts = np.bincount(labels, minlength=k)
            sums = np.zeros((k, d))
            np.add.at(sums, labels, X)
            nonempty = counts > 0
            new_centers[nonempty] = sums[nonempty] / counts[nonempty, None]
            if not nonempty.all():
                d2 = _sq_dist(X, new_centers)
                far = np.argsort(-d2[np.arange(n), labels])
                for j, idx in zip(np.where(~nonempty)[0], far):
                    new_centers[j] = X[idx]
            shift = float(np.sum((new_centers - centers) ** 2))
            centers = new_centers
            labels = assign_to_centers(X, centers)
            if shift <= tol:
                break
        d2 = _sq_dist(X, centers)
        inertia = float(d2[np.arange(n), labels].sum())
        if best is None or inertia < best.inertia:
            best = ClusteringResult(
                method="kmeans",
                centers=centers,
                labels=labels.astype(np.int64),
                inertia=inertia,
                n_iter=it,
                parameters={"n_clusters": k, "seed": seed, "n_init": n_runs, "run": run},
            )
    assert best is not None
    return best


def regular_space_cluster(
    X: np.ndarray, dmin: float, max_centers: int = 2000, stride: int = 1
) -> ClusteringResult:
    """Regular-space clustering (Prinz et al. 2011, Sec. II.B).

    Frames are visited in order; a frame becomes a new centre when its distance
    to every existing centre exceeds ``dmin``. Hence all centres are pairwise
    farther apart than ``dmin`` and every (visited) frame lies within ``dmin``
    of some centre. ``stride`` visits every ``stride``-th frame when choosing
    centres (all frames are assigned afterwards). Raises if more than
    ``max_centers`` would be created.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if dmin <= 0:
        raise ValueError("dmin must be positive")
    centers: list[np.ndarray] = []
    dmin2 = dmin * dmin
    for i in range(0, X.shape[0], max(1, int(stride))):
        x = X[i]
        if not centers:
            centers.append(x.copy())
            continue
        C = np.asarray(centers)
        if np.min(np.sum((C - x) ** 2, axis=1)) > dmin2:
            if len(centers) >= max_centers:
                raise ValueError(
                    f"regular-space clustering exceeded max_centers={max_centers}; increase dmin"
                )
            centers.append(x.copy())
    C = np.asarray(centers)
    labels = assign_to_centers(X, C)
    d2 = _sq_dist(X, C)
    inertia = float(d2[np.arange(X.shape[0]), labels].sum())
    return ClusteringResult(
        method="regular-space",
        centers=C,
        labels=labels.astype(np.int64),
        inertia=inertia,
        parameters={"dmin": float(dmin), "stride": int(stride)},
    )


# ---------------------------------------------------------------------------
# Counting and estimation
# ---------------------------------------------------------------------------


def split_discrete_trajectories(
    labels: np.ndarray, replicate_ids: np.ndarray, order: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Split a frame-wise label array into one discrete trajectory per replicate.

    Frames are kept in the order given (or sorted by ``order``, e.g. time_ps,
    within each replicate). Replicate order follows first appearance.
    """
    labels = np.asarray(labels)
    replicate_ids = np.asarray(replicate_ids)
    out: dict[str, np.ndarray] = {}
    for rep in dict.fromkeys(replicate_ids.tolist()):
        idx = np.where(replicate_ids == rep)[0]
        if order is not None:
            idx = idx[np.argsort(np.asarray(order)[idx], kind="stable")]
        out[str(rep)] = labels[idx].astype(np.int64)
    return out


def count_matrix(
    dtrajs: list[np.ndarray] | dict[str, np.ndarray],
    lag: int,
    n_states: int | None = None,
    sliding: bool = True,
) -> np.ndarray:
    """Transition count matrix at lag ``lag`` frames.

    Counts ``C_ij = #{t : s_t = i, s_{t+lag} = j}`` are accumulated separately
    for every discrete trajectory (replicate), so no pair straddles a junction.
    With ``sliding=True`` every ``t`` is used (the counts are then correlated,
    which does not bias the maximum-likelihood transition matrix); otherwise
    only ``t = 0, lag, 2 lag, ...`` are used. Negative labels are ignored.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1 frame")
    trajs = list(dtrajs.values()) if isinstance(dtrajs, dict) else list(dtrajs)
    if n_states is None:
        n_states = int(max((int(t.max()) for t in trajs if len(t)), default=-1)) + 1
    C = np.zeros((n_states, n_states), dtype=np.int64)
    for t in trajs:
        t = np.asarray(t, dtype=np.int64)
        if len(t) <= lag:
            continue
        a = t[:-lag] if sliding else t[: len(t) - lag : lag]
        b = t[lag:] if sliding else t[lag::lag]
        ok = (a >= 0) & (b >= 0)
        np.add.at(C, (a[ok], b[ok]), 1)
    return C


def connected_sets(C: np.ndarray, directed: bool = True) -> list[np.ndarray]:
    """Connected sets of the count graph, largest first (strong connectivity when directed)."""
    C = np.asarray(C)
    n_comp, comp = connected_components(
        csr_matrix(C > 0), directed=directed, connection="strong" if directed else "weak"
    )
    sets = [np.where(comp == c)[0] for c in range(n_comp)]
    sets.sort(key=lambda s: (-C[np.ix_(s, s)].sum(), -len(s), int(s[0])))
    return sets


def largest_connected_set(C: np.ndarray) -> np.ndarray:
    """Indices of the largest strongly connected set (by total counts, then size)."""
    return connected_sets(C, directed=True)[0]


def transition_matrix_nonreversible(C: np.ndarray) -> np.ndarray:
    """Row-normalised maximum-likelihood transition matrix (no detailed-balance constraint)."""
    C = np.asarray(C, dtype=np.float64)
    rows = C.sum(axis=1)
    if np.any(rows <= 0):
        raise ValueError(
            "every state needs at least one outgoing count; restrict to a connected set"
        )
    return C / rows[:, None]


def transition_matrix_reversible(
    C: np.ndarray, max_iter: int = 100_000, tol: float = 1e-12, return_info: bool = False
):
    """Reversible maximum-likelihood transition matrix (Bowman et al. 2009; Prinz et al. 2011).

    Maximises the likelihood ``prod T_ij^{C_ij}`` under detailed balance
    ``pi_i T_ij = pi_j T_ji`` with the fixed-point iteration on the symmetric
    matrix ``X_ij = pi_i T_ij``::

        X_ij <- (C_ij + C_ji) / (C_i / X_i + C_j / X_j),   X_i = sum_j X_ij

    until the largest change in ``T = X / X_i`` is below ``tol``. ``C`` must be
    strongly connected (use :func:`largest_connected_set` first).
    """
    C = np.asarray(C, dtype=np.float64)
    n = C.shape[0]
    if C.shape != (n, n):
        raise ValueError("count matrix must be square")
    if len(connected_sets(C)) != 1:
        raise ValueError(
            "count matrix is not strongly connected; restrict to largest_connected_set"
        )
    Cs = C + C.T
    c_i = C.sum(axis=1)
    X = Cs.copy()
    X *= C.sum() / X.sum()
    T_old = X / X.sum(axis=1)[:, None]
    converged = False
    it = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        for it in range(1, max_iter + 1):
            x_i = X.sum(axis=1)
            denom = (c_i / x_i)[:, None] + (c_i / x_i)[None, :]
            X_new = np.where(Cs > 0, Cs / denom, 0.0)
            X = X_new
            T = X / X.sum(axis=1)[:, None]
            delta = float(np.max(np.abs(T - T_old)))
            T_old = T
            if delta < tol:
                converged = True
                break
    if not converged:
        warnings.warn(
            f"reversible estimator did not converge in {max_iter} iterations (delta={delta:.2e})",
            RuntimeWarning,
            stacklevel=2,
        )
    T = X / X.sum(axis=1)[:, None]
    if return_info:
        return T, {"iterations": it, "converged": converged, "stationary": X.sum(axis=1) / X.sum()}
    return T


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Stationary distribution ``pi T = pi`` from the left eigenvector of eigenvalue 1."""
    T = np.asarray(T, dtype=np.float64)
    vals, vecs = np.linalg.eig(T.T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.real(vecs[:, i])
    pi = pi / pi.sum()
    pi = np.maximum(pi, 0.0)
    return pi / pi.sum()


def is_reversible(T: np.ndarray, pi: np.ndarray | None = None, tol: float = 1e-8) -> bool:
    T = np.asarray(T, dtype=np.float64)
    pi = stationary_distribution(T) if pi is None else np.asarray(pi)
    F = pi[:, None] * T
    return bool(np.max(np.abs(F - F.T)) < tol)


def eigen_decomposition(
    T: np.ndarray, k: int | None = None, reversible: bool | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalues (descending), right and left eigenvectors of ``T``.

    For a reversible ``T`` the symmetric similarity transform ``D^1/2 T D^-1/2``
    (``D = diag(pi)``) is diagonalised, so the spectrum is real by
    construction; otherwise ``numpy.linalg.eig`` is used and the real parts are
    returned with a warning when imaginary parts exceed 1e-8. The right
    eigenvectors have the first column equal to one; the left ones are
    normalised so that ``L^T R = I`` (``L[:, 0] = pi``).
    """
    T = np.asarray(T, dtype=np.float64)
    n = T.shape[0]
    k = n if k is None else min(int(k), n)
    pi = stationary_distribution(T)
    if reversible is None:
        reversible = is_reversible(T, pi)
    if reversible:
        sq = np.sqrt(pi)
        S = (sq[:, None] * T) / sq[None, :]
        S = 0.5 * (S + S.T)
        vals, V = np.linalg.eigh(S)
        order = np.argsort(-vals)
        vals, V = vals[order], V[:, order]
        R = V / sq[:, None]
        L = V * sq[:, None]
    else:
        vals, R = np.linalg.eig(T)
        order = np.argsort(-np.abs(vals))
        vals, R = vals[order], R[:, order]
        if np.max(np.abs(np.imag(vals))) > 1e-8:
            warnings.warn(
                "non-reversible transition matrix has complex eigenvalues; real parts are reported",
                RuntimeWarning,
                stacklevel=2,
            )
        vals, R = np.real(vals), np.real(R)
        L = np.linalg.inv(R).T
    # normalise: first right eigenvector = 1, L^T R = I
    scale = R[0, 0] if abs(R[0, 0]) > 1e-12 else 1.0
    R = R / scale
    L = L * scale
    for j in range(n):
        s = float(L[:, j] @ R[:, j])
        if abs(s) > 1e-14:
            L[:, j] /= s
    if R[:, 0].mean() < 0:
        R[:, 0] *= -1
        L[:, 0] *= -1
    return vals[:k], R[:, :k], L[:, :k]


def implied_timescales(eigenvalues: np.ndarray, lag: int, dt_ps: float = 1.0) -> np.ndarray:
    """``t_i = -lag * dt / ln |lambda_i|`` for the non-unit eigenvalues (NaN for lambda <= 0)."""
    lam = np.asarray(eigenvalues, dtype=np.float64)[1:]
    out = np.full(lam.shape, np.nan)
    ok = (lam > 0) & (lam < 1)
    out[ok] = -lag * dt_ps / np.log(lam[ok])
    out[lam >= 1] = np.inf
    return out


@dataclass
class MSMEstimate:
    """A transition matrix estimated at one lag."""

    lag: int
    dt_ps: float
    reversible: bool
    count_matrix: np.ndarray  # full (n_micro x n_micro)
    active_set: np.ndarray  # microstate indices retained
    transition_matrix: np.ndarray  # on the active set
    stationary: np.ndarray  # on the active set
    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    left_eigenvectors: np.ndarray
    timescales_ps: np.ndarray
    estimator_info: dict[str, Any] = field(default_factory=dict)

    @property
    def n_active(self) -> int:
        return int(len(self.active_set))

    @property
    def active_fraction_of_counts(self) -> float:
        tot = self.count_matrix.sum()
        return (
            float(self.count_matrix[np.ix_(self.active_set, self.active_set)].sum() / tot)
            if tot
            else 0.0
        )


def estimate_msm(
    dtrajs: list[np.ndarray] | dict[str, np.ndarray],
    lag: int,
    n_states: int | None = None,
    reversible: bool = True,
    dt_ps: float = 1.0,
    n_eigen: int | None = None,
) -> MSMEstimate:
    """Count at ``lag``, restrict to the largest strongly connected set, estimate ``T``.

    Assumes Markovianity at ``lag`` and (if ``reversible``) detailed balance.
    """
    C = count_matrix(dtrajs, lag, n_states=n_states, sliding=True)
    active = largest_connected_set(C)
    if len(active) < 2:
        raise ValueError("fewer than two microstates in the largest connected set at this lag")
    Ca = C[np.ix_(active, active)]
    info: dict[str, Any] = {}
    if reversible:
        T, info = transition_matrix_reversible(Ca, return_info=True)
        pi = np.asarray(info.pop("stationary"))
    else:
        T = transition_matrix_nonreversible(Ca)
        pi = stationary_distribution(T)
    vals, R, L = eigen_decomposition(T, k=n_eigen, reversible=reversible)
    ts = implied_timescales(vals, lag, dt_ps)
    return MSMEstimate(
        lag=int(lag),
        dt_ps=float(dt_ps),
        reversible=bool(reversible),
        count_matrix=C,
        active_set=active,
        transition_matrix=T,
        stationary=pi,
        eigenvalues=vals,
        right_eigenvectors=R,
        left_eigenvectors=L,
        timescales_ps=ts,
        estimator_info=info,
    )


def implied_timescales_scan(
    dtrajs: list[np.ndarray] | dict[str, np.ndarray],
    lags: list[int] | np.ndarray,
    n_timescales: int = 5,
    reversible: bool = True,
    dt_ps: float = 1.0,
    n_states: int | None = None,
) -> pd.DataFrame:
    """Implied timescales (ps) for every lag in ``lags``.

    Columns: ``lag_frames, lag_ps, n_active, active_count_fraction, ts_1 ... ts_k``.
    Timescales shorter than the lag itself are statistically unreliable and
    are kept but flagged downstream by the plot (shaded region).
    """
    rows = []
    for lag in lags:
        try:
            est = estimate_msm(
                dtrajs, int(lag), n_states=n_states, reversible=reversible, dt_ps=dt_ps
            )
        except ValueError as exc:
            logger.warning(f"lag {lag}: {exc}")
            continue
        ts = est.timescales_ps
        row = {
            "lag_frames": int(lag),
            "lag_ps": float(lag * dt_ps),
            "n_active": est.n_active,
            "active_count_fraction": est.active_fraction_of_counts,
        }
        for i in range(n_timescales):
            row[f"ts_{i + 1}"] = float(ts[i]) if i < len(ts) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def suggest_markov_lag(
    its: pd.DataFrame, rel_tol: float = 0.10, n_timescales: int | None = None
) -> dict[str, Any]:
    """Report the first lag from which the slow implied timescales are flat.

    A lag ``tau_k`` is a plateau candidate when, for every considered
    timescale, the relative change to the next scanned lag is below
    ``rel_tol`` and the timescale exceeds the lag (otherwise it cannot be
    resolved). The function *reports* (``suggested_lag_frames`` is None when no
    plateau is found); it never changes the model lag. This is a heuristic
    (UNVERIFIED beyond its definition) - inspect the plot.
    """
    cols = [c for c in its.columns if c.startswith("ts_")]
    if n_timescales is not None:
        cols = cols[: int(n_timescales)]
    lags = its["lag_frames"].to_numpy()
    rel = []
    for i in range(len(its) - 1):
        a = its.iloc[i][cols].to_numpy(dtype=float)
        b = its.iloc[i + 1][cols].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(b - a) / np.abs(a)
        resolved = np.asarray(a > its.iloc[i]["lag_ps"]) & np.isfinite(r)
        rel.append(
            {
                "lag_frames": int(lags[i]),
                "next_lag_frames": int(lags[i + 1]),
                "max_rel_change": float(np.nanmax(np.where(resolved, r, np.nan)))
                if resolved.any()
                else np.nan,
                "n_resolved": int(resolved.sum()),
            }
        )
    rel_df = pd.DataFrame(rel)
    suggested = None
    for _, row in rel_df.iterrows():
        if np.isfinite(row["max_rel_change"]) and row["max_rel_change"] < rel_tol:
            suggested = int(row["lag_frames"])
            break
    note = (
        f"Implied timescales change by < {rel_tol * 100:.0f} % between lag {suggested} "
        f"and the next scanned lag; a lag of at least {suggested} frames is consistent with "
        "Markovian behaviour for the resolved timescales. Confirm with the CK test."
        if suggested is not None
        else (
            f"No plateau found (relative change never < {rel_tol * 100:.0f} %); the "
            "discretisation may be too coarse or the trajectories too short for an MSM."
        )
    )
    return {
        "suggested_lag_frames": suggested,
        "plateau_found": suggested is not None,
        "rel_tol": rel_tol,
        "relative_changes": rel_df,
        "note": note,
    }


# ---------------------------------------------------------------------------
# PCCA+
# ---------------------------------------------------------------------------


def _pcca_isa(X: np.ndarray) -> np.ndarray:
    """Inner simplex algorithm: vertices of the eigenvector simplex -> initial A (m x m)."""
    n, m = X.shape
    index = np.zeros(m, dtype=int)
    ortho = X.copy()
    index[0] = int(np.argmax(np.linalg.norm(ortho, axis=1)))
    ortho = ortho - ortho[index[0]]
    for j in range(1, m):
        v = ortho[index[j - 1]].copy()
        nv = np.linalg.norm(v)
        if nv > 0:
            v /= nv
            ortho = ortho - np.outer(ortho @ v, v)
        dist = np.linalg.norm(ortho, axis=1)
        dist[index[:j]] = -1.0
        index[j] = int(np.argmax(dist))
    return np.linalg.inv(X[index])


def _pcca_fill(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Impose the feasibility constraints on A given the free block A[1:, 1:]."""
    A = A.copy()
    A[1:, 0] = -np.sum(A[1:, 1:], axis=1)
    A[0, :] = -np.min(X[:, 1:] @ A[1:, :], axis=0)
    A /= np.sum(A[0, :])
    return A


def _pcca_objective(alpha: np.ndarray, X: np.ndarray) -> float:
    m = X.shape[1]
    A = np.zeros((m, m))
    A[1:, :] = alpha.reshape(m - 1, m)
    A = _pcca_fill(A, X)
    return -float(np.sum(A**2 / A[0, :][None, :]))


def pcca_plus(
    T: np.ndarray, n_metastable: int, pi: np.ndarray | None = None, optimize: bool = True
) -> np.ndarray:
    """PCCA+ membership matrix chi (n_micro x n_metastable).

    Robust Perron cluster analysis (Deuflhard & Weber, Linear Algebra Appl.
    2005, 398, 161; Roeblitz & Weber, Adv. Data Anal. Classif. 2013, 7, 147):
    the first ``m`` right eigenvectors span a simplex whose vertices are found
    by the inner simplex algorithm; the linear map ``A`` is then refined by
    maximising the crispness ``sum_ij A_ij^2 / A_1j`` under ``chi >= 0``,
    ``chi 1 = 1`` (Nelder-Mead, as in deeptime). Memberships are non-negative
    and rows sum to one. Requires a transition matrix on a connected set;
    for a non-reversible ``T`` the real parts of the eigenvectors are used.
    """
    T = np.asarray(T, dtype=np.float64)
    n = T.shape[0]
    m = int(n_metastable)
    if m < 2 or m > n:
        raise ValueError(f"n_metastable must be in [2, {n}]")
    if m == n:
        return np.eye(n)
    vals, R, _ = eigen_decomposition(T, k=m)
    if np.any(np.abs(vals[1:] - 1.0) < 1e-10):
        raise ValueError("transition matrix is reducible (several unit eigenvalues)")
    X = R.copy()
    X[:, 0] = 1.0
    A = _pcca_isa(X)
    if optimize:
        A0 = _pcca_fill(A, X)
        alpha0 = A0[1:, :].ravel()
        res = minimize(
            _pcca_objective,
            alpha0,
            args=(X,),
            method="Nelder-Mead",
            options={"maxiter": 2000 * (m - 1) * m, "xatol": 1e-8, "fatol": 1e-10},
        )
        if res.fun <= _pcca_objective(alpha0, X):
            A = np.zeros((m, m))
            A[1:, :] = res.x.reshape(m - 1, m)
            A = _pcca_fill(A, X)
        else:
            A = A0
    chi = X @ A
    chi = np.clip(chi, 0.0, None)
    chi /= chi.sum(axis=1)[:, None]
    return chi


def crisp_assignment(memberships: np.ndarray) -> np.ndarray:
    """Metastable index with the largest membership for every microstate."""
    return np.argmax(np.asarray(memberships), axis=1)


def coarse_grain(T: np.ndarray, pi: np.ndarray, chi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCCA+ coarse-grained transition matrix and stationary distribution.

    Projection of Kube & Weber (J. Chem. Phys. 2007, 126, 024103) as used by
    deeptime / PyEMMA: ``T_c = (chi^T chi)^-1 chi^T T chi`` with
    ``pi_c = chi^T pi``; the flux matrix ``diag(pi_c) T_c`` is symmetrised to
    remove numerical asymmetry and renormalised row-wise. The coarse matrix is a
    descriptive summary; rates between metastable sets should be read from the
    MFPTs or the implied timescales of the microstate model.
    """
    pi_c = chi.T @ pi
    W = np.linalg.inv(chi.T @ chi)
    T_c = W @ (chi.T @ T @ chi)
    X = pi_c[:, None] * T_c
    X = 0.5 * (X + X.T)
    pi_c = X.sum(axis=1)
    T_c = X / pi_c[:, None]
    return T_c, pi_c


# ---------------------------------------------------------------------------
# Chapman-Kolmogorov test and MFPT
# ---------------------------------------------------------------------------


def _set_probability(T: np.ndarray, pi: np.ndarray, sets: list[np.ndarray], n_active: int):
    """P(in set B at time k*tau | started in set A, equilibrium within A) for all A, B."""
    m = len(sets)
    out = np.zeros((m, m))
    for a, A in enumerate(sets):
        w = np.zeros(n_active)
        w[A] = pi[A]
        if w.sum() <= 0:
            out[a] = np.nan
            continue
        w /= w.sum()
        p = w @ T
        for b, B in enumerate(sets):
            out[a, b] = p[B].sum()
    return out


def chapman_kolmogorov_test(
    dtrajs: list[np.ndarray] | dict[str, np.ndarray],
    lag: int,
    sets: list[np.ndarray],
    k_max: int = 5,
    reversible: bool = True,
    n_states: int | None = None,
    dt_ps: float = 1.0,
    abs_tol: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chapman-Kolmogorov test on metastable sets (Prinz et al. 2011, Sec. V.C).

    For ``k = 1 .. k_max`` the predicted set-to-set probabilities from the model
    at ``lag`` (``T(tau)^k``, started from the stationary distribution restricted
    to the origin set) are compared with the same quantity computed from a
    transition matrix estimated directly at ``k * lag``. ``sets`` are lists of
    microstate indices in the *full* microstate numbering; both models are
    restricted to their own largest connected set and the sets are intersected
    with it. The approximate standard error of the estimate uses the number
    of independent transition pairs ``N_A / (k lag)`` (sliding counts are
    correlated) in a binomial formula. The verdict is ``pass`` when every
    absolute deviation is below ``max(abs_tol, 3 * se)``.
    """
    ref = estimate_msm(dtrajs, lag, n_states=n_states, reversible=reversible, dt_ps=dt_ps)
    n_states = ref.count_matrix.shape[0]
    rows = []
    for k in range(1, k_max + 1):
        try:
            est = estimate_msm(
                dtrajs, lag * k, n_states=n_states, reversible=reversible, dt_ps=dt_ps
            )
        except ValueError as exc:
            logger.warning(f"CK test: lag {lag * k}: {exc}")
            continue
        # predicted from the reference model
        pos_ref = {s: i for i, s in enumerate(ref.active_set.tolist())}
        sets_ref = [np.array([pos_ref[s] for s in S if s in pos_ref], dtype=int) for S in sets]
        Tk = np.linalg.matrix_power(ref.transition_matrix, k)
        pred = _set_probability(Tk, ref.stationary, sets_ref, ref.n_active)
        # estimated directly at k*lag
        pos_est = {s: i for i, s in enumerate(est.active_set.tolist())}
        sets_est = [np.array([pos_est[s] for s in S if s in pos_est], dtype=int) for S in sets]
        obs = _set_probability(est.transition_matrix, est.stationary, sets_est, est.n_active)
        Ck = est.count_matrix
        for a in range(len(sets)):
            n_a = Ck[sets[a], :].sum()
            n_indep = max(n_a / (lag * k), 1.0)
            for b in range(len(sets)):
                p = obs[a, b]
                se = float(np.sqrt(max(p * (1 - p), 0.0) / n_indep)) if np.isfinite(p) else np.nan
                rows.append(
                    {
                        "k": k,
                        "lag_frames": lag * k,
                        "lag_ps": lag * k * dt_ps,
                        "from_set": a,
                        "to_set": b,
                        "predicted": float(pred[a, b]),
                        "estimated": float(p),
                        "estimated_se": se,
                        "abs_deviation": float(abs(pred[a, b] - p)),
                    }
                )
    df = pd.DataFrame(rows)
    if len(df):
        thr = np.maximum(abs_tol, 3.0 * df["estimated_se"].fillna(0.0))
        df["within_tolerance"] = df["abs_deviation"] <= thr
        diag = df[df["from_set"] == df["to_set"]]
        summary = {
            "lag_frames": int(lag),
            "k_max": int(k_max),
            "abs_tol": float(abs_tol),
            "max_abs_deviation": float(df["abs_deviation"].max()),
            "max_abs_deviation_diagonal": float(diag["abs_deviation"].max()),
            "n_violations": int((~df["within_tolerance"]).sum()),
            "passed": bool(df["within_tolerance"].all()),
        }
    else:
        summary = {"lag_frames": int(lag), "k_max": int(k_max), "passed": False, "n_violations": 0}
    return df, summary


def mfpt_between_sets(
    T: np.ndarray, pi: np.ndarray, sets: list[np.ndarray], lag: int, dt_ps: float = 1.0
) -> np.ndarray:
    """Mean first-passage times (ps) from every set to every other set.

    Solves ``m_i = tau + sum_j T_ij m_j`` for ``i`` outside the target set with
    ``m_i = 0`` inside, then averages over the origin set with weights ``pi``
    (the standard committor-free MFPT, e.g. deeptime ``mfpt``). Requires the
    Markov assumption at ``lag``; the result is in units of ``lag * dt_ps``.
    """
    T = np.asarray(T, dtype=np.float64)
    n = T.shape[0]
    m = len(sets)
    out = np.zeros((m, m))
    for b, B in enumerate(sets):
        if len(B) == 0:
            out[:, b] = np.nan
            continue
        keep = np.setdiff1d(np.arange(n), B)
        mvec = np.zeros(n)
        if len(keep):
            Asub = np.eye(len(keep)) - T[np.ix_(keep, keep)]
            mvec[keep] = np.linalg.solve(Asub, np.full(len(keep), float(lag)))
        for a, A in enumerate(sets):
            if a == b:
                out[a, b] = 0.0
                continue
            w = pi[A]
            out[a, b] = float((w @ mvec[A]) / w.sum()) * dt_ps if w.sum() > 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Landscape / basin overlap
# ---------------------------------------------------------------------------


def basin_overlap_table(
    metastable_labels: pd.Series | np.ndarray, basin_labels: pd.Series | np.ndarray
) -> pd.DataFrame:
    """Frame-count cross-table of PCCA+ metastable sets versus watershed basins.

    Rows are metastable sets, columns watershed states (``frame_assignments['state']``,
    including 'Unassigned'); an additional ``row_fraction_max`` column gives the
    largest fraction of a set's frames falling in one basin and ``best_basin``
    the basin it falls in, so a value near 1 means the kinetic set coincides
    with one landscape basin.
    """
    ct = pd.crosstab(
        pd.Series(np.asarray(metastable_labels), name="metastable_set"),
        pd.Series(np.asarray(basin_labels), name="basin_state"),
    )
    frac = ct.div(ct.sum(axis=1), axis=0)
    ct["best_basin"] = frac.idxmax(axis=1)
    ct["row_fraction_max"] = frac.max(axis=1)
    return ct


# ---------------------------------------------------------------------------
# Full workflow
# ---------------------------------------------------------------------------


@dataclass
class MSMResult:
    clustering: ClusteringResult
    columns: list[str]
    lag: int
    dt_ps: float
    reversible: bool
    msm: MSMEstimate
    implied_timescales: pd.DataFrame
    lag_report: dict[str, Any]
    n_metastable: int
    memberships: pd.DataFrame  # index microstate (active), columns set_0..
    metastable_sets: list[np.ndarray]  # full microstate indices per set
    frame_table: pd.DataFrame  # per frame: replicate, frame, time_ps, microstate, metastable_set
    ck_test: pd.DataFrame
    ck_summary: dict[str, Any]
    mfpt_ps: pd.DataFrame
    coarse_transition_matrix: pd.DataFrame
    coarse_stationary: pd.DataFrame
    basin_overlap: pd.DataFrame | None
    notes: list[str] = field(default_factory=list)


def _metastable_set_names(m: int) -> list[str]:
    return [f"MS{i + 1}" for i in range(m)]


def build_msm(
    projections: pd.DataFrame,
    columns: list[str],
    lag: int,
    lags: list[int] | None = None,
    n_clusters: int = 100,
    clustering: str = "kmeans",
    dmin: float | None = None,
    n_metastable: int = 2,
    reversible: bool = True,
    dt_ps: float | None = None,
    n_timescales: int = 5,
    ck_k_max: int = 5,
    seed: int = 0,
    basin_states: pd.Series | np.ndarray | None = None,
) -> MSMResult:
    """Cluster -> count -> estimate -> ITS scan -> PCCA+ -> CK -> MFPT -> basin overlap.

    ``projections`` needs columns ``replicate, frame, time_ps`` plus ``columns``
    (e.g. ``['PC1', 'PC2', 'PC3']``); frames are ordered by ``time_ps`` within
    each replicate. ``basin_states`` (per frame, aligned with ``projections``)
    enables the overlap table with the watershed states.
    """
    for c in ("replicate", "frame", "time_ps", *columns):
        if c not in projections.columns:
            raise ValueError(f"projections lacks column '{c}'")
    if dt_ps is None:
        t = np.unique(projections["time_ps"].to_numpy(dtype=float))
        dt_ps = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    X = projections[list(columns)].to_numpy(dtype=np.float64)
    if clustering == "kmeans":
        clus = kmeans_cluster(X, n_clusters, seed=seed)
    elif clustering == "regular-space":
        if dmin is None:
            raise ValueError("dmin is required for regular-space clustering")
        clus = regular_space_cluster(X, dmin)
    else:
        raise ValueError("clustering must be 'kmeans' or 'regular-space'")
    n_micro = clus.n_clusters
    dtrajs = split_discrete_trajectories(
        clus.labels, projections["replicate"].to_numpy(), order=projections["time_ps"].to_numpy()
    )
    if lags is None:
        lags = sorted({int(v) for v in np.unique(np.geomspace(1, max(lag * 5, 2), 10).round())})
    its = implied_timescales_scan(
        dtrajs,
        lags,
        n_timescales=n_timescales,
        reversible=reversible,
        dt_ps=dt_ps,
        n_states=n_micro,
    )
    report = suggest_markov_lag(its)
    msm = estimate_msm(dtrajs, lag, n_states=n_micro, reversible=reversible, dt_ps=dt_ps)
    notes = []
    if msm.n_active < n_micro:
        notes.append(
            f"{n_micro - msm.n_active} of {n_micro} microstates are outside the largest connected set "
            f"at lag {lag} ({100 * (1 - msm.active_fraction_of_counts):.2f} % of counts dropped)."
        )
    m = int(min(n_metastable, msm.n_active))
    chi = pcca_plus(msm.transition_matrix, m)
    crisp = crisp_assignment(chi)
    names = _metastable_set_names(m)
    memberships = pd.DataFrame(
        chi, index=pd.Index(msm.active_set, name="microstate"), columns=names
    )
    memberships.insert(0, "crisp_set", [names[i] for i in crisp])
    sets = [msm.active_set[crisp == i] for i in range(m)]
    # per-frame table
    micro_to_set = np.full(n_micro, -1, dtype=int)
    micro_to_set[msm.active_set] = crisp
    frame_table = projections[["replicate", "frame", "time_ps"]].copy()
    frame_table["microstate"] = clus.labels
    ms = micro_to_set[clus.labels]
    frame_table["metastable_set"] = np.where(
        ms >= 0, np.array(names + ["inactive"])[ms], "inactive"
    )
    micro_pos = np.full(n_micro, -1, dtype=int)
    micro_pos[msm.active_set] = np.arange(msm.n_active)
    pos_frame = micro_pos[clus.labels]
    ok = pos_frame >= 0
    for i in range(m):
        col = np.full(len(frame_table), np.nan)
        col[ok] = chi[pos_frame[ok], i]
        frame_table[f"membership_{names[i]}"] = col
    ck_df, ck_summary = chapman_kolmogorov_test(
        dtrajs, lag, sets, k_max=ck_k_max, reversible=reversible, n_states=n_micro, dt_ps=dt_ps
    )
    pos = {s: i for i, s in enumerate(msm.active_set.tolist())}
    sets_active = [np.array([pos[s] for s in S], dtype=int) for S in sets]
    mf = mfpt_between_sets(msm.transition_matrix, msm.stationary, sets_active, lag, dt_ps)
    mfpt_df = pd.DataFrame(
        mf, index=pd.Index(names, name="from_set"), columns=[f"to_{n}" for n in names]
    )
    Tc, pic = coarse_grain(msm.transition_matrix, msm.stationary, chi)
    coarse_T = pd.DataFrame(Tc, index=pd.Index(names, name="from_set"), columns=names)
    coarse_pi = pd.DataFrame({"metastable_set": names, "stationary_probability": pic})
    overlap = None
    if basin_states is not None:
        overlap = basin_overlap_table(
            frame_table["metastable_set"].to_numpy(), np.asarray(basin_states)
        )
    notes.append(
        f"MSM at lag {lag} frames ({lag * dt_ps:g} ps), {clus.method} with {n_micro} microstates, "
        f"{'reversible' if reversible else 'non-reversible'} estimator, {m} PCCA+ sets; "
        f"CK test {'passed' if ck_summary.get('passed') else 'FAILED'} "
        f"(max |dev| {ck_summary.get('max_abs_deviation', np.nan):.3f}). " + report["note"]
    )
    return MSMResult(
        clustering=clus,
        columns=list(columns),
        lag=int(lag),
        dt_ps=float(dt_ps),
        reversible=bool(reversible),
        msm=msm,
        implied_timescales=its,
        lag_report=report,
        n_metastable=m,
        memberships=memberships,
        metastable_sets=sets,
        frame_table=frame_table,
        ck_test=ck_df,
        ck_summary=ck_summary,
        mfpt_ps=mfpt_df,
        coarse_transition_matrix=coarse_T,
        coarse_stationary=coarse_pi,
        basin_overlap=overlap,
        notes=notes,
    )


def save_msm(result: MSMResult, output_dir: str | Path) -> Path:
    """Write the MSM outputs to ``<output_dir>/msm/``."""
    out = Path(output_dir) / "msm"
    out.mkdir(parents=True, exist_ok=True)
    est = result.msm
    active = est.active_set
    pd.DataFrame(est.count_matrix).to_csv(out / "count_matrix.csv")
    pd.DataFrame(est.transition_matrix, index=active, columns=active).rename_axis(
        "microstate"
    ).to_csv(out / "transition_matrix.csv")
    pd.DataFrame({"microstate": active, "stationary_probability": est.stationary}).to_csv(
        out / "stationary.csv", index=False
    )
    pd.DataFrame(
        result.clustering.centers,
        columns=result.columns,
    ).rename_axis("microstate").to_csv(out / "cluster_centers.csv")
    result.implied_timescales.to_csv(out / "implied_timescales.csv", index=False)
    result.lag_report["relative_changes"].to_csv(out / "implied_timescale_changes.csv", index=False)
    result.ck_test.to_csv(out / "ck_test.csv", index=False)
    result.memberships.to_csv(out / "pcca_memberships.csv")
    result.frame_table.to_csv(out / "metastable_assignments.csv", index=False)
    result.mfpt_ps.to_csv(out / "mfpt.csv")
    result.coarse_transition_matrix.to_csv(out / "coarse_transition_matrix.csv")
    result.coarse_stationary.to_csv(out / "coarse_stationary.csv", index=False)
    if result.basin_overlap is not None:
        result.basin_overlap.to_csv(out / "basin_overlap.csv")
    pd.DataFrame(
        {
            "index": np.arange(len(est.eigenvalues)),
            "eigenvalue": est.eigenvalues,
            "timescale_ps": np.concatenate([[np.inf], est.timescales_ps]),
        }
    ).to_csv(out / "eigenvalues.csv", index=False)
    summary = {
        "lag_frames": result.lag,
        "dt_ps": result.dt_ps,
        "reversible": result.reversible,
        "clustering": result.clustering.method,
        "n_microstates": result.clustering.n_clusters,
        "n_active": est.n_active,
        "active_count_fraction": est.active_fraction_of_counts,
        "n_metastable": result.n_metastable,
        "suggested_lag_frames": result.lag_report["suggested_lag_frames"],
        "plateau_found": result.lag_report["plateau_found"],
        "ck_test": result.ck_summary,
        "estimator": est.estimator_info,
        "notes": result.notes,
    }
    from conformatlas.utils import save_json

    save_json(summary, out / "msm_summary.json")
    return out
