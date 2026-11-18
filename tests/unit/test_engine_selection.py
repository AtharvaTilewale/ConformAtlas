"""Memory estimate, available-memory detection and the --engine auto decision."""

from dataclasses import replace
from pathlib import Path

import pytest

from conformatlas.config import build_analysis_config
from conformatlas.models import AnalysisConfig, TrajectoryInfo
from conformatlas.pipeline import _frames_in_window, select_engine
from conformatlas.trajectory import (
    FALLBACK_STREAMING_THRESHOLD_GB,
    _cgroup_available_bytes,
    _meminfo_available_bytes,
    available_memory_gb,
    estimate_peak_memory_gb,
    estimate_streaming_peak_gb,
    suggest_stride,
)

# Measured peak RSS (GB) of in-memory runs, full-atom adenylate kinase: Backbone
# (642), C-alpha (214), and the first 1 500 / 2 500 atoms (covariance and SVD routes).
MEASURED = [
    (5_000, 642, 0.842),
    (10_000, 642, 1.022),
    (20_000, 642, 1.690),
    (40_000, 642, 3.019),
    (20_000, 214, 0.638),
    (80_000, 214, 2.095),
    (8_000, 1_500, 2.882),
    (3_000, 2_500, 2.834),
]
# Streaming engine: (frames, atoms, replicates, measured GB).
MEASURED_STREAMING = [
    (3_000, 1_500, 1, 1.115),
    (3_000, 2_500, 1, 3.016),
    (3_000, 2_500, 2, 3.469),
]


def _config(**kw) -> AnalysisConfig:
    base = AnalysisConfig(topology=Path("top.pdb"), trajectories=[Path("run.xtc")])
    return replace(base, **kw)


# ---------------------------------------------------------------- estimate


@pytest.mark.parametrize("n_frames,n_atoms,measured", MEASURED)
def test_peak_estimate_is_conservative(n_frames, n_atoms, measured):
    est = estimate_peak_memory_gb(n_frames, n_atoms)
    assert est >= measured
    assert est <= measured + max(0.45, 0.25 * measured)  # not wildly pessimistic


@pytest.mark.parametrize("n_frames,n_atoms,n_reps,measured", MEASURED_STREAMING)
def test_streaming_estimate_is_conservative(n_frames, n_atoms, n_reps, measured):
    est = estimate_streaming_peak_gb(n_frames, n_atoms, n_reps)
    assert measured <= est <= measured + max(0.45, 0.25 * measured)


def test_streaming_estimate_grows_with_replicates_workers_and_atoms():
    base = estimate_streaming_peak_gb(1_000, 2_000, 1, 1)
    assert estimate_streaming_peak_gb(1_000, 2_000, 2, 1) > base
    assert estimate_streaming_peak_gb(1_000, 2_000, 1, 2) > base
    assert estimate_streaming_peak_gb(1_000, 2_500, 1, 1) > base


def test_peak_estimate_grows_with_frames_and_atoms():
    assert estimate_peak_memory_gb(2_000, 642) > estimate_peak_memory_gb(1_000, 642)
    assert estimate_peak_memory_gb(1_000, 700) > estimate_peak_memory_gb(1_000, 600)


def test_suggest_stride_fits_the_limit():
    n, atoms, limit = 100_000, 642, 3.0
    s = suggest_stride(n, atoms, limit)
    assert estimate_peak_memory_gb(n // s, atoms) <= limit
    assert estimate_peak_memory_gb(n // (s - 1), atoms) > limit
    assert suggest_stride(1_000, 642, 50.0) == 1
    # No stride helps when even one frame of the selection does not fit.
    assert suggest_stride(1_000, 40_000, 0.5) is None


# ------------------------------------------------------- available memory


def _proc(tmp_path, text=""):
    p = tmp_path / "proc_self_cgroup"
    p.write_text(text)
    return p


def test_cgroup_v2_limit_minus_usage(tmp_path):
    (tmp_path / "memory.max").write_text("4294967296\n")
    (tmp_path / "memory.current").write_text("1073741824\n")
    assert _cgroup_available_bytes(tmp_path, _proc(tmp_path, "0::/\n")) == 3 * 1024**3


def test_cgroup_v2_without_limit(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    assert _cgroup_available_bytes(tmp_path, _proc(tmp_path, "0::/\n")) is None


def test_cgroup_v1_limit_and_unlimited(tmp_path):
    v1 = tmp_path / "memory"
    v1.mkdir()
    (v1 / "memory.limit_in_bytes").write_text("2147483648\n")
    (v1 / "memory.usage_in_bytes").write_text("536870912\n")
    proc = _proc(tmp_path, "4:memory:/\n")
    assert _cgroup_available_bytes(tmp_path, proc) == 1536 * 1024**2
    (v1 / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    assert _cgroup_available_bytes(tmp_path, proc) is None


def test_cgroup_nested_v1_uses_tightest_limit_on_own_path(tmp_path):
    # Process in a nested cgroup (not the root of the mount): the limit is on its own
    # path, the root is unlimited.
    root = tmp_path / "memory"
    child = root / "jobs" / "job1"
    child.mkdir(parents=True)
    (root / "memory.limit_in_bytes").write_text("9223372036854771712\n")
    (root / "jobs" / "memory.limit_in_bytes").write_text(str(8 * 10**9))
    (root / "jobs" / "memory.usage_in_bytes").write_text("0")
    (child / "memory.limit_in_bytes").write_text(str(4 * 10**9))
    (child / "memory.usage_in_bytes").write_text(str(10**9))
    proc = _proc(tmp_path, "9:cpu,cpuacct:/\n4:memory:/jobs/job1\n0::/\n")
    assert _cgroup_available_bytes(tmp_path, proc) == 3 * 10**9


def test_cgroup_nested_v2_parent_limit_applies(tmp_path):
    child = tmp_path / "user.slice" / "app"
    child.mkdir(parents=True)
    (tmp_path / "user.slice" / "memory.max").write_text(str(2 * 10**9))
    (tmp_path / "user.slice" / "memory.current").write_text(str(5 * 10**8))
    (child / "memory.max").write_text("max")
    proc = _proc(tmp_path, "0::/user.slice/app\n")
    assert _cgroup_available_bytes(tmp_path, proc) == 15 * 10**8


def test_meminfo_parsing(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal:  16000000 kB\nMemFree: 1000 kB\nMemAvailable:   8000000 kB\n")
    assert _meminfo_available_bytes(mi) == 8_000_000 * 1024
    assert _meminfo_available_bytes(tmp_path / "missing") is None


def test_available_memory_takes_the_smaller(tmp_path):
    (tmp_path / "memory.max").write_text(str(4 * 10**9))
    (tmp_path / "memory.current").write_text("0")
    proc = _proc(tmp_path, "0::/\n")
    mi = tmp_path / "meminfo"
    mi.write_text("MemAvailable: 16000000 kB\n")
    assert available_memory_gb(tmp_path, mi, proc) == pytest.approx(4.0)
    mi.write_text("MemAvailable: 2000000 kB\n")
    assert available_memory_gb(tmp_path, mi, proc) == pytest.approx(2.048)
    none = tmp_path / "none"
    assert available_memory_gb(none, none, none) is None


# ------------------------------------------------------------ engine choice


def test_auto_small_run_stays_in_memory():
    d = select_engine(_config(), 5_000, 642, available_gb=4.0)
    assert not d.use_streaming and d.note is None and d.warning is None


def test_auto_streams_when_the_container_is_too_small():
    # 100 000 frames x Backbone (642 atoms): killed in memory, must stream in 4 GB.
    d = select_engine(_config(), 100_000, 642, available_gb=4.0)
    assert d.use_streaming
    assert d.limit_gb == pytest.approx(3.0)
    assert "streaming engine" in d.note and "4.0 GB available" in d.note
    assert d.warning is None


def test_auto_keeps_memory_engine_on_a_large_machine():
    d = select_engine(_config(), 100_000, 642, available_gb=64.0)
    assert not d.use_streaming and d.warning is None


def test_explicit_threshold_overrides_available_memory():
    d = select_engine(_config(streaming_threshold_gb=1.5), 20_000, 642, available_gb=64.0)
    assert d.use_streaming and "streaming_threshold_gb = 1.5 GB" in d.note
    d = select_engine(_config(streaming_threshold_gb=100.0), 100_000, 642, available_gb=4.0)
    assert not d.use_streaming
    assert "may be killed" in d.warning


def test_unknown_available_memory_uses_fallback():
    d = select_engine(_config(), 100_000, 642, available_gb=None)
    assert d.limit_gb == FALLBACK_STREAMING_THRESHOLD_GB
    assert not d.use_streaming  # 7.5 GB estimate < 8 GB fallback
    d = select_engine(_config(), 200_000, 642, available_gb=None)
    assert d.use_streaming and "available memory unknown" in d.note


def test_options_that_need_memory_engine_warn_instead_of_streaming():
    d = select_engine(_config(pca_type="dihedral"), 100_000, 642, available_gb=4.0)
    assert not d.use_streaming
    assert "--pca-type dihedral" in d.warning and "--stride" in d.warning
    assert "--engine streaming" not in d.warning
    d = select_engine(_config(residue_analysis=True), 100_000, 642, available_gb=4.0)
    assert "--residue-analysis" in d.warning


def test_forced_memory_engine_warns_and_suggests_streaming():
    d = select_engine(_config(engine="memory"), 100_000, 642, available_gb=4.0)
    assert not d.use_streaming and "--engine streaming" in d.warning


def test_forced_streaming_engine():
    d = select_engine(_config(engine="streaming"), 1_000, 642, available_gb=64.0)
    assert d.use_streaming and d.note is None and d.warning is None


def test_auto_keeps_memory_engine_when_streaming_matrices_are_larger():
    # 1 500 frames x 20 000 atoms: streaming needs D x D matrices (D = 60 000,
    # 28.8 GB each) while the in-memory SVD route needs far less.
    d = select_engine(_config(), 1_500, 20_000, available_gb=4.0)
    assert not d.use_streaming
    assert d.stream_peak_gb > d.peak_gb
    assert "smaller --cov-group" in d.warning and "--engine streaming" not in d.warning
    # 500 frames x ~6 667 atoms (D = 20 000) with 2 GB available.
    d = select_engine(_config(), 500, 6_667, available_gb=2.0)
    assert not d.use_streaming


def test_auto_streams_when_neither_fits_but_streaming_is_smaller():
    d = select_engine(_config(), 400_000, 642, available_gb=1.0)
    assert d.use_streaming and d.stream_peak_gb < d.peak_gb
    assert d.note is not None and "streaming engine needs" in d.warning


def test_forced_streaming_warns_about_large_matrices():
    d = select_engine(_config(engine="streaming"), 1_500, 20_000, available_gb=4.0)
    assert d.use_streaming
    assert "60000 x 60000" in d.warning and "--engine memory" in d.warning


def test_stride_suggestion_is_absolute():
    # --stride 2 on 100 000 frames: 50 000 frames reach the engine. The suggestion
    # must be a larger --stride that fits, not the current one again.
    d = select_engine(_config(engine="memory", stride=2), 50_000, 642, available_gb=4.0)
    suggested = int(d.warning.split("--stride ")[1].split()[0].rstrip(","))
    assert suggested > 2
    assert estimate_peak_memory_gb(100_000 // suggested, 642) <= 3.0
    assert estimate_peak_memory_gb(100_000 // (suggested - 1), 642) > 3.0


def test_no_stride_suggested_when_selection_alone_is_too_large():
    d = select_engine(_config(engine="memory"), 1_000, 40_000, available_gb=0.6)
    assert "--stride" not in d.warning and "smaller --cov-group" in d.warning


# ---------------------------------------------------- frames and YAML key


def test_frames_in_window_respects_begin_end_and_stride():
    info = TrajectoryInfo(
        path=Path("a.xtc"),
        replicate_id="r1",
        system_name="s",
        n_frames=100_001,
        start_time_ps=0.0,
        end_time_ps=1_000_000.0,
        timestep_ps=10.0,
    )
    assert _frames_in_window(info, None, None, 1) == 100_001
    assert _frames_in_window(info, 0.0, 199_990.0, 1) == 20_000
    assert _frames_in_window(info, 500_000.0, None, 2) == 25_000
    assert _frames_in_window(info, 2e6, None, 1) == 1


def test_yaml_streaming_threshold_key(tmp_path):
    for name in ("top.pdb", "run.xtc"):
        (tmp_path / name).write_text("", encoding="utf-8")
    cfg = tmp_path / "a.yaml"
    cfg.write_text(
        "topology: top.pdb\ntrajectories: [run.xtc]\nstreaming_threshold_gb: 3.5\n",
        encoding="utf-8",
    )
    assert build_analysis_config({}, cfg).streaming_threshold_gb == 3.5
    cfg.write_text("topology: top.pdb\ntrajectories: [run.xtc]\n", encoding="utf-8")
    assert build_analysis_config({}, cfg).streaming_threshold_gb is None
    cfg.write_text(
        "topology: top.pdb\ntrajectories: [run.xtc]\nstreaming_threshold_gb: -1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="streaming_threshold_gb"):
        build_analysis_config({}, cfg)
