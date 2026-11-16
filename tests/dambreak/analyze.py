#!/usr/bin/env python3
"""
analyze.py — readers + quantitative metrics for the dambreak battery.

Single source of truth for READING native AMReX plotfiles (via yt, reusing
the AMR-composite readers from plot_output.py) and for the CI metrics:

  - shock position error vs the SWASHES analytical solution
  - L1 / L2 / Linf norms of the water-depth error, computed as genuine
    integrals on the (non-uniform) AMR-composite grid
  - the same norms against the two uniform single-level runs — the level-1
    cell size (`single_level_fine`, the "stable" solution) and the level-0
    cell size (`single_level`, the unrefined baseline) — plus the accuracy
    gain each of those makes against the analytical solution

The norms are length-weighted, which matters because the AMR composite
centreline is not uniformly spaced: fine-level cells are concentrated where
the wave is, so an unweighted point average would silently reward refinement.
For a single-level run the weights are uniform and this reduces to the
textbook mean / RMS / max.

plot.py renders the figure; analysis.py combines both. Run those, not this.
"""

import numpy as np

import glob
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytical import dambreak_on_wet_no_friction_analytical

# Re-export the yt-native readers so plot.py / analysis.py import from here
from plot_output import (  # noqa: F401
    find_variable, get_domain_bounds, get_level_info, grid_boxes,
    build_amr_composite, composite_profile, plot_level_strip,
)


# ---------------------------------------------------------------------------
# Reference runs: the single-level solution is the yardstick every AMR run is
# measured against. Refinement must buy accuracy vs BOTH the analytical
# solution and the single-level run; otherwise the extra level is noise.
# ---------------------------------------------------------------------------

def plotfile_time(path):
    """Simulation time of a native AMReX plotfile, read from its Header."""
    hdr = os.path.join(str(path), "Header")
    if not os.path.isfile(hdr):
        return None
    with open(hdr) as f:
        lines = f.read().splitlines()
    try:
        return float(lines[8])
    except (IndexError, ValueError):
        return None


def find_reference_plotfile(path, t):
    """Resolve a reference to a concrete plotfile for time t.

    `path` may be a plotfile directory or a run directory holding plt*; the
    plotfile closest to t is returned. Rotation backups (plt0123.old.456) are
    ignored.
    """
    if path is None:
        return None
    if os.path.basename(str(path)).startswith("plt"):
        return path
    cands = [p for p in sorted(glob.glob(os.path.join(str(path), "plt[0-9]*")))
             if ".old." not in p]
    best = None
    for p in cands:
        tt = plotfile_time(p)
        if tt is None:
            continue
        if best is None or abs(tt - t) < abs(best[1] - t):
            best = (p, tt)
    return best[0] if best else None


#: The two single-level yardsticks, in reporting order.
#:
#:   stable — a UNIFORM run at the AMR level-1 cell size (dx0/2). This is the
#:            "stable" solution: what an AMR run with one added level should
#:            look like if the refinement were doing its job. AMR cannot beat
#:            it (it solves the same equations on the same finest dx plus
#:            coupling), so the honest test is how close it gets.
#:   base   — the unrefined level-0 solution: what AMR has to beat, otherwise
#:            the extra level is decoration.
#:
#: `single_level__*` does not match `single_level_fine__n1` (the character
#: after "single_level" must be '_'), so the two globs are disjoint.
REFERENCE_RUNS = (
    ("stable", "single_level_fine__*", "single level dx0/2"),
    ("base", "single_level__*", "single level dx0"),
)

REFERENCE_SLOTS = tuple(slot for slot, _, _ in REFERENCE_RUNS)


def _is_single_level(plotfile):
    run_dir = os.path.basename(os.path.dirname(os.path.abspath(str(plotfile))))
    return run_dir.startswith("single_level")


def default_references(plotfile):
    """[(slot, label, run_dir)] for the sibling single-level runs, stable first.

    A single-level run is its own yardstick, so `single_level*` plotfiles get
    no references at all — comparing a run with itself would only print a
    meaningless `gain x1.00`.
    """
    if _is_single_level(plotfile):
        return []
    parent = os.path.dirname(os.path.dirname(os.path.abspath(str(plotfile))))
    out = []
    for slot, pat, label in REFERENCE_RUNS:
        hits = sorted(glob.glob(os.path.join(parent, pat)))
        if hits:
            out.append((slot, label, hits[-1]))
    return out


def load_reference(ref_path, t, hl, hr, x0, L, label=None):
    """Load one reference solution at (approximately) time t.

    Returns a dict with x, h, hu, t, label and the reference's own L1/L2/Linf
    against the analytical solution at ITS time, so the accuracy gain is
    like-for-like. Returns None when nothing usable is found.
    """
    import yt

    rp = find_reference_plotfile(ref_path, t)
    if rp is None:
        return None
    rds = yt.load(rp)
    rt = float(rds.current_time)
    rlo = rds.domain_left_edge.to_ndarray()
    rhi = rds.domain_right_edge.to_ndarray()
    ry = 0.5 * (float(rlo[1]) + float(rhi[1]))

    ref = {"t": rt, "label": label or "single level", "path": rp}
    for key, field in (("h", "h"), ("hu", "hu")):
        f = find_variable(rds, field)
        if f is None:
            continue
        xr, vr, _ = composite_profile(rds, f, ry)
        if len(xr) == 0:
            continue
        ref[key] = vr
        ref.setdefault("x", xr)

    if "h" not in ref:
        return None
    rm = all_metrics(ref["x"], ref["h"], rt, L, hl, hr, x0)
    ref["L1"], ref["L2"], ref["Linf"] = rm["L1"], rm["L2"], rm["Linf"]
    return ref


def cell_weights(x):
    """Trapezoidal cell widths for a possibly non-uniform centreline profile.

    The AMR-composite profile concatenates grids of different dx, so each
    sample stands for a different length of the domain. Weighting by length
    makes the norms a discretisation of ||e|| = (int |e|^p dx / int dx)^(1/p),
    i.e. independent of how the samples happen to be distributed.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.ones_like(x)
    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return np.maximum(w, 0.0)


def norms(x, e):
    """Length-weighted L1, L2 and Linf of the error e sampled at x.

    L1   = sum(w |e|)      / sum(w)          mean absolute error
    L2   = sqrt(sum(w e^2) / sum(w))         RMS error
    Linf = max(|e|)                          worst point (measure-zero set)
    """
    e = np.asarray(e, dtype=float)
    out = {"L1": np.nan, "L2": np.nan, "Linf": np.nan}
    if e.size == 0:
        return out
    w = cell_weights(x)
    tot = float(w.sum())
    if tot > 0.0:
        out["L1"] = float(np.sum(w * np.abs(e)) / tot)
        out["L2"] = float(np.sqrt(np.sum(w * e * e) / tot))
    out["Linf"] = float(np.max(np.abs(e)))
    return out


def compare(x, h, x_ref, h_ref):
    """L1 / L2 / Linf difference between two centreline profiles on their overlap."""
    out = {"L1": np.nan, "L2": np.nan, "Linf": np.nan}
    if x_ref is None or len(x_ref) < 2 or len(x) < 2:
        return out
    lo = max(x.min(), x_ref.min())
    hi = min(x.max(), x_ref.max())
    sel = (x >= lo) & (x <= hi)
    if np.count_nonzero(sel) < 2:
        return out
    n = norms(x[sel], h[sel] - np.interp(x[sel], x_ref, h_ref))
    out["L1"], out["L2"], out["Linf"] = n["L1"], n["L2"], n["Linf"]
    return out


def exact_profile(t, L, hl, hr, x0, n=4001, x_lo=0.0):
    xg = np.linspace(x_lo, x_lo + L, n)
    h_ana, u_ana = dambreak_on_wet_no_friction_analytical(t, xg, L=L, hl=hl, hr=hr, x0=x0)
    return xg, h_ana, u_ana


def analytical_shock_position(t, L, hl, hr, x0):
    """Shock location of the exact solution: right edge of the region where
    h exceeds the h_R plateau."""
    xg, h_ana, _ = exact_profile(t, L, hl, hr, x0)
    above = np.nonzero(h_ana > hr + 1e-9)[0]
    if len(above) == 0:
        return xg[0]
    return xg[above[-1]] + (xg[1] - xg[0])


def numeric_shock_position(x, h):
    """Shock location from the numeric profile: steepest h gradient in the
    right half of the domain."""
    sel = x > x.max() * 0.5
    xs, hs = x[sel], h[sel]
    if len(xs) < 3:
        return np.nan
    dh = np.diff(hs)
    i = int(np.argmin(dh))
    return 0.5 * (xs[i] + xs[i + 1])


def _flat_runs(t, L, hl, hr, x0):
    """Flat (zero-gradient) segments of the exact solution, as index ranges."""
    xg, h_ana, _ = exact_profile(t, L, hl, hr, x0)
    dha = np.gradient(h_ana, xg)
    flat = np.abs(dha) < 1e-3 * (hl - hr)
    edges = np.flatnonzero(np.diff(flat.astype(int)) != 0) + 1
    runs = [r for r in np.split(np.arange(len(xg)), edges) if flat[r[0]]]
    return xg, h_ana, runs


def _trim(xg, run, frac=0.05):
    xa = xg[run[0]] + frac * (xg[run[-1]] - xg[run[0]])
    xb = xg[run[-1]] - frac * (xg[run[-1]] - xg[run[0]])
    return xa, xb


def plateau_band(t, L, hl, hr, x0):
    """The flat shelf between the rarefaction tail and the analytical shock
    elbow, scored and plotted in full.

    Two things are deliberate here:

      * The shelf is the intermediate state (h = h_m), NOT the left reservoir.
        The naive "flattest run with the highest h" picks the left reservoir
        (h = h_L), which is a different diagnostic entirely — that bug made
        every earlier 'plateau jitter' number a lake-at-rest measurement.
      * The band is not trimmed at the right. The shelf physically ends at the
        shock, so the metric covers the cells that see the front coming; the
        error build-up toward the elbow is part of the picture rather than a
        5 % margin we never look at. The right edge is pinned to the exact
        shock position.
    """
    xg, h_ana, runs = _flat_runs(t, L, hl, hr, x0)
    if not runs:
        return None
    tol = 0.02 * (hl - hr)  # distinguish the three levels h_L / plateau / h_R
    inner = [r for r in runs
             if not (abs(h_ana[r].mean() - hl) < tol or abs(h_ana[r].mean() - hr) < tol)]
    if not inner:
        return None
    run = max(inner, key=lambda r: len(r))
    xa = float(xg[run[0]])
    xb = float(min(analytical_shock_position(t, L, hl, hr, x0),
                   xg[run[-1]] + (xg[1] - xg[0])))
    # The rarefaction tail can come out of the gradient detector mid-fan; start
    # where the exact solution actually reaches the shelf level instead.
    if run is not None:
        at_shelf = np.abs(h_ana - h_ana[run].mean()) < 1e-3 * (hl - hr)
        mask = at_shelf & (xg >= xa)
        if np.any(mask):
            xa = float(xg[np.argmax(mask)])
    return xa, xb


def lake_band(t, L, hl, hr, x0):
    """Interior of the exact solution's left reservoir (undisturbed, at rest).
    A well-balanced solver must keep this EXACTLY constant; the classic-Roe
    noise here is a well-balancedness metric, not plateau roughness."""
    xg, h_ana, runs = _flat_runs(t, L, hl, hr, x0)
    if not runs:
        return None
    tol = 0.02 * (hl - hr)
    lakes = [r for r in runs if abs(h_ana[r].mean() - hl) < tol]
    if not lakes:
        return None
    return _trim(xg, max(lakes, key=lambda r: len(r)))


def all_metrics(x, h, t, L, hl, hr, x0, refs=None):
    """Compute every CI metric for one centerline profile.

    `refs` maps a reference slot ("stable", "base") to
    (x_ref, h_ref, L1_ref, L2_ref, Linf_ref), where the last three are that
    reference's own norms against the analytical solution. Each slot adds
    `<slot>_L1/L2/Linf` (profile difference between the two numerical
    solutions) and `gain_<slot>_L1/L2/Linf` (how much better this run is than
    the reference *against the exact solution*).
    """
    xg, h_ana, _ = exact_profile(t, L, hl, hr, x0)
    h_ref = np.interp(x, xg, h_ana)
    m = {}
    m["shock_num"] = numeric_shock_position(x, h)
    m["shock_ana"] = analytical_shock_position(t, L, hl, hr, x0)
    m["shock_err"] = m["shock_num"] - m["shock_ana"]
    m["shock_err_pct"] = 100.0 * abs(m["shock_err"]) / L
    m.update(norms(x, h - h_ref))

    # The same norms restricted to the intermediate plateau, where the
    # classic Roe discretisation is expected to be weakest.
    band = plateau_band(t, L, hl, hr, x0)
    sel = (x > band[0]) & (x < band[1]) if band is not None else None
    if sel is not None and np.count_nonzero(sel) >= 5:
        pn = norms(x[sel], h[sel] - np.interp(x[sel], xg, h_ana))
        m["plateau_L1"], m["plateau_L2"], m["plateau_Linf"] = (
            pn["L1"], pn["L2"], pn["Linf"])
    else:
        m["plateau_L1"] = m["plateau_L2"] = m["plateau_Linf"] = np.nan

    # Well-balancedness: excursion of the undisturbed left reservoir, which a
    # well-balanced solver keeps at roundoff and classic Roe does not.
    lb = lake_band(t, L, hl, hr, x0)
    sell = (x > lb[0]) & (x < lb[1]) if lb is not None else None
    if sell is not None and np.count_nonzero(sell) >= 5:
        hl_num = h[sell]
        m["lake_range"] = float(np.max(hl_num) - np.min(hl_num))
        m["lake_level_err"] = float(np.abs(np.mean(hl_num) - hl))
    else:
        m["lake_range"] = m["lake_level_err"] = np.nan

    # Single-level yardsticks. Emit every slot's keys even when absent so the
    # CSV header stays stable across runs.
    refs = refs or {}
    for slot in REFERENCE_SLOTS:
        for k in ("L1", "L2", "Linf"):
            m[f"{slot}_{k}"] = np.nan
            m[f"gain_{slot}_{k}"] = np.nan
    for slot, r in refs.items():
        if r is None:
            continue
        c = compare(x, h, r[0], r[1])
        for i, k in enumerate(("L1", "L2", "Linf")):
            m[f"{slot}_{k}"] = c[k]
            # r[2:5] = the reference's own norms against the analytical
            # solution, at the reference's own time (like-for-like).
            if r[2 + i] is not None and m[k] == m[k] and m[k] > 0.0:
                m[f"gain_{slot}_{k}"] = float(r[2 + i] / m[k])
    return m


def metrics_summary(m, ref_labels=None):
    """Human-readable metric block. ref_labels maps slot -> display label."""
    txt = (
        f"shock position: numerical {m['shock_num']:.4f}  analytical {m['shock_ana']:.4f}"
        f"  error {m['shock_err']:+.4f} ({m['shock_err_pct']:.2f}% of L)\n"
        f"error vs exact:      L1 = {m['L1']:.3e}   L2 = {m['L2']:.3e}"
        f"   Linf = {m['Linf']:.3e}\n"
        f"  plateau band:      L1 = {m['plateau_L1']:.3e}   L2 = {m['plateau_L2']:.3e}"
        f"   Linf = {m['plateau_Linf']:.3e}\n"
        f"left reservoir:      range = {m['lake_range']:.3e}"
        f"   level offset = {m['lake_level_err']:.3e}"
    )
    for slot, label in (ref_labels or {}).items():
        if m.get(f"{slot}_L1", np.nan) != m.get(f"{slot}_L1", np.nan):
            continue
        txt += (
            f"\nvs {label}:"
            f" L1 = {m[slot + '_L1']:.3e}   L2 = {m[slot + '_L2']:.3e}"
            f"   Linf = {m[slot + '_Linf']:.3e}"
            f"   [profile difference]"
        )
        if slot == "stable":
            txt += (
                f"\naccuracy recovered from {label}:"
                f" L1 {(100.0 * m['gain_stable_L1']):.0f}%"
                f"   L2 {(100.0 * m['gain_stable_L2']):.0f}%"
                f"   Linf {(100.0 * m['gain_stable_Linf']):.0f}%"
                f"   [100% = as good as the uniform dx0/2 run]"
            )
        else:
            txt += (
                f"\naccuracy gain over {label}:"
                f" L1 x{m['gain_base_L1']:.2f}   L2 x{m['gain_base_L2']:.2f}"
                f"   Linf x{m['gain_base_Linf']:.2f}"
                f"   [must be > 1, else the refinement is decoration]"
            )
    return txt
