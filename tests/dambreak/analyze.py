#!/usr/bin/env python3
"""
analyze.py — readers + quantitative metrics for the dambreak battery.

Single source of truth for READING native AMReX plotfiles (via yt, reusing
the AMR-composite readers from plot_output.py) and for the CI metrics:
  - shock position error vs the SWASHES analytical solution
  - profile L1 / Linf error on the centerline
  - plateau jitter (detrended peak-to-peak of h on the post-rarefaction
    plateau — the classic-Roe ringing metric)

plot.py renders the figure; analysis.py combines both. Run those, not this.
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytical import dambreak_on_wet_no_friction_analytical

# Re-export the yt-native readers so plot.py / analysis.py import from here
from plot_output import (  # noqa: F401
    find_variable, get_domain_bounds, get_level_info,
    build_amr_composite, composite_profile, plot_level_strip,
)


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


def plateau_band(t, L, hl, hr, x0):
    """Interior (trimmed 5% each side) of the exact solution's flat
    intermediate plateau — the post-rarefaction plateau that ends at the
    shock. Returns (xa, xb) or None."""
    xg, h_ana, _ = exact_profile(t, L, hl, hr, x0)
    dha = np.gradient(h_ana, xg)
    flat = np.abs(dha) < 1e-3 * (hl - hr)
    edges = np.flatnonzero(np.diff(flat.astype(int)) != 0) + 1
    runs = [r for r in np.split(np.arange(len(xg)), edges) if flat[r[0]]]
    if not runs:
        return None
    run = max(runs, key=lambda r: np.max(h_ana[r]))
    xa = xg[run[0]] + 0.05 * (xg[run[-1]] - xg[run[0]])
    xb = xg[run[-1]] - 0.05 * (xg[run[-1]] - xg[run[0]])
    return xa, xb


def all_metrics(x, h, t, L, hl, hr, x0):
    """Compute every CI metric for one centerline profile."""
    xg, h_ana, _ = exact_profile(t, L, hl, hr, x0)
    h_ref = np.interp(x, xg, h_ana)
    m = {}
    m["shock_num"] = numeric_shock_position(x, h)
    m["shock_ana"] = analytical_shock_position(t, L, hl, hr, x0)
    m["shock_err"] = m["shock_num"] - m["shock_ana"]
    m["shock_err_pct"] = 100.0 * abs(m["shock_err"]) / L
    m["L1"] = float(np.mean(np.abs(h - h_ref)))
    m["Linf"] = float(np.max(np.abs(h - h_ref)))
    band = plateau_band(t, L, hl, hr, x0)
    if band is not None:
        sel = (x > band[0]) & (x < band[1])
        if np.count_nonzero(sel) >= 5:
            hp = h[sel]
            m["jitter"] = float(np.max(hp) - np.min(hp))
            m["plateau_Linf"] = float(np.max(np.abs(hp - np.interp(x[sel], xg, h_ana))))
        else:
            m["jitter"] = m["plateau_Linf"] = np.nan
    else:
        m["jitter"] = m["plateau_Linf"] = np.nan
    return m


def metrics_summary(m):
    return (
        f"shock position: numerical {m['shock_num']:.4f}  analytical {m['shock_ana']:.4f}"
        f"  error {m['shock_err']:+.4f} ({m['shock_err_pct']:.2f}% of L)\n"
        f"profile error vs exact: L1 = {m['L1']:.3e}   Linf = {m['Linf']:.3e}\n"
        f"plateau jitter (detrended p2p): {m['jitter']:.3e}"
        f"   plateau Linf err: {m['plateau_Linf']:.3e}"
    )
