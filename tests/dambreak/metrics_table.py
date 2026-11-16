#!/usr/bin/env python3
"""
metrics_table.py — the dambreak battery scoreboard, as CSV.

One row per (run, score time). Accuracy is scored against three yardsticks:

  * the analytical (SWASHES) solution -> L1, L2, Linf, shock error
  * the STABLE single-level run       -> stable_*, gain_stable_*
    (uniform grid at the AMR level-1 cell size, dx0/2: what an AMR run with
    one added level should look like if the refinement were doing its job)
  * the level-0 single-level run      -> base_*, gain_base_*
    (the unrefined solution AMR has to beat)

The two gain families use the same definition:

    gain_<slot>_p = (that single-level run's error vs exact)_p / (this run's error)_p

Read them differently because they answer different questions:

  gain_base > 1     refinement earns its keep (AMR beats no-AMR).
                    ~1 means the extra level is decoration.
  gain_stable ~ 1   AMR recovers the uniform fine-grid accuracy.
                    << 1 means a level's worth of resolution is being lost
                    somewhere in the AMR coupling.

Usage:
  # every run under outputs/, scored at t=15
  uv run python metrics_table.py --times 15 -o metrics.csv

  # specific runs / times
  uv run python metrics_table.py --times 5,15,30 outputs/roe/amr__n1

Deep initial conditions (hl/hr not 0.005/0.001) must be passed explicitly,
since the HDF5 filename is not a reliable record of the ICs.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yt
yt.set_log_level(50)

from analyze import (all_metrics, composite_profile, find_variable,
                     find_reference_plotfile, REFERENCE_RUNS)

NORM_COLS = ("L1", "L2", "Linf")

COLUMNS = (
    ["solver", "variant", "tag", "run_dir", "max_level", "t", "hl", "hr", "L",
     "shock_num", "shock_ana", "shock_err", "shock_err_pct"]
    + list(NORM_COLS)
    + ["plateau_L1", "plateau_L2", "plateau_Linf", "lake_range", "lake_level_err"]
    + [f"{slot}_{k}" for slot, _, _ in REFERENCE_RUNS for k in ("t",) + NORM_COLS]
    + [f"gain_{slot}_{k}" for slot, _, _ in REFERENCE_RUNS for k in NORM_COLS]
)


def iter_runs(outputs_root):
    """Yield (solver, variant, tag, run_dir) for every outputs/<solver>/<run>."""
    if not os.path.isdir(outputs_root):
        return
    for solver in sorted(os.listdir(outputs_root)):
        sdir = os.path.join(outputs_root, solver)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir)):
            rdir = os.path.join(sdir, name)
            if not os.path.isdir(rdir) or not name.startswith(
                    ("amr", "single", "noamr", "cfl", "reflev")):
                continue
            variant, _, tag = name.partition("__")
            yield solver, variant, tag, rdir


def load_profile(plotfile):
    ds = yt.load(plotfile)
    lo = ds.domain_left_edge.to_ndarray()
    hi = ds.domain_right_edge.to_ndarray()
    y_center = 0.5 * (float(lo[1]) + float(hi[1]))
    field = find_variable(ds, "h")
    if field is None:
        raise RuntimeError(f"no water-depth variable in {plotfile}")
    x, h, _ = composite_profile(ds, field, y_center)
    nlvl = int(np.max(np.asarray(ds.index.grid_levels, dtype=int))) + 1
    return float(ds.current_time), x, h, nlvl


def references_for(run_dir):
    """Sibling single-level runs for this run: {slot: run_dir}, stable first."""
    if os.path.basename(os.path.abspath(run_dir)).startswith("single_level"):
        return {}
    parent = os.path.dirname(os.path.abspath(run_dir))
    out = {}
    for slot, pat, _label in REFERENCE_RUNS:
        hits = sorted(os.path.join(parent, h) for h in os.listdir(parent)
                      if h.startswith(pat.rstrip("*")))
        if hits:
            out[slot] = hits[-1]
    return out


def score(run_dir, t_target, hl, hr, x0, L, references=None):
    plt = find_reference_plotfile(run_dir, t_target)
    if plt is None:
        return None
    t, x, h, nlvl = load_profile(plt)
    if len(x) == 0:
        return None

    refs, ref_t = {}, {slot: np.nan for slot, _, _ in REFERENCE_RUNS}
    for slot, path in (references or {}).items():
        rplt = find_reference_plotfile(path, t_target)
        if not rplt:
            continue
        rt, xr, hr_prof, _ = load_profile(rplt)
        rm = all_metrics(xr, hr_prof, rt, L, hl, hr, x0)
        refs[slot] = (xr, hr_prof, rm["L1"], rm["L2"], rm["Linf"])
        ref_t[slot] = rt

    m = all_metrics(x, h, t, L, hl, hr, x0, refs=refs)
    row = dict(m)
    row.update(t=t, max_level=nlvl - 1, **{f"{s}_t": v for s, v in ref_t.items()})
    return row


def main():
    p = argparse.ArgumentParser(description="Dambreak battery metrics -> CSV")
    p.add_argument("runs", nargs="*",
                   help="Run dirs (default: every run under <test>/outputs)")
    p.add_argument("--outputs-root", default=None,
                   help="Root holding <solver>/<run>/ dirs (default: ./outputs)")
    p.add_argument("--times", default="15",
                   help="Comma-separated score times (default: 15)")
    p.add_argument("-o", "--output", default="metrics.csv")
    p.add_argument("--hl", type=float, default=0.005)
    p.add_argument("--hr", type=float, default=0.001)
    p.add_argument("--x0", type=float, default=5.0)
    p.add_argument("--L", type=float, default=10.0)
    p.add_argument("--no-reference", action="store_true",
                   help="Skip the AMR-vs-single-level columns")
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.runs:
        runs = []
        for r in args.runs:
            r = os.path.abspath(r)
            solver = os.path.basename(os.path.dirname(r))
            variant, _, tag = os.path.basename(r).partition("__")
            runs.append((solver, variant, tag, r))
    else:
        root = args.outputs_root or os.path.join(here, "outputs")
        runs = list(iter_runs(root))

    times = [float(v) for v in str(args.times).split(",") if v.strip()]

    rows = []
    for solver, variant, tag, rdir in runs:
        references = {} if args.no_reference else references_for(rdir)
        for tt in times:
            try:
                m = score(rdir, tt, args.hl, args.hr, args.x0, args.L, references)
            except Exception as exc:  # keep the table going on a bad plotfile
                print(f"  ! {rdir} @ t~{tt}: {exc}", file=sys.stderr)
                continue
            if m is None:
                continue
            row = {c: "" for c in COLUMNS}
            row.update(solver=solver, variant=variant, tag=tag,
                       run_dir=os.path.relpath(rdir, here),
                       hl=args.hl, hr=args.hr, L=args.L)
            for c in COLUMNS:
                if c in m and m[c] == m[c]:
                    row[c] = m[c]
            rows.append(row)

            def g(key):
                v = m.get(key, float("nan"))
                return f"{v:.2f}" if v == v else "  - "

            print(f"  {solver}/{variant}__{tag}  t={m['t']:.4f}  "
                  f"L1={m['L1']:.3e}  vs base x{g('gain_base_L1')}  "
                  f"vs stable {100 * m.get('gain_stable_L1', float('nan')):.0f}%")

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.output}  ({len(rows)} rows, {len(COLUMNS)} cols)")


if __name__ == "__main__":
    main()
