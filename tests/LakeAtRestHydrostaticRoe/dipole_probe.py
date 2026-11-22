#!/usr/bin/env python3
"""Locate velocity dipoles / eta errors in a native AMReX plotfile (T-battery probe).

Usage: python dipole_probe.py <plotfile_dir> [--H 0.010] [--top 5]

Uses yt (already proven to read HydroEXA's native plotfiles). Prints, per level:
  - max |u|, max |hu| and the exact cell (x, y) of the worst offender
  - max |eta - H| and its cell
  - the 5x5 hu/eta neighbourhood around the worst |u| cell (to expose dipole structure)
"""
import sys
import argparse
import numpy as np
import yt

NODATA = -9999.0


def fmt_cell(g, gi, gj, extra=""):
    le = g.LeftEdge.v
    dx = g.dds.v
    xc = le[0] + (gi + 0.5) * dx[0]
    yc = le[1] + (gj + 0.5) * dx[1]
    return f"lvl={g.Level} grid=({le[0]:.4f},{le[1]:.4f})-({g.RightEdge.v[0]:.4f},{g.RightEdge.v[1]:.4f}) cell=({gi},{gj}) @ (x={xc:.5f}, y={yc:.5f}) {extra}"


def neighbourhood(arr, ii, jj, rad=2):
    sl = arr[max(0, ii - rad):ii + rad + 1, max(0, jj - rad):jj + rad + 1]
    return sl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plotfile")
    ap.add_argument("--H", type=float, default=0.010, help="target lake level")
    ap.add_argument("--top", type=int, default=5, help="how many worst cells to list")
    args = ap.parse_args()

    ds = yt.load(args.plotfile)
    print(f"time={float(ds.current_time):.9g}  max_level={ds.max_level}  H={args.H}")

    for lev in range(ds.max_level + 1):
        grids = [g for g in ds.index.grids if g.Level == lev]
        u_best = (-1.0, None)
        eta_best = (0.0, None)
        hu_best = (0.0, None)
        for g in grids:
            h = g["h_fluid"].v
            hu = g["hu_momentum"].v
            hv = g["hv_momentum"].v
            eta = h + g["z_bathymetry"].v
            wet = h > 1e-12
            with np.errstate(all="ignore"):
                u = np.where(wet, np.abs(hu) / np.maximum(h, 1e-300), 0.0)
            v = np.where(wet, np.abs(hv) / np.maximum(h, 1e-300), 0.0)
            # yt field arrays are (x, y, z) with a dummy z axis of 1
            u2, hu2, hv2, eta2, h2 = (a[:, :, 0] for a in (u, hu, hv, eta, h))
            k = int(np.argmax(u2))
            ii, jj = np.unravel_index(k, u2.shape)
            if u2[ii, jj] > u_best[0]:
                u_best = (u2[ii, jj], (g, ii, jj, h2[ii, jj], hu2[ii, jj], hv2[ii, jj], v2 := v[:, :, 0][ii, jj]))
            e = int(np.argmax(np.abs(eta2 - args.H)))
            ei, ej = np.unravel_index(e, eta2.shape)
            if abs(eta2[ei, ej] - args.H) > abs(eta_best[0]):
                eta_best = (eta2[ei, ej] - args.H, (g, ei, ej, h2[ei, ej], eta2[ei, ej]))
            m = int(np.argmax(np.abs(hu2)))
            mi, mj = np.unravel_index(m, hu2.shape)
            if abs(hu2[mi, mj]) > abs(hu_best[0]):
                hu_best = (hu2[mi, mj], (g, mi, mj, h2[mi, mj]))

        if u_best[1] is None:
            print(f"L{lev}: no grids")
            continue

        # --- census of dry/defective cells ---------------------------------
        dry_n, mask_n, bbox = 0, 0, None
        for g in grids:
            h = g["h_fluid"].v[:, :, 0]
            mask = g["extra_comp_4"].v[:, :, 0]
            bad = h <= 1e-12
            n = int(bad.sum())
            if n:
                dry_n += n
                mask_n += int((mask[bad] <= -9998.0).sum())
                ii_b = np.where(bad.any(axis=1))[0][[0, -1]]
                jj_b = np.where(bad.any(axis=0))[0][[0, -1]]
                le = g.LeftEdge.v
                dx = g.dds.v
                bb = (le[0] + ii_b[0] * dx[0], le[1] + jj_b[0] * dx[1],
                      le[0] + (ii_b[1] + 1) * dx[0], le[1] + (jj_b[1] + 1) * dx[1])
                bbox = bb if bbox is None else (
                    min(bbox[0], bb[0]), min(bbox[1], bb[1]),
                    max(bbox[2], bb[2]), max(bbox[3], bb[3]))
        if dry_n:
            print(f"\nL{lev}: DRY CELLS n={dry_n}  of which mask==-9999: {mask_n}  bbox={bbox}")

        g, ii, jj, h, hu, hv, v = u_best[1]
        print(f"\nL{lev}: max|u|={u_best[0]:.6g}  max|hu|={hu_best[0]:.3e}  max|eta-H|={eta_best[0]:+.3e}")
        print(f"  worst |u| cell: {fmt_cell(g, ii, jj, extra=f'h={h:.6g} hu={hu:.6g} hv={hv:.6g} v={v:.6g}')}")
        ge, ei, ej, he, etae = eta_best[1]
        print(f"  worst eta cell: {fmt_cell(ge, ei, ej, extra=f'h={he:.6g} eta={etae:.9g} eta-H={eta_best[0]:+.3e}')}")
        if abs(eta_best[0]) > 1e-12:
            # 5x5 neighbourhood AROUND THE ETA-WORST cell: h / z / mask / eta-H
            h_g = ge["h_fluid"].v[:, :, 0]
            z_g = ge["z_bathymetry"].v[:, :, 0]
            m_g = ge["extra_comp_4"].v[:, :, 0]
            sl = (slice(max(0, ei - 2), ei + 3), slice(max(0, ej - 2), ej + 3))
            print(f"  h / z / mask / eta-H around worst eta cell (x {ei-2}..{ei+2}, y {ej-2}..{ej+2}, transposed):")
            for name, arr in (("h", h_g), ("z", z_g), ("mask", m_g), ("eta-H", (h_g + z_g) - args.H)):
                print(f"    {name}:")
                for row in arr[sl].T:
                    print("      " + " ".join(f"{x:+.3e}" if name != "mask" else f"{x:+.0f}" for x in row))


if __name__ == "__main__":
    main()
