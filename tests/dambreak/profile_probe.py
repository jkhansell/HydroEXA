#!/usr/bin/env python
"""Centerline h(x) profile probe for native AMReX plotfiles (yt-based).

Usage: python profile_probe.py <plotfile> <out_png> [y_frac]
Prints level grid boxes and C-F boundary x-locations; plots h along a
horizontal line (default mid-domain y).
"""
import sys
import numpy as np
import yt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pf = sys.argv[1]
out = sys.argv[2]
y_frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

ds = yt.load(pf)
time = float(ds.current_time)
dom = ds.domain_left_edge, ds.domain_right_edge
print(f"time={time}  dom_x=[{dom[0][0]},{dom[1][0]}]  dom_y=[{dom[0][1]},{dom[1][1]}]")

# level grid boxes from the AMR structure
geo = ds.index.grids
for g in sorted(geo, key=lambda g: g.Level):
    if g.Level > 0:
        L = g.Level
        le, re = g.LeftEdge, g.RightEdge
        print(f"  L{L} grid: x=[{le[0]:.4f},{re[0]:.4f}] y=[{le[1]:.4f},{re[1]:.4f}]")

y = dom[0][1] + y_frac * (dom[1][1] - dom[0][1])
ray = ds.ortho_ray(0, (1, y))   # ray along x at fixed y
srt = np.argsort(np.array(ray[("index", "x")]))
x = np.array(ray[("index", "x")][srt])
h = np.array(ray[("boxlib", "h_fluid")][srt])

# report the sharpest jump in h (potential jitter at C-F boundary)
dh = np.abs(np.diff(h))
i = int(np.argmax(dh))
print(f"  max |dh| between adjacent samples: {dh[i]:.3e} at x=[{x[i]:.4f},{x[i+1]:.4f}]  (h={h[i]:.6f} -> {h[i+1]:.6f})")
# top 5 jumps
idx = np.argsort(dh)[-5:][::-1]
for j in idx:
    print(f"    jump {dh[j]:.3e} at x=[{x[j]:.4f},{x[j+1]:.4f}] h={h[j]:.6f}->{h[j+1]:.6f}")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(x, h, lw=0.7, marker=".", ms=2)
ax.set_xlabel("x"); ax.set_ylabel("h")
ax.set_title(f"{pf.split('/')[-1]}  t={time:.3f}  y={y:.3f}")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("wrote", out)
