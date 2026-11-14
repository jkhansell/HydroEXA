#!/usr/bin/env python3
"""
Analyze rectangular dam break results and compare against the exact analytical solution.

The exact solution for the 1D SWE dam break (h_L, h_R, flat bed, zero initial velocity)
is self-similar: h(x,t) = h(eta) where eta = x/t.

Solution structure:
  - Left of rarefaction (eta < lambda_L): h = h_L, u = 0
  - Rarefaction fan (lambda_L <= eta <= lambda_R): smooth transition
  - Right of shock (eta > lambda_S): h = h_R, u = 0
  - Between fan and shock (lambda_R < eta < lambda_S): post-rarefaction state

This script:
  1. Reads the latest plotfile from AMReX plotfile format
  2. Extracts a y-averaged profile of h
  3. Computes and overlays the exact solution
  4. Plots the comparison
"""

import numpy as np
import h5py
import argparse
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def exact_dam_break_solution(x, t, h_L, h_R, g=9.81, z=0.0):
    """
    Compute the exact self-similar solution to the 1D SWE dam break.

    Parameters
    ----------
    x : ndarray
        Position array.
    t : float
        Time.
    h_L : float
        Water depth left of dam.
    h_R : float
        Water depth right of dam.
    g : float
        Gravitational acceleration.
    z : float
        Bathymetry (must be flat for this solution).

    Returns
    -------
    h, u : ndarray
        Water depth and x-velocity at each x.
    """
    h = np.zeros_like(x, dtype=np.float64)
    u = np.zeros_like(x, dtype=np.float64)

    # Wave speeds
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    # Left edge of rarefaction: lambda_L = -c_L
    lambda_L = -c_L

    # State at the bottom of the rarefaction fan (interface between fan and shock)
    # From the rarefaction R1 characteristic: u + 2c = u_L + 2c_L = 2c_L
    # So at the interface: u_Rstar + 2*c_Rstar = 2*c_L
    # Also from the shock jump conditions across S:
    #   [hu] = 0  =>  h_Rstar * u_Rstar = h_R * u_S  (but we use the standard formula)
    #
    # The standard approach: solve for h_interface from the shock relation:
    #   u_S = 2*(c_L - c_interface)  (from rarefaction)
    #   u_S = 2*c_R*(h_interface/h_R - 1) / sqrt((h_interface/h_R + 1)/2)  (from shock)
    # Equating: 2*(c_L - c_interface) = 2*c_R*(h_interface/h_R - 1) / sqrt((h_interface/h_R + 1)/2)
    #
    # Simplified: solve for h_interface using the Rankine-Hugoniot condition:
    #   u_Rstar = 2*c_L - 2*c_Rstar  (from rarefaction fan)
    #   Shock speed: s = u_Rstar/2 + sqrt((u_Rstar/2)^2 + g*h_R/2)  ... no, let's use the direct formula.

    # Direct formula for h_interface (depth just right of rarefaction, left of shock):
    # From Toro Eq. (3.48):
    h_interface = ((4 * c_L / (4 * c_R) + 1) / 2) ** 2 * h_R
    # Wait, let me use the correct formula from Toro.
    # For the dam break: the interface depth h* satisfies:
    #   u*(h*) = 2*(sqrt(g*h_L) - sqrt(g*h*))  (from left rarefaction)
    #   u*(h*) = 2*g*(sqrt(h_L) - sqrt(h*)) / (sqrt(h_L) + sqrt(h*))  ... no.
    #
    # Let me use the standard approach:
    # From the left rarefaction: u + 2c = 2*c_L  =>  u = 2*(c_L - c)
    # From the right shock: the Rankine-Hugoniot gives:
    #   u = 2*c_R * (h_interface/h_R - 1) * sqrt(h_R / (2*h_L * (h_interface/h_R + 1)))  ... this is getting complex.
    #
    # Simplest: use the formula from many textbooks:
    #   h_interface = h_R * (1 + (9*g*h_L)**(1/4) / (2*c_R) - 1)^2  ... no.
    #
    # Let me just use the iterative approach or the closed form:
    # From Toro (SWE, Sec 3.3):
    #   h_interface = h_R * ((7*c_R - 6*c_L + 7*sqrt(g)*sqrt(h_R)) / (c_R + ...))  -- too complex.
    #
    # Simplest correct approach: solve the nonlinear equation for h_interface.
    from scipy.optimize import brentq

    def shock_relation(h_star):
        """
        The shock relation: the velocity from the rarefaction must match
        the velocity from the shock jump condition.
        """
        if h_star <= 0 or h_star >= h_L:
            return 1e10
        c_star = np.sqrt(g * h_star)
        u_from_rarefaction = 2 * (c_L - c_star)

        # Shock speed from RH: s = (h_Rstar*u_Rstar - h_L*u_L) / (h_Rstar - h_L)
        # But u_L = 0, so s = h_star * u_from_rarefaction / (h_star - h_R)
        # Also s = u_Rstar + c_Rstar * sqrt(1 + (h_star/h_R)/2 * (h_star/h_R - 1) / ...)
        # Standard shock speed: s = u_Rstar/2 + sqrt((u_Rstar/2)^2 + g*h_R*h_star/(h_R+h_star))  -- no.
        #
        # Correct shock speed (Toro Eq. 3.28 for SWE):
        s = 0.5 * (u_from_rarefaction + np.sqrt(u_from_rarefaction**2 + 2*g*h_star*h_R/(h_star + h_R)))
        # But also: s = u_Rstar + c_Rstar * sqrt((h_star + h_R)/(2*h_R))  -- no.
        #
        # Actually the RH condition for the shock gives:
        #   s = (h_star * u_from_rarefaction) / (h_star - h_R)
        s_rh = h_star * u_from_rarefaction / (h_star - h_R)

        return s - s_rh

    # h_interface is between h_R and h_L
    try:
        h_interface = brentq(shock_relation, h_R * 1.001, h_L * 0.999, xtol=1e-12)
    except ValueError:
        h_interface = h_R  # fallback

    c_interface = np.sqrt(g * h_interface)
    u_interface = 2 * (c_L - c_interface)

    # Shock speed
    s_shock = u_interface * h_interface / (h_interface - h_R)

    # lambda_R = speed of the right edge of the rarefaction fan = u_interface - c_interface
    lambda_R = u_interface - c_interface

    # Now compute the solution at each x
    eta = x / t

    for idx in range(len(x)):
        eta_i = eta[idx]

        if eta_i < lambda_L:
            # Left of rarefaction: undisturbed left state
            h[idx] = h_L
            u[idx] = 0.0

        elif eta_i < lambda_R:
            # Inside the rarefaction fan
            # u + 2*c = 2*c_L  (R1 invariant)
            # x/t = u - c  (characteristic: dx/dt = u - c = eta)
            # So: u = (2*c_L + eta) / 2 * 2/3 ... let me solve:
            #   u - c = eta
            #   u + 2c = 2*c_L
            # => 3c = 2*c_L - eta  =>  c = (2*c_L - eta) / 3
            # => u = eta + c = eta + (2*c_L - eta)/3 = (2*eta + 2*c_L)/3
            c_val = (2 * c_L - eta_i) / 3.0
            h[idx] = max(c_val**2 / g, 0.0)
            u[idx] = (2 * eta_i + 2 * c_L) / 3.0

        elif eta_i < s_shock:
            # Between rarefaction and shock: post-rarefaction state
            h[idx] = h_interface
            u[idx] = u_interface

        else:
            # Right of shock: undisturbed right state
            h[idx] = h_R
            u[idx] = 0.0

    return h, u


def read_amrex_plotfile(plotfile_path):
    """
    Read an AMReX plotfile and return cell-centered data.

    Returns
    -------
    data : dict of ndarray
        Variable name -> (nc, ny, nx) array
    geom : dict
        x_ll, y_ll, dx, dy, nx, ny
    """
    # AMReX plotfiles are in HDF5 format
    with h5py.File(plotfile_path, "r") as f:
        # Get geometry
        x_ll = float(f.attrs.get("x_ll", 0.0))
        y_ll = float(f.attrs.get("y_ll", 0.0))
        dx = float(f.attrs.get("dx", 1.0))
        dy = float(f.attrs.get("dy", 1.0))

        # Read variables
        data = {}
        for key in f.keys():
            if key.startswith("cell") or key in ["h", "hu", "hv", "depth", "water_depth"]:
                data[key] = np.array(f[key])

        # If no standard names, try to read all 2D/3D arrays
        if not data:
            for key in f.keys():
                dset = f[key]
                if isinstance(dset, h5py.Dataset):
                    arr = np.array(dset)
                    if arr.ndim in [2, 3]:
                        data[key] = arr

    nx = data[list(data.keys())[0]].shape[-1] if data else 0
    ny = data[list(data.keys())[0]].shape[-2] if data else 0

    geom = {
        "x_ll": x_ll,
        "y_ll": y_ll,
        "dx": dx,
        "dy": dy,
        "nx": nx,
        "ny": ny,
    }

    return data, geom


def main():
    parser = argparse.ArgumentParser(description="Analyze rectangular dam break")
    parser.add_argument("plotfile", help="Path to AMReX plotfile (e.g., plotfiles/plt00000)")
    parser.add_argument("--h-L", type=float, default=0.5, help="Initial left water depth")
    parser.add_argument("--h-R", type=float, default=0.1, help="Initial right water depth")
    parser.add_argument("--output", type=str, default="dam_break_analysis.png", help="Output plot path")
    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("ERROR: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    # Read plotfile
    print(f"Reading plotfile: {args.plotfile}")
    data, geom = read_amrex_plotfile(args.plotfile)

    # Find the water depth variable
    h_key = None
    for key in ["h", "depth", "water_depth", "Cell0", "Cell_0"]:
        if key in data:
            h_key = key
            break
    if h_key is None:
        print(f"Available variables: {list(data.keys())}")
        sys.exit(1)

    h_arr = data[h_key]
    print(f"Using variable '{h_key}' for water depth")
    print(f"Shape: {h_arr.shape}, Range: [{h_arr.min():.6f}, {h_arr.max():.6f}]")

    # Extract y-averaged profile
    h_yavg = h_arr.mean(axis=0) if h_arr.ndim == 2 else h_arr[0]

    x_coords = (np.arange(geom["nx"]) + 0.5) * geom["dx"] + geom["x_ll"]

    # Compute exact solution
    # We need the time — try to get it from the plotfile
    t = 0.3  # default
    try:
        with h5py.File(args.plotfile, "r") as f:
            t = float(f.attrs.get("time", 0.3))
    except Exception:
        pass

    h_exact, u_exact = exact_dam_break_solution(x_coords, t, args.h_L, args.h_R)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Water depth
    ax1.plot(x_coords, h_yavg, "b-", linewidth=1.5, label="Simulation (y-averaged)")
    ax1.plot(x_coords, h_exact, "r--", linewidth=2, label="Exact solution")
    ax1.axvline(x=128.0, color="gray", linestyle=":", alpha=0.5, label="Initial dam location")
    ax1.set_ylabel("Water depth h [m]")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Rectangular Dam Break — h(x, t={t:.3f}s)")

    # Velocity
    u_key = None
    for key in ["hu", "Cell1", "Cell_1"]:
        if key in data:
            u_key = key
            break
    if u_key:
        u_arr = data[u_key]
        u_yavg = np.where(h_yavg > 1e-6, h_yavg / np.where(h_yavg > 1e-6, u_arr.mean(axis=0), 1e-12), 0.0)
        # Actually hu is the momentum, so u = hu/h
        u_mom = u_arr.mean(axis=0) if u_arr.ndim == 2 else u_arr[0]
        u_yavg = np.where(h_yavg > 1e-6, u_mom / h_yavg, 0.0)
        ax2.plot(x_coords, u_yavg, "b-", linewidth=1.5, label="Simulation (y-averaged)")
        ax2.plot(x_coords, u_exact, "r--", linewidth=2, label="Exact solution")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("x-velocity u [m/s]")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to: {args.output}")

    # Mass conservation check
    total_mass_sim = np.sum(h_yavg) * geom["dx"]
    total_mass_exact = args.h_L * 128.0 + args.h_R * 128.0  # initial mass
    mass_error = abs(total_mass_sim - total_mass_exact) / total_mass_exact * 100
    print(f"\nMass conservation:")
    print(f"  Initial mass:  {total_mass_exact:.4f} m^2")
    print(f"  Current mass:  {total_mass_sim:.4f} m^2")
    print(f"  Error:         {mass_error:.4f}%")

    # Y-asymmetry check (should be near zero for a correct 1D flux)
    h_std_per_y = h_arr.std(axis=0) if h_arr.ndim == 2 else h_arr.std()
    max_y_std = h_std_per_y.max()
    print(f"\nY-asymmetry (should be ~0 for correct 1D flux):")
    print(f"  Max std(h) across y at each x: {max_y_std:.6f} m")


if __name__ == "__main__":
    main()
