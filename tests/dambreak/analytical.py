import numpy as np
import functools
from scipy.optimize import fsolve

g = 9.81 #gravity

def solve_polynomial_numerically(hl, hr):
    # Fallback to fsolve if polynomial root finding fails or selects an unphysical one
    def equation_for_cm(cm, g, hlt, hrt):

        cr = np.sqrt(g*hrt)
        cl = np.sqrt(g*hlt)

        term1 = -8 * cr**2 * cm**2 * (cl - cm)**2
        term2 = (cm**2 - cr**2)**2 * (cm**2 + cr**2)
        return term1 + term2

    func_to_solve_cm = functools.partial(equation_for_cm, g=g, hlt=hl, hrt=hr)
    initial_guess_cm = (np.sqrt(g * hl) + np.sqrt(g * hr))
    c_m = fsolve(func_to_solve_cm, initial_guess_cm)
    print("Obtained c_m: {}".format(c_m))

    return c_m[0]

def dambreak_on_wet_no_friction_analytical(t, x, L=10, hl=0.005, hr=0.001, x0=5):
    # Specifically designed to test SWE solver only
    # SWASHES

    cm = solve_polynomial_numerically(hl, hr)

    xat = lambda t: x0 - t*np.sqrt(g*hl)
    xbt = lambda t: x0 + t*(2*np.sqrt(g*hl)-3*cm)
    xct = lambda t: x0 + t*(2*cm**2*(np.sqrt(g*hl)-cm))/(cm**2-g*hr)

    h_1 = lambda t, x: hl
    h_2 = lambda t, x: (4/(9*g))*(np.sqrt(g*hl)-(x - x0)/(2*t))**2
    h_3 = lambda t, x: cm**2/g
    h_4 = lambda t, x: hr

    u_1 = lambda t, x: 0
    u_2 = lambda t, x: (2/3)*((x-x0)/t + np.sqrt(g*hl))
    u_3 = lambda t, x: 2*(np.sqrt(g*hl)-cm)
    u_4 = lambda t, x: 0

    h = np.where(
        x <= xat(t), h_1(t,x), np.where(
            (xat(t) <= x) & (x <= xbt(t)), h_2(t,x), np.where(
                (xbt(t) <= x) & (x <= xct(t)), h_3(t,x), np.where(
                    xct(t) <= x, h_4(t,x), h_4(t,x)
                )
            )
        )
    )

    u = np.where(
        x <= xat(t), u_1(t,x), np.where(
            (xat(t) <= x) & (x <= xbt(t)), u_2(t,x), np.where(
                (xbt(t) <= x) & (x <= xct(t)), u_3(t,x), np.where(
                    xct(t) <= x, u_4(t,x), u_4(t,x)
                )
            )
        )
    )

    return h, u
