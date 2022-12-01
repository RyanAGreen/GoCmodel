import numpy as np


def carbon_add(num_tracer, num_box, rate, box_name, mass_array,geologic_d13c):
    """geologic carbon addition"""

    if box_name == "marchitto":
        i = 0
    elif box_name == "subsurface":
        i = 1
    elif box_name == "surface":
        i = 2

    carbon_flux = rate * 1e15 / 12 * 1e6 / mass_array[i]

    d_dt = np.zeros((num_tracer, num_box))
    # DIC
    d_dt[0, i] = carbon_flux
    # ALK
    d_dt[1, i] = 1 * carbon_flux
    # d13C
    d_dt[3, i] = geologic_d13c * carbon_flux
    # D14C
    d_dt[4, i] = -1000 * carbon_flux
    # Cum Carbon
    d_dt[5, i] = rate

    return d_dt
