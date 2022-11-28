import numpy as np
import src.conversions as conversions


def circ(
    num_box, num_bc, advection, mix_np_baja, mix_baja_gulf, mix_gulf_gulf, mix_gulf_np
):
    """function that takes in circulation (units Sv) and populates a circulation matrix
    with the size based on model boxes plus boundary conditions"""

    advect = np.zeros((num_box + num_bc, num_box + num_bc))
    advect[1, 0] = advection
    advect[2, 1] = advection
    advect[4, 2] = advection
    advect[0, 3] = advection

    mix_n_b = np.zeros((num_box + num_bc, num_box + num_bc))
    mix_n_b[0, 3] = mix_np_baja
    mix_n_b[3, 0] = mix_np_baja

    mix_b_g = np.zeros((num_box + num_bc, num_box + num_bc))
    mix_b_g[1, 0] = mix_baja_gulf
    mix_b_g[0, 1] = mix_baja_gulf

    mix_g_g = np.zeros((num_box + num_bc, num_box + num_bc))
    mix_g_g[2, 1] = mix_gulf_gulf
    mix_g_g[1, 2] = mix_gulf_gulf

    mix_g_n = np.zeros((num_box + num_bc, num_box + num_bc))
    mix_g_n[4, 2] = mix_gulf_np
    mix_g_n[2, 4] = mix_gulf_np

    # print("Advect: " + str(advection))
    # print("Mix between NP-I and Baja: " + str(mix_n_b))
    # print("Mix between Baja and Gulf: " + str(mix_b_g))
    # print("Mix between Gulf-D and Gulf-S: " + str(mix_g_g))
    # print("Mix between Gulf-S and NP-S: " + str(mix_g_n))

    return advect + mix_n_b + mix_b_g + mix_g_g + mix_g_n


def make_transport_matrix(num_box, num_bc, svedrup_matrix, mass):
    """makeTM() returns a NxN matrix defining the fractional mixing system of equations,
    representing 1 year of ocean circulation
    Function inputs:
    1. m: ocean box mass vector (kg) e.g. [mass_1, mass_2, mass_3] for 3 box ocean
    2. SvM: Sverdrup matrix of fluxes (Sv) e.g [[0, f1-0, f2-0], [f0-1, 0, f2-1],
    [f0-2, f1-2, 0]] where fx-y is flux from box x to box y

    This function converts SvM to mass (kg) fluxes per timestep (units = yrs) and the mass lost
    from each box is calculated as the sum of each column (sum along rows). The fraction of each
    ocean box's mass retained after moving fluxes is given by the diagonal of the transport
    matrix. Unique transport matrices are needed for concentration and inventory fluxes
    (TM_ForConcentrations,TM_ForInventories). The difference in the transport matrices is the
    definition of "fractional fluxes" which describe the transport from one box to another with
    respect to the size (mass) of the receiving box (for concentration) or with respect to the
    giving box (for inventory). The new concentration of a given box is equal to the sum of the
    fractions of contributing boxes multiplied by their respective concentrations (i.e. mixing
    equation where the new concentration of box0 = fraction of box0 remaining * concentration
    of box0 + fraction of box0 contributed by box1 * concentration of box1). The new inventory
    of a given box is equal to the sum of the contributions from all boxes (i.e. the new
    inventory of box0 = fraction of box0 remaining * box0 inventory + fraction of box1 given
    to box0 * box1 inventory)
    TM_ForConcentrations is NxN matrix defining the fractional mixing system of equations for
    concentration units, representing 1 year of ocean circulation
    TM_ForInventories is NxN matrix defining the fractional mixing system of equations for
    inventory units, representing 1 year of ocean circulation
    """

    time_step = 1  # timestep (yr)
    flux = conversions.svedrup_to_kg_year(svedrup_matrix) * time_step
    mass_lost = np.sum(flux, axis=0)  # sum of all mass fluxes out of each box

    # fraction of mass retained in each box
    fraction_retained = (mass - mass_lost) / mass
    # wouldnt this be kg / kg ??
    # divide flux array rows by mass for concentration
    fractional_fluxes = flux / mass.reshape((len(mass), 1))
    fractional_fluxes_inv = (
        flux / mass.T
    )  # divide flux array columns by mass for inventory
    transport_matrix_concentrations = fractional_fluxes + np.diag(fraction_retained)
    transport_matrix_inventories = fractional_fluxes_inv + np.diag(fraction_retained)
    # print("Transport Matrix: " + str(transport_matrix_concentrations))
    # print("Returned: " + str(transport_matrix_concentrations - np.identity(num_box + num_bc)))
    # print(
    #     "Transport Matrix: "
    #     + str(
    #         transport_matrix_concentrations
    #         - np.identity(num_box + num_bc)
    #     )
    # )
    return transport_matrix_concentrations - np.identity(num_box + num_bc)
