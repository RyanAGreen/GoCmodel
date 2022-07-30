import numpy as np

# Biological Productivity
def productivity(current_state, bc, num_tracer, num_box, num_bc, CaRatio, export_matrix, remin_matrix):
    idxP = 2  # index of phosphorus in the tracers array
    boxesP = [2, 4]  # GoC Surface and NP Surface

    state = np.hstack((current_state, bc))  # [6 x 5]
    P = state[idxP, :]  # [1 x 5]
    offset_value = 20
    del_13_c_cc = state[3] / state[0]
    del_13_c_org = del_13_c_cc + offset_value
    del_14_c_cc = state[4] / state[0]
    del_14_c_org = del_14_c_cc + 2 * offset_value

    exportP = np.zeros(num_box + num_bc)
    setP = np.array([0, 0, 0.001, 0, 0.001])
    timescale = 5  # year
    for box in boxesP:
        if P[box] - setP[box] > 0:
            exportP[box] = (P[box] - setP[box]) / timescale  # umol surface N/year
        else:
            pass  # not enough nutrients to sustain productivity

    exportCa = exportP * 106 * CaRatio

    productP = export_matrix @ exportP  # [5 x 5] x [5,] = [5,]
    # Let X be the amount from GoC surface to GoC subsurface
    # Let Y be the amount from NP surface to Marchitto
    # Then, ExportN will equal a column vector [0,0,X,0,Y]
    # Finally, EM @ ExportN will equal a column vector [Y,X,-X,0,Y],
    # which is correct and can be added to d_dt
    productP = productP[: num_box]
    productCa = export_matrix @ exportCa
    productCa = productCa[: num_box]

    d_dt = np.zeros((num_tracer, num_box))
    d_dt[0] += productP * 106  # Redfield ratio
    d_dt[1] += productP * -16
    d_dt[2] += productP
    d_dt[3] += productP * 106 * del_13_c_org[: num_box]
    d_dt[4] += productP * 106 * del_14_c_org[: num_box]

    d_dt[0] += productCa
    d_dt[1] += productCa * 2
    d_dt[3] += productCa * del_13_c_org[: num_box]
    d_dt[4] += productCa * del_14_c_org[: num_box]

    return d_dt, exportP, del_13_c_org, del_14_c_org

# Remineralization
def remin(exportP, del_13_c_org, del_14_c_org, num_tracer, num_box, remin_matrix):
    addOrg = remin_matrix @ exportP
    addOrg_del_13_c = remin_matrix @ (exportP * del_13_c_org)
    addOrg_del_14_c = remin_matrix @ (exportP * del_14_c_org)

    addOrg = addOrg[: num_box]
    addOrg_del_13_c = addOrg_del_13_c[: num_box]
    addOrg_del_14_c = addOrg_del_14_c[: num_box]

    d_dt = np.zeros((num_tracer, num_box))
    d_dt[0] += addOrg * 106
    d_dt[1] += addOrg * -16
    d_dt[2] += addOrg
    d_dt[3] += addOrg_del_13_c * 106
    d_dt[4] += addOrg_del_14_c * 106

    return d_dt
