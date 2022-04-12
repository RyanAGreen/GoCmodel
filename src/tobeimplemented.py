import numpy as np


class tobeimplemented:
    """everything that hasnt been put into GoC model"""

    def __init__(self):

        # yet to be put into the model
        # initialize biological pump export
        self.num_surf = 1
        self.num_interior = 2
        # Export matrix; fraction of export from surface (column) to interior (row)
        self.export_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    def production(self, state_a):
        """this could be used to calculate bio pump"""
        net_primary_prod = (
            4 * state_a[0, 0:5] * self.mass[0:5]
        )  # 1/yr * µmol/kg * kg = µmol/yr
        d_dt = np.zeros((self.num_tracer, self.num_box))
        d_dt[0, :] = self.export_matrix @ net_primary_prod
        d_dt[1, :] = self.export_matrix @ net_primary_prod / 16
        d_dt[2, :] = self.export_matrix @ (
            net_primary_prod * (state_a[2, 0:5] / state_a[0, 0:5] - self.epsi_assim)
        )
        return (
            d_dt,
            net_primary_prod * 1e-6 * 1e-12 * 14,
            (state_a[2, 0:5] / state_a[0, 0:5] - self.epsi_assim),
        )

    def export_phosphorus(self, state):
        """computes phosphorus export"""
        export_phos = np.zeros(3).T
        state = state[:, :-2]
        phos = state.reshape(3, 6)[:, 2] / self.mass[:]  # mol/kg P
        set_phos = np.array([1e-6, 1e-7])
        for surf_boxes in range(0, self.num_surf):
            timescale = 20  # year
            diff = phos[surf_boxes] - set_phos[surf_boxes]
            if diff > 0:
                export_phos[surf_boxes] = (
                    diff / timescale * self.mass[surf_boxes]
                )  # mol surfacePO4/year

            else:
                # print(P[s],SetP[s],P[s]-SetP[s])
                pass  # not enough nutrients to sustain productivity
        print(self.export_matrix)
        print(export_phos)
        return self.export_matrix @ export_phos
