import PyCO2SYS as pyco2
import numpy as np


def carb_chem(alk, dic):
    """
        Converts DIC and ALK to microles/kg then
        uses pyCO2sys to solve carbonate chemistry
        returns DIC speciation, pH, omega and pCO2
        """
    # dic = []
    # alk = []
    # for i in range(3):
    #     dic.append(self.result.y[i, :])
    #     alk.append(self.result.y[i + 3, :])

    carbon_chemistry = pyco2.sys(par1=alk, par2=dic, par1_type=1, par2_type=2)
    values = ["HCO3", "CO3", "CO2", "pH", "saturation_calcite", "pCO2", "k_CO2"]
    carbonate_results = []
    for term in values:
        carbonate_results.append(carbon_chemistry[term])
    return np.array(carbonate_results)
