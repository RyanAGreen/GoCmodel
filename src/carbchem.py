import PyCO2SYS as pyco2
import numpy as np


def carb_chem(results):
    """
        Converts DIC and ALK to microles/kg then
        uses pyCO2sys to solve carbonate chemistry
        returns DIC speciation, pH, omega and pCO2
        """
    # dic = []
    # alk = []
    # for i in range(3):
    #     dic.append(results[i, :])
    #     alk.append(results[i + 3, :])

    dic_bc = results[0, :]
    dic_goc_deep = results[1, :]
    dic_goc_surf = results[2, :]
    alk_bc = results[3, :]
    alk_goc_deep = results[4, :]
    alk_goc_surf = results[5, :]
    dic = [dic_bc, dic_goc_deep, dic_goc_surf]
    alk = [alk_bc, alk_goc_deep, alk_goc_surf]
    carbon_chemistry = pyco2.sys(par1=alk, par2=dic, par1_type=1, par2_type=2)
    values = ["HCO3", "CO3", "CO2", "pH", "saturation_calcite", "pCO2"]
    carbonate_results = []
    for term in values:
        carbonate_results.append(carbon_chemistry[term])
    return np.array(carbonate_results)

    # carbon_chemistry = pyco2.sys(par1=alk, par2=dic, par1_type=1, par2_type=2)
    # values = ["HCO3", "CO3", "CO2", "pH", "saturation_calcite", "pCO2", "k_CO2"]
    # # values = ["pH"]
    # carbonate_results = []
    # for term in values:
    #     carbonate_results.append(carbon_chemistry[term])
    # return np.array(carbonate_results)
