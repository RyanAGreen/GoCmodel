import numpy as np
import pandas as pd

# fix years
def fix(x):
    return x / 1000


def landchange_shade(df):
    col1 = []
    col2 = []
    for val in df["ForestUptakeRate"]:
        if val > 0.0:
            col1.append("green")
    for val in df["PermafrostReleaseRate"]:
        if val > 0.0:
            col1.append("red")
    return col1, col2
    
def organize_CYCLOPS_data(NP,NP_LC,NP_LC_PF,NP_LC_PF_RF,control,control_RC):
    # feed in data as dataframes
    experiments = [NP,NP_LC,NP_LC_PF,NP_LC_PF_RF,control,control_RC]
    names = ["NP", "NP+LC","NP+LC+PF","NP+LC+PF+RC","Control","Control+RC"]
    
    for i in range(len(experiments)): 
        # drop the ones that dont matter
        experiments[i] = experiments[i].drop([6, 7,8,9,10,11,13,14,15,16,17,18], axis=1)
        # reorganize 
        experiments[i] = experiments[i][[0,1,3,2,23,24,25,26,4,5,12,19,20,21,22]]
        experiments[i] = experiments[i].rename(
            columns={
                0: "year_kyrBP",
                1: "geologic_carbon_rate_PgCyr",
                2: "alk_to_dic_ratio",
                3: "geologic_carbon_cumulative_PgC",
                4: "atmospheric_CO2_ppm",
                5: "atmospheric_∆14C_permil",
                12: "intNP_∆14C_permil",
                19: "deep_atlantic_CO3_umolkg",
                20: "deep_indian_CO3_umolkg",
                21: "deep_south_pacific_CO3_umolkg",
                22: "deep_north_pacific_CO3_umolkg",
                23: "terrestrial_carbon_uptake_rate_PgCyr",
                24: "terrestrial_carbon_uptake_cumulative_PgC",
                25: "terrestrial_carbon_release_rate_PgCyr",
                26: "terrestrial_carbon_release_cumulative_PgC",
            }
        )
        experiments[i]["year_kyrBP"] = experiments[i]["year_kyrBP"]/1000
        experiments[i].to_csv("data/model/"+ names[i] + '.txt', sep='\t', index=False)
        
    return 

def decompose(df):
    # set the total amount of each
    df["PgCO2"] = 0
    df["PgHCO3"] = 0
    df["PgCO3"] = 0
    df.loc[df["ALKtoDIC"] <= 1, "PgCO2"] = (1 - df["ALKtoDIC"]) * (df["Crate"] * 100)
    df.loc[df["ALKtoDIC"] <= 1, "PgHCO3"] = df["ALKtoDIC"] * (df["Crate"] * 100)
    df.loc[df["ALKtoDIC"] > 1, "PgHCO3"] = (2 - df["ALKtoDIC"]) * (df["Crate"] * 100)
    df.loc[df["ALKtoDIC"] > 1, "PgCO3"] = (df["ALKtoDIC"] - 1) * (df["Crate"] * 100)
    df["PgCO2rate"] = df["PgCO2"] / 100
    df["PgHCO3rate"] = df["PgHCO3"] / 100
    df["PgCO3rate"] = df["PgCO3"] / 100
    # get the cum sum through time
    df["PgCO2cumsum"] = (df["PgCO2"].cumsum()).round(0)
    df["PgHCO3cumsum"] = (df["PgHCO3"].cumsum()).round(0)
    df["PgCO3cumsum"] = (df["PgCO3"].cumsum()).round()

    return df


def binning(df):
    # first pulse
    # test1 = df[((df.year < 20) & (df.year > 14.5))]
    # firstpulse_mean = (test1['ALKtoDIC'] * test1['Crate']).sum() / test1['Crate'].sum()
    firstpulse_cum = df.Ccum[55] - df.Ccum[20]
    # first = (firstpulse_cum,firstpulse_mean)

    # second pulse
    # test2 = df[((df.year < 14.5) & (df.year > 10))]
    # secondpulse_mean = (test2['ALKtoDIC'] * test2['Crate']).sum() / test2['Crate'].sum()
    secondpulse_cum = df.Ccum[100] - df.Ccum[55]
    # second = (secondpulse_cum,secondpulse_mean)
    # third pulse
    thirdpulse_cum = df.Ccum[200] - df.Ccum[100]

    # try:
    #     test3 = df[((df.year < 5) & (df.year > 2.5))]
    #     # thirdpulse_mean = (test3['ALKtoDIC'] * test3['Crate']).sum() / test3['Crate'].sum()
    #     thirdpulse_cum = df.Ccum[175]-df.Ccum[150]
    #     third = (thirdpulse_cum,thirdpulse_mean)
    # except:
    #     third = (0,0)
    return firstpulse_cum, secondpulse_cum, thirdpulse_cum


def binning_species(df):
    # first pulse
    pulses_CO2 = [
        df.PgCO2[55] - df.PgCO2[20],
        df.PgCO2[100] - df.PgCO2[55],
        df.PgCO2[200] - df.PgCO2[100],
    ]
    pulses_HCO3 = []
    pulses_CO3 = []


def D14Cscore(df, control):
    D14Cerror_ind = np.argmax(df["D14Cerror"])
    D14Cerror = df["D14Cerror"][D14Cerror_ind]
    control_D14Cerror_ind = np.argmax(control["D14Cerror"])
    controlD14Cerror = control["D14Cerror"][control_D14Cerror_ind]
    score = (1 - ((controlD14Cerror - D14Cerror) / controlD14Cerror)) * 100
    return score


def CO2score(df, control):
    CO2error_ind = np.argmax(df["CO2error"])
    CO2error = df["CO2error"][CO2error_ind]
    control_CO2error_ind = np.argmax(control["CO2error"])
    controlCO2error = control["CO2error"][control_CO2error_ind]
    score = (1 - ((controlCO2error - CO2error) / controlCO2error)) * 100
    return score


def max(df):
    Ccum_ind1 = np.argmax(df["Ccum"])
    maximum = df["Ccum"][Ccum_ind1]
    return maximum


def organizedata(df):
    df = df.rename(
        columns={
            0: "year",
            1: "Crate",
            2: "ALKtoDIC",
            3: "Ccum",
            4: "CO2",
            5: "D14C",
            6: "D14Cerror",
            7: "CO2error",
            8: "totalerror",
            9: "DICintNP",
            10: "ALKintNP",
            11: "d13CintNP",
            12: "D14CintNP",
            13: "NintNP",
            14: "DICsurfNP",
            15: "ALKsurfNP",
            16: "d13CsurfNP",
            17: "D14CsurfNP",
            18: "NsurfNP",
            19: "AtlCO3",
            20: "IndCO3",
            21: "SPacCO3",
            22: "NPacCO3",
            23: "ForestUptakeRate",
            24: "CumForestUptake",
            25: "PermafrostReleaseRate",
            26: "CumPermafrostRelease",
        }
    )
    df["year"] = df["year"] / 1000
    return df


def organizedata_goc(df):
    df = df.rename(
        columns={
            0: "year",
            1: "DIC_mar",
            2: "DIC_sub",
            3: "DIC_surf",
            4: "ALK_mar",
            5: "ALK_sub",
            6: "ALK_surf",
            7: "d13C_mar",
            8: "d13C_sub",
            9: "d13C_surf",
            10: "D14C_mar",
            11: "D14C_sub",
            12: "D14C_surf",
            13: "pH_mar",
            14: "pH_sub",
            15: "pH_surf",
            16: "Ccum_mar",
            17: "Ccum_sub",
            18: "Ccum_surf",
            19: "Crate_mar",
            20: "Crate_sub",
            21: "Crate_surf",
        }
    )
    # df["year"] = df["year"] / 1000
    return df


def readD14C(path):
    result = []
    for i in range(19):
        test = float(i * 0.1)
        testresult = path + "1Dinversion/D14C_1Dmin_{:.1f}.txt".format(test)
        testresult = pd.read_fwf(testresult, header=None, infer_nrows=1000)
        result.append(testresult)
        result[i] = result[i].rename(
            columns={
                0: "year",
                1: "Crate",
                2: "ALKtoDIC",
                3: "Ccum",
                4: "CO2",
                5: "D14C",
                6: "D14Cerror",
                7: "CO2error",
                8: "totalerror",
                9: "DICintNP",
                10: "ALKintNP",
                11: "d13CintNP",
                12: "D14CintNP",
                13: "NintNP",
                14: "DICsurfNP",
                15: "ALKsurfNP",
                16: "d13CsurfNP",
                17: "D14CsurfNP",
                18: "NsurfNP",
                19: "AtlCSH",
                20: "IndCSH",
                21: "SPacCSH",
                22: "NPacCSH",
            }
        )
        result[i]["year"] = result[i]["year"].apply(fix)
    return result


def read1Dboth(path):
    result = []
    for i in range(21):
        test = float(i * 0.1)
        testresult = path + "1Dinversion/Powell1D_2constraints_{:.1f}.txt".format(test)
        testresult = pd.read_fwf(testresult, header=None, infer_nrows=1000)
        result.append(testresult)
        result[i] = result[i].rename(
            columns={
                0: "year",
                1: "Crate",
                2: "ALKtoDIC",
                3: "Ccum",
                4: "CO2",
                5: "D14C",
                6: "D14Cerror",
                7: "CO2error",
                8: "totalerror",
                9: "DICintNP",
                10: "ALKintNP",
                11: "d13CintNP",
                12: "D14CintNP",
                13: "NintNP",
                14: "DICsurfNP",
                15: "ALKsurfNP",
                16: "d13CsurfNP",
                17: "D14CsurfNP",
                18: "NsurfNP",
                19: "AtlCSH",
                20: "IndCSH",
                21: "SPacCSH",
                22: "NPacCSH",
            }
        )
        result[i]["year"] = result[i]["year"].apply(fix)
    return result


def readD14C_full(path):
    result = []
    for i in range(21):
        test = float(i * 0.1)
        testresult = path + "1Dinversion/D14C_1Dmin_{:.1f}.txt".format(test)
        testresult = pd.read_fwf(testresult, header=None, infer_nrows=1000)
        result.append(testresult)
        result[i] = result[i].rename(
            columns={
                0: "year",
                1: "Crate",
                2: "ALKtoDIC",
                3: "Ccum",
                4: "CO2",
                5: "D14C",
                6: "D14Cerror",
                7: "CO2error",
                8: "totalerror",
                9: "DICintNP",
                10: "ALKintNP",
                11: "d13CintNP",
                12: "D14CintNP",
                13: "NintNP",
                14: "DICsurfNP",
                15: "ALKsurfNP",
                16: "d13CsurfNP",
                17: "D14CsurfNP",
                18: "NsurfNP",
                19: "AtlCSH",
                20: "IndCSH",
                21: "SPacCSH",
                22: "NPacCSH",
            }
        )
        result[i]["year"] = result[i]["year"].apply(fix)
    return result


def readCO2(path):
    result = []
    for i in range(14):
        test = float(i * 0.1)
        testresult = path + "1Dinversion/CO2_1Dmin_{:.1f}.txt".format(test)
        testresult = pd.read_fwf(testresult, header=None, infer_nrows=1000)
        result.append(testresult)
        result[i] = result[i].rename(
            columns={
                0: "year",
                1: "Crate",
                2: "ALKtoDIC",
                3: "Ccum",
                4: "CO2",
                5: "D14C",
                6: "D14Cerror",
                7: "CO2error",
                8: "totalerror",
                9: "DICintNP",
                10: "ALKintNP",
                11: "d13CintNP",
                12: "D14CintNP",
                13: "NintNP",
                14: "DICsurfNP",
                15: "ALKsurfNP",
                16: "d13CsurfNP",
                17: "D14CsurfNP",
                18: "NsurfNP",
                19: "AtlCSH",
                20: "IndCSH",
                21: "SPacCSH",
                22: "NPacCSH",
            }
        )
        result[i]["year"] = result[i]["year"].apply(fix)
    return result


def calc_high_score(list):
    max_ind = np.argmax(list)
    max_score = list[max_ind]
    return max_ind, max_score


def magnitude(df):
    # df must be in 20.0 - 0.0 years
    # cut into 2.5k intervals
    q1 = df[(df.year <= 20) & (df.year >= 17.5)]
    q2 = df[(df.year <= 17.5) & (df.year >= 15)]
    q3 = df[(df.year <= 15) & (df.year >= 12.5)]
    q4 = df[(df.year <= 12.5) & (df.year >= 10)]
    q5 = df[(df.year <= 10) & (df.year >= 7.5)]
    q6 = df[(df.year <= 7.5) & (df.year >= 5)]
    q7 = df[(df.year <= 5) & (df.year >= 2.5)]
    q8 = df[(df.year <= 2.5) & (df.year >= 0)]

    qs = [q1, q2, q3, q4, q5, q6, q7, q8]

    # calculate range for each interval
    qrange = []
    for i in range(len(qs)):
        Range = np.max(qs[i].D14CintNP) - np.min(qs[i].D14CintNP)
        newrange = np.nan_to_num(Range)
        qrange.append(newrange)

    # store and return maximum range
    mag = np.max(qrange)
    return mag, qrange


def w_avg(df, values, weights):
    d = df.values
    w = df.weights
    return (d * w).sum() / w.sum()
