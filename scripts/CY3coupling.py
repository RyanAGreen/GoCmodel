import os
import pandas as pd

# bc_path_from_CY3_into_GoC = "../GoCmodel/data/ISchange/ForwardRun/CoupledRun.txt"
# bc_path_from_CY3 = "OUTPUT/GlobalCadd/ISchange/ForwardRun/CoupledRun.txt"
# rate_path_from_GoC_into_CY3 = "../CY3/FORCING/Project1/total_GoC_rates.txt"
# rate_path_from_GoC = "results/total_GoC_rates.txt"


def CheckRate(filename):
    df = pd.read_table(
        "~/GoCmodel/results/simulations/" + filename,
        sep="\s+",
        header=None,
    )
    total_carbon_added = df[16][200] + df[17][200] + df[18][200]
    print("TOTAL CARBON IS: ", total_carbon_added)
    return total_carbon_added


print(
    "Running optimization with no carbon added to CYCLOPS ('CYCLOPS_control') as the boundary condition\n"
)
os.system('python scripts/GoCmodel.py "CYCLOPS_control"')

print(
    "Checking carbon rate from GoC simulation with CYCLOPS control boundary condition \n"
)
# moving GoC rates to CYCLOPS
os.system(
    "cp results/simulations/total_GoC_rates.txt ../CY3/FORCING/Project1/total_GoC_rates.txt"
)


new_total_carbon = CheckRate("d13c--1_ALK_DIC-1_optimization_CYCLOPS_control.txt")
old_total_carbon = 0
counter = 0

print("STARTING WHILE LOOP...")
while abs(new_total_carbon - old_total_carbon) > 1:
    print("Run CYCLOPS with GoC rates \n ")

    ### Run CYCLOPS with the GoC output ###
    # Change directory into CY3
    os.chdir("../CY3")
    # Run CY3
    os.system("make clean runex ExString=CoupledRun")
    # Move CY3 boundary condition into GoC
    os.system(
        "cp OUTPUT/GlobalCadd/NoISchange/ForwardRun/CoupledRun.txt ../GoCmodel/data/model/CoupledRun.txt"
    )

    ### Run GoC with the new CYCLOPS output ###
    # Change directory into GoC
    os.chdir("../GoCmodel")
    print("Run GoC with new boundary conditions \n")
    # Rerun GoC model
    os.system('python scripts/GoCmodel.py "coupled"')

    # Move GoC rates into CYCLOPS
    os.system(
        "cp results/simulations/total_GoC_rates.txt ../CY3/FORCING/Project1/total_GoC_rates.txt"
    )

    old_total_carbon = new_total_carbon
    new_total_carbon = CheckRate("d13c--1_ALK_DIC-1_optimization_coupled.txt")
    counter += 1
    print("Loop #: ", counter)


print("Finished the model coupling!!!!")
