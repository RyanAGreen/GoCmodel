import os
import pandas as pd

# bc_path_from_CY3_into_GoC = "../GoCmodel/data/ISchange/ForwardRun/CoupledRun.txt"
# bc_path_from_CY3 = "OUTPUT/GlobalCadd/ISchange/ForwardRun/CoupledRun.txt"
# rate_path_from_GoC_into_CY3 = "../CY3/FORCING/Project1/total_GoC_rates.txt"
# rate_path_from_GoC = "results/total_GoC_rates.txt"


def CheckRate():
    df = pd.read_table(
        "~/GoCmodel/results/optimizedrun_CO2carbonate_source.txt",
        sep="\s+",
        header=None,
    )
    total_carbon_added = df[13][200] + df[14][200] + df[15][200]
    print("TOTAL CARBON IS: ", total_carbon_added)
    return total_carbon_added


# Run GoC model the first time
print("Running initial GoC simulation \n")
os.system("python scripts/GoCmodel.py")

new_total_carbon = CheckRate()
old_total_carbon = 0
counter = 0

print("Starting for loop...")
if abs(new_total_carbon - old_total_carbon) > 1:
    old_total_carbon = CheckRate()

    print("Run CYCLOPS with GoC rates \n ")
    # Move GoC rates into CYCLOPS
    os.system(
        "cp results/total_GoC_rates.txt ../CY3/FORCING/Project1/total_GoC_rates.txt"
    )
    # Change directory into CY3
    os.chdir("../CY3")
    # Run CY3
    os.system("make clean runex ExString=CoupledRun")
    # Move CY3 boundary condition into GoC
    os.system(
        "cp OUTPUT/GlobalCadd/ISchange/ForwardRun/CoupledRun.txt ../GoCmodel/data/ISchange/ForwardRun/CoupledRun.txt"
    )

    ### Run GoC with the new CYCLOPS output ###
    # Change directory into GoC
    os.chdir("../GoCmodel")
    print("Run GoC with new boundary conditions \n")
    # Rerun GoC model
    os.system("python scripts/GoCmodel.py")

    new_total_carbon = CheckRate()
    counter += 1
    print("Loop #: ", counter)


print("Finished the model coupling!!!!")
