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
    return df[13][200] + df[14][200] + df[15][200]


# Run GoC model the first time
print("Running GoC model")
os.system("python scripts/GoCmodel.py")

new_total_carbon = CheckRate()
old_total_carbon = 0
counter = 0

if abs(new_total_carbon - old_total_carbon) > 1:
    old_total_carbon = CheckRate()
    print("LAST RUN TOTAL CARBON : ", old_total_carbon)

    # Move GoC rates into CYCLOPS
    print("Moving the GoC rates into the CY3 folder")
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
    # Rerun GoC model
    os.system("python scripts/GoCmodel.py")

    new_total_carbon = CheckRate()
    print("NEW RUN TOTAL CARBON : ", new_total_carbon)
    counter += 1
    print("Loop #: ", counter)


print("Finished the model coupling!!!!")
