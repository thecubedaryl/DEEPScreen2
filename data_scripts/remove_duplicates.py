import pandas as pd
from operator import itemgetter

def create_preprocessed_bioact_file(chembl_filtered_bioact_fl, chembl_version):
    training_files_path = "/home/hayriye/Documents/DEEPScreen2.0/training_files"
    raw_dataset_df = pd.read_csv("{}/{}".format(training_files_path, chembl_filtered_bioact_fl), sep="\t", index_col=False, low_memory=False)
    # keys are compound protein pairs values are list of bioactivities
    annot_dict = dict()
    for ind, row in raw_dataset_df.iterrows():
        chembl_tid = row["Target_CHEMBL_ID"]
        chembl_cid = row["Compound_CHEMBL_ID"]
        standard_units = row["standard_units"]
        row["year"] = row["year"]

        if standard_units in ["uM", "nM", "M"]:
            if standard_units == "nM":
                row["standard_value"] = round(row["standard_value"] / pow(10,3), 3)
            elif standard_units == "M":
                row["standard_value"] = round(row["standard_value"] / pow(10,6), 3)
            else:
                row["standard_value"] = round(row["standard_value"], 3)
            # standard_units = "uM"
            row["standard_units"] = "uM"
            try:
                annot_dict["{},{}".format(chembl_tid, chembl_cid)].append(list(row))
            except:
                annot_dict["{},{}".format(chembl_tid, chembl_cid)] = [list(row)]

    out_fl = open("{}/{}_preprocessed_filtered_bioactivity_dataset.tsv".format(training_files_path, chembl_version), "w")
    out_fl.write("\t".join(list(raw_dataset_df.columns)))
    for key in annot_dict.keys():
        if len(annot_dict[key])>1:
            median_std_val = 0.0

            annot_dict[key]  = sorted(annot_dict[key], key=itemgetter(6))
            # print(annot_dict[key])

            if len(annot_dict[key])%2==1:
                median = int(len(annot_dict[key])/2)
                median_bioactivity = annot_dict[key][median]
                out_fl.write("\n" + "\t".join([str(col) for col in median_bioactivity]))

            else:
                median = int(len(annot_dict[key])/2)

                median_std_val = (annot_dict[key][median][6]+annot_dict[key][median-1][6])/2
                annot_dict[key][median][6] = median_std_val
                median_bioactivity = annot_dict[key][median]
                out_fl.write("\n" + "\t".join([str(col) for col in median_bioactivity]))
        else:

            out_fl.write("\n"+"\t".join([str(col) for col in annot_dict[key][0]]))

    out_fl.close()

import sys
"""
if len(sys.argv) != 2:
    raise ValueError('Please provide parameters.')
    """

print(f'Script Name is {sys.argv[0]}')

version = "chembl28"
chembl_filtered_bioact_fl = "chembl28_filtered_bioact_fl.tsv"
    
create_preprocessed_bioact_file(chembl_filtered_bioact_fl, version)


