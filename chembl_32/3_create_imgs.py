import os
import cv2
import sys
import json
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem import Draw

import cairosvg
import subprocess

import tqdm
from tqdm import contrib as tc


# Needed because of type-casting
try:
    # Optional command-line argument starting protein index with default value 0
    starting_protein_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
except:
    pass


def save_comp_imgs_from_smiles(training_files_path, tar_id, comp_id, smiles, rot=0, SIZE=200, rot_size=300):
    mol = Chem.MolFromSmiles(smiles)
    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 1.5
    # Use MolToFile(mol, path, size, imageType="png", fitImage=True)
    
    # For higher quality of image
    path_to_give_svg = os.path.join(training_files_path, "target_training_datasets", 
                                tar_id, "imgs", "{}.svg".format(comp_id))
    
    path_to_give_png = os.path.join(training_files_path, "target_training_datasets", 
                                    tar_id, "imgs", "{}.png".format(comp_id))
    
    Draw.MolToFile(mol, path_to_give_svg , size = (SIZE, SIZE ))
    cairosvg.svg2png(url = path_to_give_svg, write_to = path_to_give_png)
    subprocess.call(["rm", path_to_give_svg])
    
    # Make it larger with padding to prevent data loss while rotation
    image = cv2.imread(path_to_give_png)
    
    white_color = (255,255,255)
    full_image = np.full((rot_size, rot_size, 3), white_color, dtype = np.uint8)
    # compute center offset
    gap = rot_size - SIZE
    (cX, cY) = (gap // 2, gap // 2)
    
    # copy image into center of result image
    full_image[cY:cY + SIZE, cX:cX + SIZE] = image
    
    if rot != 0:
        # Rotate it
        (cX, cY) = (rot_size // 2, rot_size // 2)
        M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
        full_image = cv2.warpAffine(full_image, M, (rot_size, rot_size), borderMode=cv2.INTER_LINEAR, #cv2.BORDER_CONSTANT, 
                                    borderValue = white_color)
    
    # save result
    cv2.imwrite(path_to_give_png, full_image)
    
    
    
code_path = os.getcwd()
project_file_path = code_path.split("chembl")[0]
training_files_path = os.path.join(project_file_path, "training_files")

chembl_version = code_path.split("chembl")[1].strip("_")
deepscreen_version = code_path.split("DEEPScreen")[1].split("/")[0]
media_project_file_path = f"/media/ubuntu/8TB/hayriye/DEEPScreen{deepscreen_version}"

media_training_files_path = os.path.join(media_project_file_path, "training_files")

protein_list = pd.read_csv(training_files_path + f"/chembl{chembl_version}_training_target_list.txt", header=None)
protein_list = [line[0] for line in protein_list.values.tolist()]
print(len(protein_list), protein_list[:5])


focus_proteins = ["CHEMBL4282", "CHEMBL4683", "CHEMBL284", "CHEMBL2409", "CHEMBL5658"]

for protein in reversed(focus_proteins):
    #i = protein_list.index(protein)
    protein_list.remove(protein)
    protein_list.insert(0, protein)
    
print(len(protein_list), protein_list[:10])

protein_list = protein_list[starting_protein_index:]
print(starting_protein_index)
print(len(protein_list), protein_list[:10])


smiles_df = pd.read_csv(training_files_path + f"/chembl_{chembl_version}_chemreps.txt", sep = "\t")
smiles_df = smiles_df[["chembl_id","canonical_smiles"]]
print(smiles_df.head())

print("smiles len", len(smiles_df))
print("smiles len drop na!", len(smiles_df.dropna()))

smiles_dict = pd.Series(smiles_df.canonical_smiles.values, index=smiles_df.chembl_id).to_dict()

print("CHEMBL370638", smiles_dict["CHEMBL370638"])


del smiles_df



ttpath = os.path.join(training_files_path, "target_training_datasets")
mediattpath = os.path.join(media_training_files_path, "target_training_datasets")
print(mediattpath)


angle_list = [str(angle) for angle in range(10,360,10)]
creating_imgs_out_file = open(os.path.join(media_project_file_path, "result_files/bash_outputs", 
                                           "3creating_imgs.out"), "a+")

sys.stdout = open(os.path.join(media_project_file_path, "result_files/bash_outputs", 
                                           "3creating_imgs.warn"), "a+") # RDKit warnings to a separate file.

for ip, protein_chembl_id in tc.tenumerate(protein_list, tqdm_class=tqdm.auto.tqdm, 
                                           position=0, leave=True, ascii=True, ncols=150,
                                          desc = "Main loop for 790 proteins' train_val_test_compounds' lists..."):
#for ip, protein_chembl_id in enumerate(protein_list):
    
    print("i: {} Protein: {}".format(ip, protein_chembl_id), file=creating_imgs_out_file)
    
    
    """
    command = subprocess.run("df -h --output=avail /home | tail -n 1", shell=True, check=True, 
                             executable='/bin/bash', capture_output = True)
    disk_space = int(command.stdout.decode().strip().split("G")[0])
    if disk_space < 50:
        sys.exit("Disk space in /home directory is less than 50G!")
    """
    
    f = open(os.path.join(mediattpath, protein_chembl_id, "train_val_test_dict.json"), "r")
    data = json.load(f)
    f.close()
    
    current_list_of_compounds = []
    total_len = 0
    for tuple_list in data.values():

        current_len = len(tuple_list)
        total_len += current_len
        print("train_val_test_compounds_current_len: ", current_len, file=creating_imgs_out_file)

        for current_tuple in tuple_list:
            current_compound = current_tuple[0]
            current_list_of_compounds.append(current_compound)

    print("total_compounds_len: ", total_len, file=creating_imgs_out_file)
    
    current_list_of_compounds_len = len(current_list_of_compounds)
    error_count = 0
    
    saved_compound_count=0
    for i, comp_name_with_angle in tc.tenumerate(current_list_of_compounds, tqdm_class=tqdm.auto.tqdm,
                            position=1, leave=False, ascii=True, ncols=150,
                            desc = f"Creating Protein i = {ip} / {len(protein_list)} {protein_chembl_id} compounds' imgs..."):
    #for i, comp_name_with_angle in enumerate(current_list_of_compounds):
        
        comp_name = comp_name_with_angle
        rot = 0
        
        # 35/36 compounds are like CHEMBL3699688_270, 1/36 are like CHEMBL3699688
        if "_" in comp_name_with_angle:
            comp_name = comp_name_with_angle.split("_")[0]
            rot = int(comp_name_with_angle.split("_")[1])
        
        
        try:
            current_smiles = smiles_dict[comp_name]
        except:
            
            error_count += 1
            
            print("cur_comp/err_count/all: {}/{}/{}. Dictionary KeyError happened. The smiles value cannot be found for i:{}/target:{}, comp:{}".format(
                    i, error_count, current_list_of_compounds_len, ip, protein_chembl_id, comp_name_with_angle),
                 file = creating_imgs_out_file)
            
            continue
        
        save_comp_imgs_from_smiles(media_training_files_path, 
                                   protein_chembl_id, comp_name_with_angle, current_smiles, rot)
        saved_compound_count += 1
        
        if (i % 100000) == 0:
            creating_imgs_out_file.write(
                "cur_comp/err_count/all: {}/{}/{} for i:{}/protein:{}, compound:{}, smiles:{}\n".format(
                    i, error_count, current_list_of_compounds_len, ip, protein_chembl_id, comp_name_with_angle, current_smiles))
        
    print("Saved compound count for i:{}/{}: {}/{}".format(ip, protein_chembl_id, saved_compound_count, total_len),
          file=creating_imgs_out_file)
    
creating_imgs_out_file.close()


