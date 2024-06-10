import os
import cv2
import json
import rdkit
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw.MolDrawing import DrawingOptions
from rdkit.Chem import Draw

import cairosvg
import subprocess

prediction_files_path = "/home/atabey/DEEPScreen2.1/"
target_prediction_dataset_path = prediction_files_path + "target_prediction_dataset_druggen/"
smiles_path = prediction_files_path + "molecule_smiles_dataset/"
smiles_file = "/home/atabey/DEEPScreen2.1/target_prediction_dataset/CrossLoss_Selection_Chembl_wpreruns_smimaestro_docking_02.06.2024.csv"
protein_name = "AKT"

def save_comp_imgs_from_smiles(tar_id, comp_id, smiles, rot=0, SIZE=200, rot_size=300):
    mol = Chem.MolFromSmiles(smiles)
    DrawingOptions.atomLabelFontSize = 55
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 1.5
    # Use MolToFile(mol, path, size, imageType="png", fitImage=True)
    
    # For higher quality of image
    path_to_give_svg = os.path.join(prediction_files_path, "target_prediction_dataset_druggen", 
                                tar_id, "imgs", "{}.svg".format(comp_id))
    
    path_to_give_png = os.path.join(prediction_files_path, "target_prediction_dataset_druggen", 
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
    
    
if not os.path.exists(target_prediction_dataset_path + protein_name + "/imgs"):
    os.makedirs(target_prediction_dataset_path + protein_name + "/imgs")

f = open(target_prediction_dataset_path + protein_name + "/prediction_dict.json", "w+")

json_dict = {"prediction": list()}
json_object = json.dumps(json_dict) 

f.write(json_object)
f.close()


# removing the new line characters
#with open(smiles_file) as f:
#    smiles_list = [line.rstrip() for line in f]


smiles_list = pd.read_csv(smiles_file)["smiles"].tolist()
    
compound_prefix = "GANt"
GAN_name_count = 0
total_image_count = 0
angle_list = [str(angle) for angle in range(10,360,10)]

for current_smiles in smiles_list:
    
    compound_id = compound_prefix + str(GAN_name_count)
    
    try:
        save_comp_imgs_from_smiles(protein_name, compound_id, current_smiles)
        total_image_count += 1
        #print(total_image_count, compound_id, current_smiles)
        
        for angle in angle_list:
        
            save_comp_imgs_from_smiles(protein_name, compound_id+"_"+angle, current_smiles, int(angle))
            total_image_count += 1
            #print(total_image_count, compound_id+"_"+angle, current_smiles, int(angle))
            
    except Exception as e:
        print(e, GAN_name_count, compound_id, current_smiles)
        
    print(GAN_name_count, compound_id, current_smiles)
    GAN_name_count += 1
        