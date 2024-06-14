import os
import json
import cv2
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from concurrent.futures import ProcessPoolExecutor
#import time

"""prediction_files_path = "/Users/furkannecatiinan/DeepScreen/DEEPScreen2"
target_prediction_dataset_path = "/Users/furkannecatiinan/DeepScreen/Optimizasyon1/pythonProject2/target_prediction_dataset_druggen/"
smiles_file = "/Users/furkannecatiinan/DeepScreen/Optimizasyon1/pythonProject2/shortexample.csv"
protein_name = "AKT"
"""
current_path_beginning = os.getcwd().split("DEEPScreen")[0]
current_path_version = os.getcwd().split("DEEPScreen")[1].split("/")[0]



prediction_files_path = "{}DEEPScreen{}".format(current_path_beginning, current_path_version)
target_prediction_dataset_path = prediction_files_path + "target_prediction_dataset_druggen/"
smiles_path = prediction_files_path + "molecule_smiles_dataset/"
smiles_file = prediction_files_path + "molecule_smiles_dataset/"
protein_name = "AKT"

def save_comp_imgs_from_smiles(tar_id, comp_id, smiles, rotations, SIZE=300, rot_size=300):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    Draw.DrawingOptions.atomLabelFontSize = 55
    Draw.DrawingOptions.dotsPerAngstrom = 100
    Draw.DrawingOptions.bondLineWidth = 1.5

    base_path = os.path.join(target_prediction_dataset_path, tar_id, "imgs")

    try:
        image = Draw.MolToImage(mol, size=(SIZE, SIZE))
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        for rot, suffix in rotations:
            if rot != 0:
                full_image = np.full((rot_size, rot_size, 3), (255, 255, 255), dtype=np.uint8)
                gap = rot_size - SIZE
                (cX, cY) = (gap // 2, gap // 2)
                full_image[cY:cY + SIZE, cX:cX + SIZE] = image_bgr
                (cX, cY) = (rot_size // 2, rot_size // 2)
                M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)
                full_image = cv2.warpAffine(full_image, M, (rot_size, rot_size), borderMode=cv2.INTER_LINEAR,
                                            borderValue=(255, 255, 255))
            else:
                full_image = image_bgr

            path_to_save = os.path.join(base_path, f"{comp_id}{suffix}.png")
            cv2.imwrite(path_to_save, full_image)
    except Exception as e:
        print(f"Error creating PNG for {comp_id}: {e}")

def initialize_dirs(protein_name):
    if not os.path.exists(os.path.join(target_prediction_dataset_path, protein_name, "imgs")):
        os.makedirs(os.path.join(target_prediction_dataset_path, protein_name, "imgs"))

    f = open(os.path.join(target_prediction_dataset_path, protein_name, "prediction_dict.json"), "w+")
    json_dict = {"prediction": list()}
    json_object = json.dumps(json_dict)
    f.write(json_object)
    f.close()

def process_smiles(smiles_data):
    current_smiles, compound_id = smiles_data
    rotations = [(0, ""), *[(angle, f"_{angle}") for angle in range(10, 360, 10)]]
    save_comp_imgs_from_smiles(protein_name, compound_id, current_smiles, rotations)

def generate_images(smiles_file, protein_name):
    smiles_list = pd.read_csv(smiles_file)["smiles"].tolist()
    compound_prefix = "GANt"
    compound_ids = [compound_prefix + str(i) for i in range(len(smiles_list))]
    smiles_data_list = list(zip(smiles_list, compound_ids))
    
    #start_time = time.time()
    with ProcessPoolExecutor() as executor:
        executor.map(process_smiles, smiles_data_list)
    #end_time = time.time()
    
    #print(f"Time taken for all: {end_time - start_time}")
    total_image_count = len(smiles_list) * len([(0, ""), *[(angle, f"_{angle}") for angle in range(10, 360, 10)]])
    print(f"Total images generated: {total_image_count}")

# Örnek fonksiyon çağrısı
if __name__ == "__main__":
    initialize_dirs(protein_name)
    generate_images(smiles_file, protein_name)
