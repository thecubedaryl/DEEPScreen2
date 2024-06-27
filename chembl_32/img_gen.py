import os
import json
import cv2
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from concurrent.futures import ProcessPoolExecutor
import time
import argparse
import multiprocessing

def save_comp_imgs_from_smiles(tar_id, comp_id, smiles, rotations, target_prediction_dataset_path, SIZE=300, rot_size=300):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return

    Draw.DrawingOptions.atomLabelFontSize = 55
    Draw.DrawingOptions.dotsPerAngstrom = 100
    Draw.DrawingOptions.bondLineWidth = 1.5

    base_path = os.path.join(target_prediction_dataset_path, tar_id, "imgs")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

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

def initialize_dirs(protein_name, target_prediction_dataset_path):
    if not os.path.exists(os.path.join(target_prediction_dataset_path, protein_name, "imgs")):
        os.makedirs(os.path.join(target_prediction_dataset_path, protein_name, "imgs"))

    f = open(os.path.join(target_prediction_dataset_path, protein_name, "prediction_dict.json"), "w+")
    json_dict = {"prediction": list()}
    json_object = json.dumps(json_dict)
    f.write(json_object)
    f.close()

def process_smiles(smiles_data):
    current_smiles, compound_id, target_prediction_dataset_path, protein_name = smiles_data
    rotations = [(0, "_0"), *[(angle, f"_{angle}") for angle in range(10, 360, 10)]]
    save_comp_imgs_from_smiles(protein_name, compound_id, current_smiles, rotations, target_prediction_dataset_path)

def generate_images(smiles_file, protein_name, target_prediction_dataset_path, max_cores):
    smiles_list = pd.read_csv(smiles_file)["smiles"].tolist()
    compound_prefix = "GANt"
    compound_ids = [compound_prefix + str(i) for i in range(len(smiles_list))]
    smiles_data_list = [(smiles, compound_ids[i], target_prediction_dataset_path, protein_name) for i, smiles in enumerate(smiles_list)]
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        executor.map(process_smiles, smiles_data_list)
    end_time = time.time()
    
    print(f"Time taken for all: {end_time - start_time}")
    total_image_count = len(smiles_list) * len([(0, ""), *[(angle, f"_{angle}") for angle in range(10, 360, 10)]])
    print(f"Total images generated: {total_image_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Usage format:
    # --dataset_file <dataset_file_path> --max_cores <max_cores> --target_prediction_dataset_path <target_prediction_dataset_path> --protein_name <protein_name>
    
    parser.add_argument('--dataset_file', type=str, default='path/to/your/default/dataset.csv', help='Path to the dataset file')
    parser.add_argument('--max_cores', type=int, default=multiprocessing.cpu_count() - 1, help='Maximum number of cores to use')
    parser.add_argument('--target_prediction_dataset_path', type=str, default='path/to/your/default/target_prediction_dataset/', help='Path to the target prediction dataset directory')
    parser.add_argument('--protein_name', type=str, default='AKT', help='Name of the protein')

    args = parser.parse_args()

    if args.max_cores > multiprocessing.cpu_count():
        print(f"Warning: Maximum number of cores is {multiprocessing.cpu_count()}. Using maximum available cores.")
        args.max_cores = multiprocessing.cpu_count()

    smiles_file = args.dataset_file
    protein_name = args.protein_name
    target_prediction_dataset_path = args.target_prediction_dataset_path

    initialize_dirs(protein_name, target_prediction_dataset_path)
    generate_images(smiles_file, protein_name, target_prediction_dataset_path, args.max_cores)
