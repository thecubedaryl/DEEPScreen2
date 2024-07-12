## DEEPScreen2: An Automated Tool for Drug-Target Interaction Prediction Using Deep Convolutional Neural Networks Fed by 2-D Images of Compounds

Hayriye Çelikbilek1, Tunca Doğan1,2

1Biological Data Science Lab, Dept. of Computer Engineering & AI Engineering, Hacettepe University, Ankara, Turkey <br>
2Dept. of Bioinformatics, Graduate School of Health Sciences, Hacettepe University, Ankara, Turkey

Accurately predicting drug-target interactions (DTI) possess critical importance in drug discovery and development, due to the labour-intensive and costly nature of conventional experimental screening techniques. A widely-utilised deep learning-based DTI prediction tool is DEEPScreen, which was previously developed by our group. DEEPScreen2, an advanced framework for DTI prediction (Figure 1), emerges as a DCNN solution for researchers regardless of technical background, building upon the groundwork laid by its predecessor, DEEPScreen.<br>
In this study, we proposed DEEPScreen2 (Figure 1) by offering numerous innovations over the previous implementation: (i) adopting 300x300 compounds images to increase the resolution and enable the capture of nuanced structural features; (ii) augmenting data by 36 different 10-degree rotations of the original compound images to render the model rotation invariant and thus more robust; (iii) fully automatizing the data preparation, data loading and inference/prediction procedures; (iv) updating the training, validation and test data via employing ChEMBL database v33 (as opposed to v23 in the old version); (v) introducing percentile-60 based data splitting as active and inactive data points with respect to the pChEMBL values, which extended the library over 800 target proteins.<br>
With its advanced image-based representations and flexible usability, DEEPScreen2 illustrates the relationship between artificial intelligence centric modeling and molecular understanding. We expect that DEEPScreen2 will be utilised in translational research, by scientists working on drug discovery and repurposing for fast and preliminary virtual screening of large chemical libraries. DEEPScreen2 is available as a programmatic tool together with its datasets and results at https://github.com/HUBioDataLab/DEEPScreen2.<br>

# Molecule Image Generation Using SMILES Dataset

## Overview
This script generates molecule images from a dataset of SMILES strings using parallel processing.

## Steps to Use

1. **Download the Script**
   Save the script as `img_gen.py`.

2. **Prepare Your Dataset**
   Ensure you have a CSV file with a column named `smiles`.

3. **Run the Script**
   - Open a terminal and navigate to the directory containing `generate_images.py`.
   - Use the following command format to run the script:
     ```bash
     python generate_images.py --dataset_file="{path to your dataset file}" --max_cores="{number of cores}" --target_prediction_dataset_path="{path to save images}" --protein_name="{protein name}"
     ```

   - **Example Command:**
     ```bash
     python generate_images.py --dataset_file="path/to/dataset.csv" --max_cores=4 --target_prediction_dataset_path="path/to/target_prediction_dataset/" --protein_name="AKT"
     ```

## Output
The generated images will be saved in the specified `target_prediction_dataset_path` under subdirectories named after the protein.


# ChEMBL Data Downloader

This script downloads ChEMBL activity data and associated SMILES strings for specified targets. The data is saved to a CSV file.

## Prerequisites

- Python 3.x
- Install required packages:

  ```bash
  pip install requests pandas
  ```
## Usage
**Navigate to the bin directory:**

   ```bash
   cd bin
   ```

Run the script:
   
   ```bash
   python download.py --target_chembl_id "CHEMBL286" --assay_type "B" --pchembl_threshold 6.0 --output_file "activity_data.csv"
   ```

--target_chembl_id: ChEMBL ID(s) of the target(s), comma-separated.
--assay_type: Assay type(s), comma-separated (default: 'B').
--pchembl_threshold: Threshold for pChembl value (default: 6.0).
--output_file: Output file name (default: 'activity_data.csv').


**Example Command:**

   ```bash
      python download.py --target_chembl_id "CHEMBL286" --assay_type "B" --pchembl_threshold 6.0 --output_file "activity_data.csv"
   ```

The script will check if the dataset exists in training_files/target_training_datasets. If it exists, the download is skipped; otherwise, it downloads and saves the data.
