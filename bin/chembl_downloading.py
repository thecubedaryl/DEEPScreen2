import requests
import pandas as pd
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def fetch_activities(target_chembl_ids, assay_types, pchembl_threshold):
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    params = {
        'target_chembl_id__in': ','.join(target_chembl_ids),
        'assay_type__in': ','.join(assay_types),
        'pchembl_value__isnull': 'false',
        'only': 'molecule_chembl_id,pchembl_value,target_chembl_id,bao_label'
    }
    
    activities = []
    while True:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
            break
        
        data = response.json()
        if 'activities' in data:
            activities.extend(data['activities'])
        else:
            print("No activities found.")
            break
        
        if 'page_meta' in data and data['page_meta']['next']:
            params['offset'] = data['page_meta']['offset'] + data['page_meta']['limit']
        else:
            break

    if activities:
        df = pd.DataFrame(activities)
        if 'pchembl_value' in df.columns:
            df['pchembl_value'] = pd.to_numeric(df['pchembl_value'], errors='coerce')
            df = df[df['pchembl_value'].notnull() & (df['pchembl_value'] >= pchembl_threshold)]
            df.drop(columns=['bao_label'], errors='ignore', inplace=True)
        else:
            print("pchembl_value column not found.")
            return pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    return df

def fetch_smiles(compound_id):
    url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{compound_id}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        smiles = data.get('molecule_structures', {}).get('canonical_smiles', None)
        return compound_id, smiles
    else:
        print(f"Failed to fetch data for {compound_id}. HTTP Status Code: {response.status_code}")
        return compound_id, None

def check_and_download_smiles(compound_ids):
    smiles_data = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(fetch_smiles, compound_ids))
        
    for compound_id, smiles in results:
        if smiles:
            smiles_data.append((compound_id, smiles))
        else:
            print(f"No SMILES found for {compound_id}")
    
    return smiles_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ChEMBL activity data and SMILES")
    parser.add_argument('--target_chembl_id', type=str, help="Target ChEMBL ID(s) to search for, comma-separated")
    parser.add_argument('--assay_type', type=str, default='B', help="Assay type(s) to search for, comma-separated")
    parser.add_argument('--pchembl_threshold', type=float, default=6.0, help="Threshold for pChembl value to determine active/inactive")
    parser.add_argument('--output_file', type=str, default='activity_data.csv', help="Output file to save activity data")
    parser.add_argument('--max_cores', type=int, default=multiprocessing.cpu_count() - 1, help="Maximum number of CPU cores to use")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'training_files', 'target_training_datasets', args.target_chembl_id.replace(',', '_'))
    output_path = os.path.join(output_dir, args.output_file)
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        
        if args.target_chembl_id:
            target_chembl_ids = args.target_chembl_id.split(',')
        else:
            target_chembl_ids = []

        assay_types = args.assay_type.split(',')

        all_data = pd.DataFrame()
        if target_chembl_ids:
            data = fetch_activities(target_chembl_ids, assay_types, args.pchembl_threshold)
            all_data = pd.concat([all_data, data])

        if not all_data.empty:
            compound_ids = all_data['molecule_chembl_id'].unique().tolist()
            smiles_data = check_and_download_smiles(compound_ids)
            
            if smiles_data:
                smiles_df = pd.DataFrame(smiles_data, columns=["molecule_chembl_id", "canonical_smiles"])
                all_data = all_data.merge(smiles_df, on='molecule_chembl_id')
            
            all_data.to_csv(output_path, index=False)
            print(f"Activity data saved to {output_path}")
        else:
            print("No activity data found.")
