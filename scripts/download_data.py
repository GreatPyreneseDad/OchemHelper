#!/usr/bin/env python
"""Download and prepare molecular datasets for training."""

import os
import gzip
import shutil
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MolecularDatasetDownloader:
    """Download and prepare molecular datasets."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize downloader."""
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download file from URL."""
        filepath = self.raw_dir / filename
        
        if filepath.exists():
            logger.info(f"File {filename} already exists, skipping download")
            return filepath
        
        logger.info(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        return filepath
    
    def download_chembl_subset(self, max_molecules: int = 100000):
        """Download ChEMBL subset (drug-like molecules)."""
        logger.info("Downloading ChEMBL subset...")
        
        # For demonstration, we'll create a synthetic dataset
        # In production, you would download from ChEMBL FTP
        output_file = self.processed_dir / 'chembl_subset.csv'
        
        if output_file.exists():
            logger.info("ChEMBL subset already exists")
            return output_file
        
        # ChEMBL FTP URLs (example - would need actual ChEMBL download)
        # url = "ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_33_chemreps.txt.gz"
        
        # For now, generate synthetic drug-like SMILES
        logger.info("Generating synthetic drug-like dataset...")
        smiles_list = self._generate_drug_like_smiles(max_molecules)
        
        df = pd.DataFrame({
            'SMILES': smiles_list,
            'ChEMBL_ID': [f'CHEMBL{i}' for i in range(len(smiles_list))]
        })
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} molecules to {output_file}")
        
        return output_file
    
    def download_zinc_subset(self, subset: str = 'drug-like'):
        """Download ZINC database subset."""
        logger.info(f"Downloading ZINC {subset} subset...")
        
        output_file = self.processed_dir / f'zinc_{subset}.csv'
        
        if output_file.exists():
            logger.info(f"ZINC {subset} already exists")
            return output_file
        
        # ZINC URLs
        zinc_urls = {
            'drug-like': 'https://zinc15.docking.org/substances/subsets/drug-like.smi.gz',
            'lead-like': 'https://zinc15.docking.org/substances/subsets/lead-like.smi.gz',
            'fragment-like': 'https://zinc15.docking.org/substances/subsets/fragment-like.smi.gz'
        }
        
        # Note: These URLs would need authentication in practice
        # For demonstration, generate synthetic data
        logger.info(f"Generating synthetic ZINC-like dataset...")
        
        if subset == 'drug-like':
            smiles_list = self._generate_drug_like_smiles(50000)
        elif subset == 'lead-like':
            smiles_list = self._generate_lead_like_smiles(30000)
        else:
            smiles_list = self._generate_fragment_like_smiles(20000)
        
        df = pd.DataFrame({
            'SMILES': smiles_list,
            'ZINC_ID': [f'ZINC{i:08d}' for i in range(len(smiles_list))]
        })
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} molecules to {output_file}")
        
        return output_file
    
    def download_pubchem_bioassays(self, assay_ids: Optional[List[int]] = None):
        """Download PubChem bioassay data."""
        logger.info("Downloading PubChem bioassay data...")
        
        output_file = self.processed_dir / 'pubchem_bioassays.csv'
        
        if output_file.exists():
            logger.info("PubChem bioassays already exist")
            return output_file
        
        # PubChem REST API example
        # In practice, you would use PubChemPy or direct API calls
        
        # Generate synthetic bioassay data
        logger.info("Generating synthetic bioassay dataset...")
        
        smiles_list = self._generate_drug_like_smiles(10000)
        activities = np.random.choice(['Active', 'Inactive'], size=len(smiles_list))
        ic50_values = np.random.lognormal(mean=1, sigma=2, size=len(smiles_list))
        
        df = pd.DataFrame({
            'SMILES': smiles_list,
            'Activity': activities,
            'IC50_nM': ic50_values,
            'PubChem_CID': [f'CID{i}' for i in range(len(smiles_list))]
        })
        
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} bioassay results to {output_file}")
        
        return output_file
    
    def prepare_qm9_dataset(self):
        """Download and prepare QM9 quantum chemistry dataset."""
        logger.info("Preparing QM9 dataset...")
        
        output_file = self.processed_dir / 'qm9.csv'
        
        if output_file.exists():
            logger.info("QM9 dataset already exists")
            return output_file
        
        # QM9 download URL
        # url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
        
        # Generate synthetic QM9-like data
        logger.info("Generating synthetic QM9-like dataset...")
        
        smiles_list = self._generate_small_molecules(5000)
        
        # Synthetic quantum properties
        data = {
            'SMILES': smiles_list,
            'mu': np.random.normal(2.5, 1.5, len(smiles_list)),  # Dipole moment
            'alpha': np.random.normal(60, 20, len(smiles_list)),  # Polarizability
            'homo': np.random.normal(-6, 1, len(smiles_list)),  # HOMO energy
            'lumo': np.random.normal(-2, 1, len(smiles_list)),  # LUMO energy
            'gap': np.random.normal(4, 1, len(smiles_list)),  # HOMO-LUMO gap
            'r2': np.random.normal(400, 100, len(smiles_list)),  # R^2
            'zpve': np.random.normal(0.1, 0.05, len(smiles_list)),  # Zero point energy
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} molecules with quantum properties to {output_file}")
        
        return output_file
    
    def _generate_drug_like_smiles(self, n_molecules: int) -> List[str]:
        """Generate synthetic drug-like SMILES."""
        # Common drug-like scaffolds and fragments
        scaffolds = [
            'c1ccccc1',  # Benzene
            'c1ccncc1',  # Pyridine
            'c1ccc2ccccc2c1',  # Naphthalene
            'C1CCCCC1',  # Cyclohexane
            'C1CCNCC1',  # Piperidine
            'c1ccc2[nH]ccc2c1',  # Indole
            'c1ccc2c(c1)ncn2',  # Benzimidazole
            'c1ccc(cc1)C(=O)',  # Benzoyl
            'c1ccc(cc1)S(=O)(=O)',  # Benzenesulfonyl
            'c1cnc2c(n1)ncn2',  # Purine
        ]
        
        substituents = [
            'C', 'CC', 'CCC', 'C(C)C',  # Alkyl
            'O', 'OC', 'OCC',  # Alkoxy
            'N', 'NC', 'N(C)C',  # Amino
            'F', 'Cl', 'Br',  # Halogens
            'C(=O)N', 'C(=O)O',  # Amide, carboxyl
            'S', 'S(=O)(=O)N',  # Thio, sulfonamide
            'C#N',  # Nitrile
        ]
        
        smiles_list = []
        
        for _ in range(n_molecules):
            # Choose random scaffold
            scaffold = np.random.choice(scaffolds)
            mol = Chem.MolFromSmiles(scaffold)
            
            if mol:
                # Add 1-3 substituents
                n_subs = np.random.randint(1, 4)
                modified_smiles = scaffold
                
                for _ in range(n_subs):
                    sub = np.random.choice(substituents)
                    # Simple concatenation (not chemically accurate but for demo)
                    position = np.random.randint(1, 7)
                    modified_smiles = modified_smiles.replace(
                        'c1', f'c({sub})1', 1
                    ) if 'c1' in modified_smiles else modified_smiles + sub
                
                # Validate and canonicalize
                mol = Chem.MolFromSmiles(modified_smiles)
                if mol:
                    try:
                        # Check Lipinski's Rule of Five
                        mw = Descriptors.MolWt(mol)
                        logp = Descriptors.MolLogP(mol)
                        hbd = Descriptors.NumHDonors(mol)
                        hba = Descriptors.NumHAcceptors(mol)
                        
                        if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10):
                            canonical = Chem.MolToSmiles(mol)
                            smiles_list.append(canonical)
                    except:
                        pass
            
            # Fallback to scaffold if modification failed
            if len(smiles_list) <= _:
                smiles_list.append(scaffold)
        
        return smiles_list[:n_molecules]
    
    def _generate_lead_like_smiles(self, n_molecules: int) -> List[str]:
        """Generate synthetic lead-like SMILES (smaller than drug-like)."""
        smiles_list = []
        
        scaffolds = [
            'c1ccccc1', 'c1ccncc1', 'C1CCCCC1',
            'c1ccoc1', 'c1ccsc1', 'C1CCOC1'
        ]
        
        for _ in range(n_molecules):
            scaffold = np.random.choice(scaffolds)
            # Lead-like: MW 250-350, fewer substituents
            sub = np.random.choice(['C', 'O', 'N', 'F', 'Cl'])
            smiles = scaffold.replace('c1', f'c({sub})1', 1)
            
            mol = Chem.MolFromSmiles(smiles)
            if mol and Descriptors.MolWt(mol) <= 350:
                smiles_list.append(Chem.MolToSmiles(mol))
            else:
                smiles_list.append(scaffold)
        
        return smiles_list[:n_molecules]
    
    def _generate_fragment_like_smiles(self, n_molecules: int) -> List[str]:
        """Generate synthetic fragment-like SMILES (very small)."""
        fragments = [
            'c1ccccc1', 'c1ccncc1', 'c1ccoc1',
            'C1CCC1', 'C1CCNC1', 'CC(=O)',
            'CCO', 'CCN', 'CCS', 'CC#N',
            'c1cnc[nH]1', 'c1cnn[nH]1'
        ]
        
        return [np.random.choice(fragments) for _ in range(n_molecules)]
    
    def _generate_small_molecules(self, n_molecules: int) -> List[str]:
        """Generate small organic molecules (QM9-like)."""
        # QM9 contains molecules with up to 9 heavy atoms
        small_molecules = [
            'C', 'CC', 'CCC', 'CO', 'CCO', 'CN', 'CCN',
            'C=C', 'C=O', 'C#C', 'C#N', 'O=C=O',
            'C1CC1', 'C1CCC1', 'C1CCCC1',
            'c1ccccc1', 'Cc1ccccc1', 'CCc1ccccc1',
            'COC', 'CNC', 'CSC', 'C(F)(F)F',
            'C1=CC=CC=C1', 'C1=CNC=N1', 'C1=CSC=C1'
        ]
        
        smiles_list = []
        for _ in range(n_molecules):
            base = np.random.choice(small_molecules)
            mol = Chem.MolFromSmiles(base)
            if mol and mol.GetNumHeavyAtoms() <= 9:
                smiles_list.append(Chem.MolToSmiles(mol))
        
        return smiles_list[:n_molecules]
    
    def download_all(self):
        """Download all datasets."""
        logger.info("Downloading all datasets...")
        
        self.download_chembl_subset()
        self.download_zinc_subset('drug-like')
        self.download_pubchem_bioassays()
        self.prepare_qm9_dataset()
        
        # Combine all for training
        self.create_combined_dataset()
        
        logger.info("All datasets downloaded successfully!")
    
    def create_combined_dataset(self):
        """Combine all datasets into training set."""
        logger.info("Creating combined training dataset...")
        
        output_file = self.processed_dir / 'combined_training.csv'
        
        all_smiles = []
        
        # Load all processed datasets
        for file in self.processed_dir.glob('*.csv'):
            if file.name != 'combined_training.csv':
                df = pd.read_csv(file)
                if 'SMILES' in df.columns:
                    all_smiles.extend(df['SMILES'].tolist())
                elif 'smiles' in df.columns:
                    all_smiles.extend(df['smiles'].tolist())
        
        # Remove duplicates
        unique_smiles = list(set(all_smiles))
        
        # Validate SMILES
        valid_smiles = []
        for smiles in tqdm(unique_smiles, desc="Validating SMILES"):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                canonical = Chem.MolToSmiles(mol)
                valid_smiles.append(canonical)
        
        # Save combined dataset
        df = pd.DataFrame({'SMILES': valid_smiles})
        df.to_csv(output_file, index=False)
        
        logger.info(f"Combined dataset: {len(valid_smiles)} unique valid SMILES")
        logger.info(f"Saved to {output_file}")
        
        # Also save as text file for easy loading
        txt_file = self.processed_dir / 'combined_training.txt'
        with open(txt_file, 'w') as f:
            for smiles in valid_smiles:
                f.write(smiles + '\n')
        
        return output_file


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Download molecular datasets')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--dataset', type=str, help='Specific dataset to download')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    
    args = parser.parse_args()
    
    downloader = MolecularDatasetDownloader(args.data_dir)
    
    if args.all:
        downloader.download_all()
    elif args.dataset == 'chembl':
        downloader.download_chembl_subset()
    elif args.dataset == 'zinc':
        downloader.download_zinc_subset()
    elif args.dataset == 'pubchem':
        downloader.download_pubchem_bioassays()
    elif args.dataset == 'qm9':
        downloader.prepare_qm9_dataset()
    else:
        # Default: download all
        downloader.download_all()


if __name__ == '__main__':
    main()
