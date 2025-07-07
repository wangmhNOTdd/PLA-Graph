#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Processing script for PDBBind v2020-other-PL dataset - Test version (first 100 samples)
"""
import os
import sys
import json
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

# Add project directory to path
PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.pdb_utils import Protein, VOCAB
from data.dataset import blocks_interface, blocks_to_data
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.mol2_to_blocks import mol2_to_blocks
from utils.logger import print_log

def parse_args():
    parser = argparse.ArgumentParser(description='Process PDBBind v2020-other-PL dataset - Test version')
    parser.add_argument('--data_dir', type=str, 
                        default='./datasets/v2020-other-PL',
                        help='Directory containing the v2020-other-PL dataset')
    parser.add_argument('--out_dir', type=str,
                        default='./datasets/v2020-other-PL/processed_test',
                        help='Output directory for processed data')
    parser.add_argument('--interface_dist_th', type=float, default=8.0,
                        help='Distance threshold for interface residues')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of samples to process for testing')
    parser.add_argument('--fragment', default=None, choices=['PS_300', 'PS_500'],
                        help='Fragment representation for small molecules')
    return parser.parse_args()

def parse_binding_affinity(affinity_str):
    """
    Parse binding affinity string from PDBBind index file
    Examples: 'Kd=49uM', 'Ki=0.43uM', 'IC50=355mM'
    Returns -log10(affinity_in_M)
    """
    # Remove spaces and convert to lowercase
    affinity_str = affinity_str.replace(' ', '').lower()
    
    # Extract numeric value and unit
    match = re.search(r'([kd|ki|ic50]+)=([0-9.]+)([a-z]+)', affinity_str)
    if not match:
        return None
    
    value = float(match.group(2))
    unit = match.group(3).lower()
    
    # Convert to Molar
    unit_multipliers = {
        'pm': 1e-12,
        'nm': 1e-9,
        'um': 1e-6,
        'mm': 1e-3,
        'm': 1.0
    }
    
    if unit not in unit_multipliers:
        print_log(f'Unknown unit: {unit}', level='WARN')
        return None
    
    affinity_m = value * unit_multipliers[unit]
    neglog_aff = -np.log10(affinity_m)
    
    return neglog_aff

def load_index_file(data_dir, max_samples=100):
    """Load the index file to get PDB codes and binding affinities"""
    index_file = os.path.join(data_dir, 'index', 'INDEX_general_PL.2020')
    
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    entries = []
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            if len(entries) >= max_samples:
                break
                
            parts = line.split()
            if len(parts) < 4:
                continue
                
            pdb_code = parts[0]
            binding_data = parts[3]
            
            # Parse affinity
            neglog_aff = parse_binding_affinity(binding_data)
            if neglog_aff is None:
                continue
                
            entries.append({
                'pdb_code': pdb_code,
                'neglog_aff': neglog_aff,
                'raw_affinity': binding_data
            })
    
    print_log(f'Loaded {len(entries)} entries from index file')
    return entries

def process_one_complex(entry, data_dir, interface_dist_th, fragment):
    """Process one protein-ligand complex"""
    pdb_code = entry['pdb_code']
    neglog_aff = entry['neglog_aff']
    
    # File paths
    complex_dir = os.path.join(data_dir, pdb_code)
    if not os.path.exists(complex_dir):
        print_log(f'Complex directory not found: {complex_dir}', level='ERROR')
        return None
    
    protein_file = os.path.join(complex_dir, f'{pdb_code}_protein.pdb')
    ligand_file = os.path.join(complex_dir, f'{pdb_code}_ligand.mol2')
    
    if not os.path.exists(protein_file):
        print_log(f'Protein file not found: {protein_file}', level='ERROR')
        return None
    
    if not os.path.exists(ligand_file):
        print_log(f'Ligand file not found: {ligand_file}', level='ERROR')
        return None
    
    try:
        # Parse protein
        list_blocks1 = pdb_to_list_blocks(protein_file)
        blocks1 = []
        for b in list_blocks1:
            blocks1.extend(b)
        
        # Parse ligand
        blocks2 = mol2_to_blocks(ligand_file, fragment=fragment)
        
        # Find interface
        blocks1, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
        if len(blocks1) == 0:
            print_log(f'{pdb_code} has no interface', level='ERROR')
            return None
        
        # Convert to data format
        data = blocks_to_data(blocks1, blocks2)
        
        # Convert numpy arrays to lists for serialization
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        
        item = {
            'id': pdb_code,
            'affinity': {'neglog_aff': neglog_aff}
        }
        
        # Add data fields directly
        item.update(data)
        
        return item
        
    except Exception as e:
        print_log(f'Error processing {pdb_code}: {e}', level='ERROR')
        return None

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load index file
    print_log('Loading index file...')
    entries = load_index_file(args.data_dir, max_samples=args.max_samples)
    
    # Filter entries that have corresponding files
    print_log('Filtering entries with existing files...')
    valid_entries = []
    for entry in tqdm(entries, desc='Checking files'):
        pdb_code = entry['pdb_code']
        complex_dir = os.path.join(args.data_dir, pdb_code)
        protein_file = os.path.join(complex_dir, f'{pdb_code}_protein.pdb')
        ligand_file = os.path.join(complex_dir, f'{pdb_code}_ligand.mol2')
        
        if os.path.exists(complex_dir) and os.path.exists(protein_file) and os.path.exists(ligand_file):
            valid_entries.append(entry)
    
    print_log(f'Found {len(valid_entries)} valid entries out of {len(entries)}')
    
    # Simple split: 80% train, 10% valid, 10% test
    np.random.seed(42)
    indices = np.random.permutation(len(valid_entries))
    
    n_test = max(1, len(valid_entries) // 10)
    n_valid = max(1, len(valid_entries) // 10) 
    n_train = len(valid_entries) - n_test - n_valid
    
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]
    
    train_entries = [valid_entries[i] for i in train_indices]
    valid_entries_split = [valid_entries[i] for i in valid_indices]
    test_entries = [valid_entries[i] for i in test_indices]
    
    print_log(f'Dataset split: Train={len(train_entries)}, Valid={len(valid_entries_split)}, Test={len(test_entries)}')
    
    # Process each split
    for split_name, split_entries in [('train', train_entries), ('valid', valid_entries_split), ('test', test_entries)]:
        print_log(f'Processing {split_name} set...')
        processed_data = []
        
        for entry in tqdm(split_entries, desc=f'Processing {split_name}'):
            result = process_one_complex(entry, args.data_dir, args.interface_dist_th, args.fragment)
            if result is not None:
                processed_data.append(result)
        
        # Save processed data
        output_file = os.path.join(args.out_dir, f'{split_name}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print_log(f'Saved {len(processed_data)} processed {split_name} samples to {output_file}')
    
    # Save split information
    split_info = {
        'train_size': len(train_entries),
        'valid_size': len(valid_entries_split), 
        'test_size': len(test_entries),
        'total_processed': len(valid_entries),
        'interface_dist_th': args.interface_dist_th,
        'fragment': args.fragment,
        'max_samples': args.max_samples
    }
    
    split_info_file = os.path.join(args.out_dir, 'split_info.json')
    with open(split_info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print_log(f'Test dataset processing completed. Split info saved to {split_info_file}')

if __name__ == '__main__':
    main()
