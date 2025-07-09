#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation script for MACE-En vs MACE vs GET models
Compares performance on both v2020-other-PL and standard PDBBind datasets
"""

import os
import sys
import pickle
import argparse
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model imports
from models.MACE.model import MACE_MP
from models.MACE_En.model import MACE_En
from models.GET.model import GET  # Assuming GET model exists
from models.MACE.modules.tools.utils import AtomicNumberTable
from utils.convert import load_data


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.z_table = AtomicNumberTable([1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
        
    def load_model_checkpoint(self, checkpoint_path: str, model_type: str) -> nn.Module:
        """Load model from checkpoint."""
        print(f"Loading {model_type} model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        if model_type.lower() == 'mace_en':
            model = self._build_mace_en_model(config)
        elif model_type.lower() == 'mace':
            model = self._build_mace_model(config)
        elif model_type.lower() == 'get':
            model = self._build_get_model(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _build_mace_en_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build MACE-En model from config."""
        from models.MACE.modules.blocks import InteractionBlock
        
        atomic_energies = torch.zeros(len(self.z_table.zs), dtype=torch.get_default_dtype())
        
        model = MACE_En(
            r_max=config.get('r_max', 5.0),
            num_bessel=config.get('num_bessel', 8),
            num_polynomial_cutoff=config.get('num_polynomial_cutoff', 5),
            max_ell=config.get('max_ell', 3),
            interaction_cls=InteractionBlock,
            interaction_cls_first=InteractionBlock,
            num_interactions=config.get('num_interactions', 2),
            num_elements=len(self.z_table.zs),
            hidden_irreps=config.get('hidden_irreps', '128x0e + 128x1o'),
            MLP_irreps=config.get('MLP_irreps', '16x0e'),
            atomic_energies=atomic_energies.numpy(),
            avg_num_neighbors=config.get('avg_num_neighbors', 50.0),
            atomic_numbers=self.z_table.zs,
            correlation=config.get('correlation', 3),
            gate=torch.nn.functional.silu,
            radial_type=config.get('radial_type', 'enhanced_bessel'),
            numerical_eps=config.get('numerical_eps', 1e-8),
        )
        
        return model.to(self.device)
    
    def _build_mace_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build standard MACE model from config."""
        from models.MACE.modules.blocks import InteractionBlock
        
        atomic_energies = torch.zeros(len(self.z_table.zs), dtype=torch.get_default_dtype())
        
        model = MACE_MP(
            r_max=config.get('r_max', 5.0),
            num_bessel=config.get('num_bessel', 8),
            num_polynomial_cutoff=config.get('num_polynomial_cutoff', 5),
            max_ell=config.get('max_ell', 3),
            interaction_cls=InteractionBlock,
            interaction_cls_first=InteractionBlock,
            num_interactions=config.get('num_interactions', 2),
            num_elements=len(self.z_table.zs),
            hidden_irreps=config.get('hidden_irreps', '128x0e + 128x1o'),
            MLP_irreps=config.get('MLP_irreps', '16x0e'),
            atomic_energies=atomic_energies.numpy(),
            avg_num_neighbors=config.get('avg_num_neighbors', 50.0),
            atomic_numbers=self.z_table.zs,
            correlation=config.get('correlation', 3),
            gate=torch.nn.functional.silu,
        )
        
        return model.to(self.device)
    
    def _build_get_model(self, config: Dict[str, Any]) -> nn.Module:
        """Build GET model from config (placeholder - adjust based on actual GET implementation)."""
        # This is a placeholder - adjust based on your actual GET model implementation
        print("GET model building not implemented - placeholder")
        return None
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader, 
                      model_name: str) -> Dict[str, float]:
        """Evaluate a single model on a dataset."""
        print(f"Evaluating {model_name}...")
        
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                try:
                    batch = batch.to(self.device)
                    
                    # Handle different model types
                    if hasattr(model, 'forward'):
                        if 'MACE' in model.__class__.__name__:
                            output = model(batch.to_dict(), training=False)
                            pred = output['energy'].cpu().numpy()
                        else:
                            # For GET or other models
                            pred = model(batch).cpu().numpy()
                    else:
                        continue
                    
                    target = batch.y.cpu().numpy()
                    
                    predictions.extend(pred)
                    targets.extend(target)
                    
                    if batch_idx % 100 == 0:
                        print(f"  Processed batch {batch_idx}/{len(data_loader)}")
                        
                except Exception as e:
                    print(f"  Error in batch {batch_idx}: {e}")
                    continue
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Remove any NaN or inf values
        valid_mask = np.isfinite(predictions) & np.isfinite(targets)
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if len(predictions) == 0:
            print(f"Warning: No valid predictions for {model_name}")
            return {}
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets)
        
        print(f"  {model_name} Results:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")
            
        return metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['r2'] = r2_score(targets, predictions)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(targets, predictions)
        spearman_r, spearman_p = spearmanr(targets, predictions)
        
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
        
        # Additional metrics
        residuals = targets - predictions
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_absolute_error'] = np.max(np.abs(residuals))
        
        return metrics
    
    def compare_models(self, model_configs: List[Dict[str, Any]], 
                      datasets: Dict[str, DataLoader]) -> pd.DataFrame:
        """Compare multiple models on multiple datasets."""
        results = []
        
        for dataset_name, data_loader in datasets.items():
            print(f"\n=== Evaluating on {dataset_name} dataset ===")
            
            for config in model_configs:
                model_name = config['name']
                model_path = config['checkpoint_path']
                model_type = config['model_type']
                
                if not os.path.exists(model_path):
                    print(f"Warning: Checkpoint not found for {model_name}: {model_path}")
                    continue
                
                try:
                    # Load and evaluate model
                    model = self.load_model_checkpoint(model_path, model_type)
                    metrics = self.evaluate_model(model, data_loader, model_name)
                    
                    # Add to results
                    result = {
                        'dataset': dataset_name,
                        'model': model_name,
                        'model_type': model_type,
                        **metrics
                    }
                    results.append(result)
                    
                    # Clean up memory
                    del model
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        return df
    
    def save_results(self, results_df: pd.DataFrame, output_dir: str):
        """Save results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        results_df.to_csv(os.path.join(output_dir, 'comprehensive_evaluation_results.csv'), index=False)
        
        # Save summary by dataset
        for dataset in results_df['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            dataset_results.to_csv(os.path.join(output_dir, f'{dataset}_results.csv'), index=False)
        
        # Save summary by model
        for model in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model]
            model_results.to_csv(os.path.join(output_dir, f'{model}_results.csv'), index=False)
        
        # Create summary table
        summary_metrics = ['pearson_r', 'rmse', 'mae', 'r2']
        summary = results_df.pivot_table(
            index='model', 
            columns='dataset', 
            values=summary_metrics,
            aggfunc='first'
        )
        summary.to_csv(os.path.join(output_dir, 'summary_comparison.csv'))
        
        print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--config_file', type=str, required=True,
                       help='JSON config file with model and dataset paths')
    parser.add_argument('--output_dir', type=str, default='./results/comprehensive_evaluation',
                       help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device)
    
    # Load datasets
    datasets = {}
    for dataset_name, dataset_path in config['datasets'].items():
        if os.path.exists(dataset_path):
            print(f"Loading {dataset_name} dataset from {dataset_path}")
            data = load_data(dataset_path)
            datasets[dataset_name] = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        else:
            print(f"Warning: Dataset not found: {dataset_path}")
    
    # Run comparison
    results_df = evaluator.compare_models(config['models'], datasets)
    
    # Save results
    evaluator.save_results(results_df, args.output_dir)
    
    # Print summary
    print("\n=== SUMMARY ===")
    if not results_df.empty:
        print("\nBest performing models by dataset (Pearson R):")
        for dataset in results_df['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            best_model = dataset_results.loc[dataset_results['pearson_r'].idxmax()]
            print(f"  {dataset}: {best_model['model']} (r={best_model['pearson_r']:.4f})")
        
        print("\nOverall best performing model (average Pearson R):")
        avg_performance = results_df.groupby('model')['pearson_r'].mean()
        best_overall = avg_performance.idxmax()
        print(f"  {best_overall}: {avg_performance[best_overall]:.4f}")
    else:
        print("No results to summarize.")


if __name__ == '__main__':
    # Example config file content (save as evaluation_config.json):
    example_config = {
        "models": [
            {
                "name": "MACE_En_enhanced_bessel",
                "model_type": "mace_en",
                "checkpoint_path": "./checkpoints/mace_en_v2020/best_model.pth"
            },
            {
                "name": "MACE_standard",
                "model_type": "mace", 
                "checkpoint_path": "./checkpoints/mace_pdbbind/best_model.pth"
            },
            {
                "name": "MACE_gaussian",
                "model_type": "mace",
                "checkpoint_path": "./checkpoints/mace_gaussian/version_1/best_model.pth"
            }
        ],
        "datasets": {
            "v2020_test": "./datasets/v2020-other-PL/processed_get_format/test.pkl",
            "pdbbind_identity30_test": "./datasets/PDBBind/processed/identity30/test.pkl",
            "v2020_valid": "./datasets/v2020-other-PL/processed_get_format/valid.pkl",
            "pdbbind_identity30_valid": "./datasets/PDBBind/processed/identity30/valid.pkl"
        }
    }
    
    print("Example evaluation_config.json:")
    print(json.dumps(example_config, indent=2))
    print("\nSave this as evaluation_config.json and run:")
    print("python evaluate_comprehensive.py --config_file evaluation_config.json")
    
    main()
