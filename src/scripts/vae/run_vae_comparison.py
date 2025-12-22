"""Script to run multiple VAE configurations and compare results."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
import pandas as pd


class VAEExperimentRunner:
    """Run multiple VAE training configurations and compare results."""
    
    def __init__(self, project_root: Path, output_dir: Path = None):
        """
        Initialize experiment runner.
        
        Args:
            project_root: Root directory of the project
            output_dir: Directory to save comparison results
        """
        self.project_root = Path(project_root)
        self.output_dir = output_dir or self.project_root / "outputs" / "vae_experiments"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.script_path = self.project_root / "src" / "scripts" / "vae" / "run_vae_training.py"
        self.results = []
    
    def run_configuration(
        self,
        config_name: str,
        model_variant: str = "default",
        trainer_variant: str = "default",
        epochs: int = None,
        batch_size: int = None,
        lr: float = None,
    ) -> bool:
        """
        Run a single VAE training configuration.
        
        Args:
            config_name: Name for this configuration run
            model_variant: Model variant (default, small, medium, large, beta_default, etc.)
            trainer_variant: Trainer variant (default, quick_test, distributed)
            epochs: Number of epochs (overrides config)
            batch_size: Batch size (overrides config)
            lr: Learning rate (overrides config)
        
        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Running configuration: {config_name}")
        print(f"Model: {model_variant}, Trainer: {trainer_variant}")
        print(f"{'='*80}\n")
        
        # Build command
        cmd = [
            sys.executable,
            str(self.script_path),
            f"model=vae/{model_variant}",
            f"trainer=vae/{trainer_variant}",
        ]
        
        if epochs is not None:
            cmd.append(f"trainer.max_epochs={epochs}")
        if batch_size is not None:
            cmd.append(f"data.batch_size={batch_size}")
        if lr is not None:
            cmd.append(f"trainer.optimizer.lr={lr}")
        
        # Add run ID
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cmd.append(f"run_id={config_name}_{timestamp}")
        
        try:
            # Run the training script
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=False,
                check=True,
            )
            print(f"✓ Configuration {config_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Configuration {config_name} failed with error:")
            print(f"  {e}")
            return False
        except FileNotFoundError:
            print(f"✗ Training script not found at {self.script_path}")
            return False
    
    def collect_results(self, metrics_dir: Path = None) -> Dict[str, Any]:
        """
        Collect metrics from all runs for comparison.
        
        Args:
            metrics_dir: Directory containing metrics files
        
        Returns:
            Dictionary with aggregated results
        """
        if metrics_dir is None:
            metrics_dir = Path(__file__).parent.parent.parent.parent / "models" / "metrics"
        
        if not metrics_dir.exists():
            print(f"Warning: Metrics directory not found at {metrics_dir}")
            return {}
        
        results = {}
        
        # Find all metrics files
        for metrics_file in metrics_dir.glob("*_metrics.json"):
            config_name = metrics_file.stem.replace("_metrics", "")
            
            try:
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                results[config_name] = metrics
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read {metrics_file}: {e}")
        
        return results
    
    def compare_configurations(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create comparison dataframe from results.
        
        Args:
            results: Dictionary with results from each configuration
        
        Returns:
            Pandas DataFrame with comparison
        """
        comparison_data = []
        
        for config_name, metrics in results.items():
            row = {
                'configuration': config_name,
            }
            
            # Extract key metrics from test phase
            if 'test' in metrics:
                test_metrics = metrics['test']
                # Loss metrics
                row['test_loss'] = test_metrics.get('test/loss', None)
                row['test_bce'] = test_metrics.get('test/bce', None)
                row['test_kld'] = test_metrics.get('test/kld', None)
                
                # Reconstruction metrics
                row['hamming_loss'] = test_metrics.get('test/hamming_loss', None)
                row['jaccard_index'] = test_metrics.get('test/jaccard_index', None)
                
                # Latent space metrics
                row['pct_active_units'] = test_metrics.get('test/active_units_pct_active_units', None)
                row['silhouette_score'] = test_metrics.get('test/silhouette_score', None)
                
                # Diversity metrics
                row['pct_unique_combinations'] = test_metrics.get('test/diversity_pct_unique_combinations', None)
                row['entropy'] = test_metrics.get('test/diversity_entropy', None)
                row['gini_coefficient'] = test_metrics.get('test/diversity_gini_coefficient', None)
                
                # Co-occurrence metrics
                row['cosine_similarity'] = test_metrics.get('test/cooccurrence_cosine_similarity', None)
                row['kl_divergence'] = test_metrics.get('test/cooccurrence_kl_divergence', None)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def save_comparison(self, df: pd.DataFrame, filename: str = "comparison.csv") -> Path:
        """
        Save comparison results to CSV.
        
        Args:
            df: Comparison dataframe
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\nComparison saved to {filepath}")
        return filepath
    
    def generate_summary_report(self, df: pd.DataFrame) -> str:
        """
        Generate a text summary report of the comparison.
        
        Args:
            df: Comparison dataframe
        
        Returns:
            Summary report text
        """
        report = []
        report.append("\n" + "="*100)
        report.append("VAE CONFIGURATION COMPARISON REPORT")
        report.append("="*100)
        
        report.append(f"\nTotal configurations evaluated: {len(df)}")
        
        # Loss comparison
        report.append("\n" + "-"*100)
        report.append("Loss Metrics (lower is better)")
        report.append("-"*100)
        for metric in ['test_loss', 'test_bce', 'test_kld']:
            if metric in df.columns and df[metric].notna().any():
                best_idx = df[metric].idxmin()
                best_config = df.loc[best_idx, 'configuration']
                best_value = df.loc[best_idx, metric]
                report.append(f"{metric:30s}: Best={best_config:30s} ({best_value:10.4f})")
        
        # Reconstruction metrics
        report.append("\n" + "-"*100)
        report.append("Reconstruction Metrics")
        report.append("-"*100)
        for metric in ['hamming_loss', 'jaccard_index']:
            if metric in df.columns and df[metric].notna().any():
                if 'loss' in metric:
                    best_idx = df[metric].idxmin()
                else:
                    best_idx = df[metric].idxmax()
                best_config = df.loc[best_idx, 'configuration']
                best_value = df.loc[best_idx, metric]
                report.append(f"{metric:30s}: Best={best_config:30s} ({best_value:10.4f})")
        
        # Latent space metrics
        report.append("\n" + "-"*100)
        report.append("Latent Space Metrics (higher is better)")
        report.append("-"*100)
        for metric in ['pct_active_units', 'silhouette_score']:
            if metric in df.columns and df[metric].notna().any():
                best_idx = df[metric].idxmax()
                best_config = df.loc[best_idx, 'configuration']
                best_value = df.loc[best_idx, metric]
                report.append(f"{metric:30s}: Best={best_config:30s} ({best_value:10.4f})")
        
        # Diversity metrics
        report.append("\n" + "-"*100)
        report.append("Diversity Metrics (higher is better for entropy, lower for Gini)")
        report.append("-"*100)
        for metric in ['pct_unique_combinations', 'entropy', 'gini_coefficient']:
            if metric in df.columns and df[metric].notna().any():
                if metric == 'gini_coefficient':
                    best_idx = df[metric].idxmin()
                else:
                    best_idx = df[metric].idxmax()
                best_config = df.loc[best_idx, 'configuration']
                best_value = df.loc[best_idx, metric]
                report.append(f"{metric:30s}: Best={best_config:30s} ({best_value:10.4f})")
        
        # Co-occurrence metrics
        report.append("\n" + "-"*100)
        report.append("Co-occurrence Metrics (higher is better for similarity and correlation)")
        report.append("-"*100)
        for metric in ['cosine_similarity', 'kl_divergence']:
            if metric in df.columns and df[metric].notna().any():
                if 'similarity' in metric or 'correlation' in metric:
                    best_idx = df[metric].idxmax()
                else:
                    best_idx = df[metric].idxmin()
                best_config = df.loc[best_idx, 'configuration']
                best_value = df.loc[best_idx, metric]
                report.append(f"{metric:30s}: Best={best_config:30s} ({best_value:10.4f})")
        
        report.append("\n" + "="*100)
        
        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple VAE configurations and compare results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run preset configurations
  python run_vae_comparison.py --preset standard
  
  # Run custom configurations
  python run_vae_comparison.py \\
    --config "vae_small" model=vae/small trainer=vae/default \\
    --config "vae_large" model=vae/large trainer=vae/default
  
  # Run with custom hyperparameters
  python run_vae_comparison.py \\
    --config "beta_vae_4" model=vae/beta_default beta=4.0 \\
    --config "beta_vae_10" model=vae/beta_high beta=10.0
        """
    )
    
    parser.add_argument(
        '--preset',
        type=str,
        choices=['standard', 'beta', 'all'],
        help="Run preset configurations",
    )
    parser.add_argument(
        '--config',
        action='append',
        nargs='+',
        metavar=('NAME', 'ARGS'),
        help="Custom configuration: --config <name> <args>",
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path(__file__).parent.parent.parent.parent,
        help="Project root directory",
    )
    parser.add_argument(
        '--collect-only',
        action='store_true',
        help="Only collect and compare existing results (don't run new training)",
    )
    parser.add_argument(
        '--metrics-dir',
        type=Path,
        help="Directory containing metrics files for comparison",
    )
    
    args = parser.parse_args()
    
    runner = VAEExperimentRunner(args.project_root)
    
    configurations = []
    
    # Preset configurations
    if args.preset == 'standard':
        configurations = [
            ('vae_small', {'model_variant': 'small', 'trainer_variant': 'default'}),
            ('vae_medium', {'model_variant': 'medium', 'trainer_variant': 'default'}),
            ('vae_large', {'model_variant': 'large', 'trainer_variant': 'default'}),
        ]
    elif args.preset == 'beta':
        configurations = [
            ('beta_vae_default', {'model_variant': 'beta_default', 'trainer_variant': 'default'}),
            ('beta_vae_balanced', {'model_variant': 'beta_balanced', 'trainer_variant': 'default'}),
            ('beta_vae_high', {'model_variant': 'beta_high', 'trainer_variant': 'default'}),
        ]
    elif args.preset == 'all':
        configurations = [
            ('vae_small', {'model_variant': 'small', 'trainer_variant': 'default'}),
            ('vae_medium', {'model_variant': 'medium', 'trainer_variant': 'default'}),
            ('vae_large', {'model_variant': 'large', 'trainer_variant': 'default'}),
            ('beta_vae_default', {'model_variant': 'beta_default', 'trainer_variant': 'default'}),
            ('beta_vae_balanced', {'model_variant': 'beta_balanced', 'trainer_variant': 'default'}),
            ('beta_vae_high', {'model_variant': 'beta_high', 'trainer_variant': 'default'}),
            ('beta_vae_low', {'model_variant': 'beta_low', 'trainer_variant': 'default'}),
        ]
    
    # Custom configurations
    if args.config:
        for config_args in args.config:
            config_name = config_args[0]
            kwargs = {}
            for arg in config_args[1:]:
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    kwargs[key] = value
                elif arg in ['small', 'medium', 'large', 'beta_default', 'beta_balanced', 'beta_high']:
                    kwargs['model_variant'] = arg
                elif arg in ['default', 'quick_test', 'distributed']:
                    kwargs['trainer_variant'] = arg
            configurations.append((config_name, kwargs))
    
    # Run configurations unless collect-only mode
    if not args.collect_only and configurations:
        print(f"\nRunning {len(configurations)} configuration(s)...\n")
        
        success_count = 0
        for config_name, kwargs in configurations:
            if runner.run_configuration(config_name, **kwargs):
                success_count += 1
        
        print(f"\n\nCompleted {success_count}/{len(configurations)} configurations successfully")
    
    # Collect and compare results
    print("\n\nCollecting results for comparison...")
    results = runner.collect_results(args.metrics_dir)
    
    if results:
        df = runner.compare_configurations(results)
        runner.save_comparison(df)
        
        # Print summary report
        report = runner.generate_summary_report(df)
        print(report)
        
        # Save report
        report_file = runner.output_dir / "comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to {report_file}")
    else:
        print("No results found to compare")


if __name__ == "__main__":
    main()
