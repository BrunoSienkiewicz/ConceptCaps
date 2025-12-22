"""Metrics saving utilities for VAE training."""
import json
from pathlib import Path
from typing import Dict, Any
import lightning as pl


class MetricsSaver:
    """Utility class to save and manage training metrics."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize MetricsSaver.
        
        Args:
            output_dir: Directory to save metrics files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train': {},
            'val': {},
            'test': {},
        }
    
    def update(self, stage: str, metrics: Dict[str, float]) -> None:
        """
        Update metrics for a specific stage.
        
        Args:
            stage: Stage name ('train', 'val', 'test')
            metrics: Dictionary of metric values
        """
        if stage not in self.metrics:
            self.metrics[stage] = {}
        self.metrics[stage].update(metrics)
    
    def save(self, filename: str = "metrics.json") -> Path:
        """
        Save metrics to JSON file.
        
        Args:
            filename: Name of the metrics file
        
        Returns:
            Path to saved metrics file
        """
        filepath = self.output_dir / filename
        
        # Convert any non-serializable values
        metrics_clean = {}
        for stage, stage_metrics in self.metrics.items():
            metrics_clean[stage] = {}
            for key, value in stage_metrics.items():
                try:
                    # Try to serialize
                    json.dumps(value)
                    metrics_clean[stage][key] = value
                except (TypeError, ValueError):
                    # Fallback for non-serializable types
                    metrics_clean[stage][key] = float(value) if isinstance(value, (int, float)) else str(value)
        
        with open(filepath, 'w') as f:
            json.dump(metrics_clean, f, indent=2)
        
        return filepath
    
    def save_summary(self, config: Dict[str, Any], filename: str = "summary.json") -> Path:
        """
        Save metrics summary with configuration.
        
        Args:
            config: Configuration dictionary
            filename: Name of the summary file
        
        Returns:
            Path to saved summary file
        """
        filepath = self.output_dir / filename
        
        # Convert config to serializable format
        config_clean = {}
        for key, value in config.items():
            try:
                json.dumps(value)
                config_clean[key] = value
            except (TypeError, ValueError):
                config_clean[key] = str(value)
        
        summary = {
            'config': config_clean,
            'metrics': self.metrics,
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filepath


class MetricsSaveCallback(pl.Callback):
    """Lightning callback to save metrics at the end of training."""
    
    def __init__(self, metrics_saver: MetricsSaver):
        """
        Initialize callback.
        
        Args:
            metrics_saver: MetricsSaver instance
        """
        self.metrics_saver = metrics_saver
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save train metrics at end of each epoch."""
        metrics = trainer.logged_metrics
        if metrics:
            self.metrics_saver.update('train', metrics)
    
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save validation metrics at end of each epoch."""
        metrics = trainer.logged_metrics
        if metrics:
            self.metrics_saver.update('val', metrics)
    
    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save test metrics at end of testing."""
        metrics = trainer.logged_metrics
        if metrics:
            self.metrics_saver.update('test', metrics)
