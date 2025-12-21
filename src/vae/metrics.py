"""Evaluation metrics for VAE training."""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
from collections import Counter


class VAEMetrics:
    """Compute evaluation metrics for VAE training."""
    
    @staticmethod
    def hamming_loss(recon: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Compute Hamming loss: fraction of labels that are incorrectly predicted.
        
        Args:
            recon: Reconstructed output (batch_size, num_labels)
            target: Target labels (batch_size, num_labels)
            threshold: Threshold for binarizing reconstruction (default: 0.5)
        
        Returns:
            Hamming loss (lower is better)
        """
        # Binarize reconstruction
        recon_binary = (recon > threshold).float()
        
        # Compute element-wise mismatch
        mismatch = (recon_binary != target).float()
        
        # Average across all dimensions
        hamming_loss = mismatch.mean().item()
        
        return hamming_loss
    
    @staticmethod
    def jaccard_index(recon: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Compute Jaccard Index: |intersection| / |union| over all samples and labels.
        
        Also known as Intersection over Union (IoU).
        
        Args:
            recon: Reconstructed output (batch_size, num_labels)
            target: Target labels (batch_size, num_labels)
            threshold: Threshold for binarizing reconstruction (default: 0.5)
        
        Returns:
            Jaccard Index (higher is better, range [0, 1])
        """
        # Binarize reconstruction
        recon_binary = (recon > threshold).float()
        
        # Compute intersection and union
        intersection = torch.sum(recon_binary * target)
        union = torch.sum(torch.clamp(recon_binary + target, max=1.0))
        
        # Avoid division by zero
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        jaccard_idx = (intersection / union).item()
        
        return jaccard_idx
    
    @staticmethod
    def active_units(latent: torch.Tensor, threshold: float = 0.0) -> Dict[str, float]:
        """
        Compute active units statistics in the latent space.
        
        Active units are latent dimensions that have significant variance across the batch.
        
        Args:
            latent: Latent representations (batch_size, latent_dim)
            threshold: Threshold for standard deviation to consider a unit active (default: 0.0)
        
        Returns:
            Dictionary with:
                - 'num_active_units': Number of active units
                - 'pct_active_units': Percentage of active units
                - 'mean_variance': Mean variance across dimensions
                - 'min_variance': Minimum variance
                - 'max_variance': Maximum variance
        """
        # Compute variance per dimension
        variance = torch.var(latent, dim=0)
        std = torch.std(latent, dim=0)
        
        # Count active units (those with std > threshold)
        active_units = (std > threshold).sum().item()
        total_units = latent.shape[1]
        pct_active = (active_units / total_units) * 100.0
        
        return {
            'num_active_units': active_units,
            'pct_active_units': pct_active,
            'total_units': total_units,
            'mean_variance': variance.mean().item(),
            'min_variance': variance.min().item(),
            'max_variance': variance.max().item(),
            'mean_std': std.mean().item(),
        }
    
    @staticmethod
    def cluster_separability(latent: torch.Tensor, labels: torch.Tensor, metric: str = 'silhouette') -> float:
        """
        Compute cluster separability in latent space using different metrics.
        
        Args:
            latent: Latent representations (batch_size, latent_dim)
            labels: Cluster labels (batch_size,) or multi-hot vectors (batch_size, num_labels)
            metric: Metric to use - 'silhouette', 'calinski_harabasz', 'davies_bouldin'
        
        Returns:
            Cluster separability score
        """
        # Convert to numpy
        latent_np = latent.cpu().detach().numpy()
        
        # Handle multi-hot labels by finding dominant label per sample
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            labels_np = labels.argmax(dim=1).cpu().detach().numpy()
        else:
            labels_np = labels.cpu().detach().numpy().flatten()
        
        # Remove samples with no labels (all zeros) if applicable
        if (labels == 0).all(dim=1 if len(labels.shape) > 1 else 0).any():
            valid_idx = ~(labels == 0).all(dim=1 if len(labels.shape) > 1 else 0)
            latent_np = latent_np[valid_idx]
            labels_np = labels_np[valid_idx]
        
        # Handle case with insufficient samples or single class
        if len(np.unique(labels_np)) < 2 or len(latent_np) < 2:
            return 0.0
        
        return VAEMetrics._silhouette_score(latent_np, labels_np)
    
    @staticmethod
    def _silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute Silhouette Coefficient (higher is better, range [-1, 1]).
        Measures how similar an object is to its own cluster vs other clusters.
        """
        unique_labels = np.unique(labels)
        silhouette_scores = []
        
        for i in range(len(X)):
            sample = X[i]
            label = labels[i]
            
            # Intra-cluster distance (a): mean distance to samples in same cluster
            same_cluster_mask = labels == label
            same_cluster_samples = X[same_cluster_mask]
            
            if len(same_cluster_samples) > 1:
                a = np.mean(np.linalg.norm(sample - same_cluster_samples, axis=1))
            else:
                a = 0.0
            
            # Inter-cluster distance (b): min mean distance to samples in other clusters
            b = float('inf')
            for other_label in unique_labels:
                if other_label != label:
                    other_cluster_mask = labels == other_label
                    other_cluster_samples = X[other_cluster_mask]
                    if len(other_cluster_samples) > 0:
                        mean_dist = np.mean(np.linalg.norm(sample - other_cluster_samples, axis=1))
                        b = min(b, mean_dist)
            
            # Silhouette coefficient for this sample
            if b == float('inf'):
                s_i = 0.0
            else:
                s_i = (b - a) / max(a, b)
            
            silhouette_scores.append(s_i)
        
        return np.mean(silhouette_scores)
    
    @staticmethod
    def compute_all_metrics(
        recon: torch.Tensor,
        target: torch.Tensor,
        latent: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            recon: Reconstructed output
            target: Target labels
            latent: Latent representations
            threshold: Threshold for binarization
        
        Returns:
            Dictionary with all computed metrics
        """
        metrics = {}
        
        # Reconstruction metrics
        metrics['hamming_loss'] = VAEMetrics.hamming_loss(recon, target, threshold)
        metrics['jaccard_index'] = VAEMetrics.jaccard_index(recon, target, threshold)
        
        # Latent space metrics
        active_units_metrics = VAEMetrics.active_units(latent)
        metrics.update({f'active_units_{k}': v for k, v in active_units_metrics.items()})
        
        # Cluster separability (using argmax of target as pseudo-labels)
        target_labels = target.argmax(dim=1) if len(target.shape) > 1 else target
        metrics['silhouette_score'] = VAEMetrics.cluster_separability(latent, target_labels, 'silhouette')
        
        return metrics
    
    @staticmethod
    def tag_combination_diversity(predictions: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute diversity metrics to detect model collapse (generating same tag combinations).
        
        Args:
            predictions: Binary predictions (batch_size, num_tags)
            threshold: Threshold for binarization
        
        Returns:
            Dictionary with diversity metrics:
                - 'unique_combinations': Number of unique tag combinations
                - 'pct_unique_combinations': Percentage of unique combinations
                - 'entropy': Entropy of combination distribution (higher = more diverse)
                - 'gini_coefficient': Gini coefficient (0 = uniform, 1 = all in one)
        """
        # Binarize predictions
        binary_preds = (predictions > threshold).cpu().numpy()
        batch_size = binary_preds.shape[0]
        
        # Convert each sample to tuple for hashing
        combinations = [tuple(row) for row in binary_preds]
        
        # Count unique combinations
        unique_combos = len(set(combinations))
        pct_unique = (unique_combos / batch_size) * 100.0
        
        # Compute entropy of combination distribution
        combo_counts = Counter(combinations)
        probabilities = np.array(list(combo_counts.values())) / batch_size
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Compute Gini coefficient (measure of inequality)
        # Gini = 0 means perfect equality (all combinations appear equally)
        # Gini = 1 means perfect inequality (one combination dominates)
        sorted_counts = np.sort(np.array(list(combo_counts.values())))
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        return {
            'unique_combinations': unique_combos,
            'pct_unique_combinations': pct_unique,
            'entropy': entropy,
            'gini_coefficient': gini,
        }
    
    @staticmethod
    def tag_cooccurrence_matrix(predictions: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """
        Compute co-occurrence matrix for tag pairs.
        
        Args:
            predictions: Binary predictions (batch_size, num_tags)
            threshold: Threshold for binarization
        
        Returns:
            Co-occurrence matrix (num_tags, num_tags) where [i, j] = count of samples with both tag i and j
        """
        binary_preds = (predictions > threshold).cpu().numpy()
        num_tags = binary_preds.shape[1]
        
        # Initialize co-occurrence matrix
        cooccurrence = np.zeros((num_tags, num_tags), dtype=int)
        
        # Compute co-occurrence
        for sample in binary_preds:
            active_tags = np.where(sample > 0)[0]
            for i in active_tags:
                for j in active_tags:
                    cooccurrence[i, j] += 1
        
        return cooccurrence
    
    @staticmethod
    def cooccurrence_comparison(
        predicted_cooccurrence: np.ndarray,
        original_cooccurrence: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compare co-occurrence matrices between predictions and original data.
        
        Args:
            predicted_cooccurrence: Co-occurrence matrix from model predictions
            original_cooccurrence: Co-occurrence matrix from original dataset
        
        Returns:
            Dictionary with comparison metrics:
                - 'cosine_similarity': Cosine similarity between flattened matrices
                - 'kl_divergence': KL divergence between normalized matrices
                - 'hellinger_distance': Hellinger distance between distributions
                - 'pearson_correlation': Pearson correlation between matrices
        """
        # Flatten matrices
        pred_flat = predicted_cooccurrence.flatten().astype(float)
        orig_flat = original_cooccurrence.flatten().astype(float)
        
        # Normalize to prevent division by zero
        pred_flat = pred_flat / (np.sum(pred_flat) + 1e-10)
        orig_flat = orig_flat / (np.sum(orig_flat) + 1e-10)
        
        metrics = {}
        
        # Cosine similarity
        cosine_sim = np.dot(pred_flat, orig_flat) / (
            np.linalg.norm(pred_flat) * np.linalg.norm(orig_flat) + 1e-10
        )
        metrics['cosine_similarity'] = float(cosine_sim)
        
        # KL divergence
        kl_div = np.sum(orig_flat * np.log((orig_flat + 1e-10) / (pred_flat + 1e-10)))
        metrics['kl_divergence'] = float(kl_div)
        
        # Hellinger distance
        hellinger = np.sqrt(0.5 * np.sum((np.sqrt(pred_flat) - np.sqrt(orig_flat)) ** 2))
        metrics['hellinger_distance'] = float(hellinger)
        
        # Pearson correlation
        pred_reshaped = predicted_cooccurrence.flatten().astype(float)
        orig_reshaped = original_cooccurrence.flatten().astype(float)
        
        if len(pred_reshaped) > 1 and np.std(pred_reshaped) > 0 and np.std(orig_reshaped) > 0:
            pearson = np.corrcoef(pred_reshaped, orig_reshaped)[0, 1]
            metrics['pearson_correlation'] = float(pearson)
        else:
            metrics['pearson_correlation'] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_diversity_metrics(
        recon: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Compute all diversity-related metrics.
        
        Args:
            recon: Reconstructed output
            target: Target labels
            threshold: Threshold for binarization
        
        Returns:
            Dictionary with all diversity metrics
        """
        metrics = {}
        
        # Tag combination diversity
        diversity = VAEMetrics.tag_combination_diversity(recon, threshold)
        metrics.update({f'diversity_{k}': v for k, v in diversity.items()})
        
        # Co-occurrence analysis
        pred_cooccurrence = VAEMetrics.tag_cooccurrence_matrix(recon, threshold)
        target_cooccurrence = VAEMetrics.tag_cooccurrence_matrix(target, threshold=0.5)
        
        cooccurrence_metrics = VAEMetrics.cooccurrence_comparison(
            pred_cooccurrence, target_cooccurrence
        )
        metrics.update({f'cooccurrence_{k}': v for k, v in cooccurrence_metrics.items()})
        
        return metrics
