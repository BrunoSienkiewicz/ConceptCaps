from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoFeatureExtractor
from scipy.linalg import sqrtm

from src.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class CLAPScore:
    """CLAP (Contrastive Language-Audio Pretraining) score evaluator."""
    
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: str = "cuda"):
        """Initialize CLAP model.
        
        Args:
            model_name: HuggingFace model identifier for CLAP
            device: Device to run the model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info(f"Loading CLAP model: {model_name}")
        
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        log.info(f"CLAP model loaded on {self.device}")
    
    def compute_score(
        self,
        audio_paths: List[Path],
        texts: List[str],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Compute CLAP scores between audio and text.
        
        Args:
            audio_paths: List of paths to generated audio files
            texts: List of text prompts/captions
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with CLAP scores
        """
        assert len(audio_paths) == len(texts), "Number of audio files and texts must match"
        
        similarities = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(audio_paths), batch_size), desc="Computing CLAP scores"):
                batch_audio_paths = audio_paths[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                
                # Load and process audio
                audios = []
                for audio_path in batch_audio_paths:
                    audio, sr = librosa.load(audio_path, sr=48000, mono=True)
                    audios.append(audio)
                
                inputs = self.processor(
                    text=batch_texts,
                    audio=audios,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                batch_similarities = outputs.logits_per_audio.softmax(dim=-1).cpu().numpy().diagonal()
                similarities.extend(batch_similarities)
        
        similarities = np.array(similarities)
        
        return {
            "clap_score_mean": float(similarities.mean()),
            "clap_score_std": float(similarities.std()),
            "clap_score_min": float(similarities.min()),
            "clap_score_max": float(similarities.max()),
            "clap_scores": similarities.tolist(),
        }


class FADScore:
    """Fréchet Audio Distance (FAD) score evaluator."""
    
    def __init__(self, model_name: str = "google/vggish", device: str = "cuda"):
        """Initialize FAD model.
        
        Args:
            model_name: HuggingFace model identifier (e.g., "google/vggish")
            device: Device to run the model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info(f"Loading FAD model: {model_name}")
        
        # Load feature extractor model
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model.eval()
        
        # Get expected sample rate from feature extractor
        self.sample_rate = self.feature_extractor.sampling_rate
        
        log.info(f"FAD model loaded on {self.device} (sample rate: {self.sample_rate} Hz)")
    
    def extract_features(
        self,
        audio_paths: List[Path],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Extract features from audio files.
        
        Args:
            audio_paths: List of paths to audio files
            batch_size: Batch size for processing
            
        Returns:
            Array of features (n_samples, feature_dim)
        """
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting features"):
                batch_paths = audio_paths[i:i + batch_size]
                
                # Load and process audio
                audios = []
                for audio_path in batch_paths:
                    try:
                        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                        audios.append(audio)
                    except Exception as e:
                        log.warning(f"Failed to load {audio_path}: {e}")
                        continue
                
                if not audios:
                    continue
                
                # Process with feature extractor
                inputs = self.feature_extractor(
                    audios,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                
                # Extract embeddings
                outputs = self.model(**inputs)
                
                # Get last hidden state or pooler output
                if hasattr(outputs, 'last_hidden_state'):
                    # Average pool over time dimension
                    batch_features = outputs.last_hidden_state.mean(dim=1)
                elif hasattr(outputs, 'pooler_output'):
                    batch_features = outputs.pooler_output
                else:
                    # Fallback: use the first output
                    batch_features = outputs[0]
                    if len(batch_features.shape) > 2:
                        batch_features = batch_features.mean(dim=1)
                
                features.append(batch_features.cpu().numpy())
        
        return np.vstack(features)
    
    def compute_statistics(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and covariance of features.
        
        Args:
            features: Feature array (n_samples, feature_dim)
            
        Returns:
            Tuple of (mean, covariance)
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fad(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Calculate Fréchet Audio Distance.
        
        Args:
            mu1: Mean of first distribution
            sigma1: Covariance of first distribution
            mu2: Mean of second distribution
            sigma2: Covariance of second distribution
            eps: Small value for numerical stability
            
        Returns:
            FAD score
        """
        # Calculate squared difference of means
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            log.warning("FAD calculation resulted in non-finite values, adding epsilon to diagonal")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Handle complex numbers (numerical errors)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                log.warning(f"Imaginary component {m} in FAD calculation")
            covmean = covmean.real
        
        fad = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fad)
    
    def compute_score(
        self,
        generated_audio_paths: List[Path],
        reference_audio_paths: List[Path],
        batch_size: int = 8,
    ) -> Dict[str, float]:
        """Compute FAD score between generated and reference audio.
        
        Args:
            generated_audio_paths: List of paths to generated audio files
            reference_audio_paths: List of paths to reference audio files
            batch_size: Batch size for feature extraction
            
        Returns:
            Dictionary with FAD score
        """
        log.info(f"Computing FAD for {len(generated_audio_paths)} generated samples vs {len(reference_audio_paths)} reference samples")
        
        # Extract features
        log.info("Extracting features from generated audio...")
        gen_features = self.extract_features(generated_audio_paths, batch_size)
        
        log.info("Extracting features from reference audio...")
        ref_features = self.extract_features(reference_audio_paths, batch_size)
        
        # Compute statistics
        log.info("Computing statistics...")
        mu_gen, sigma_gen = self.compute_statistics(gen_features)
        mu_ref, sigma_ref = self.compute_statistics(ref_features)
        
        # Calculate FAD
        fad_score = self.calculate_fad(mu_gen, sigma_gen, mu_ref, sigma_ref)
        
        return {
            "fad_score": fad_score,
            "n_generated": len(generated_audio_paths),
            "n_reference": len(reference_audio_paths),
        }


class TTAEvaluator:
    """Comprehensive evaluator for Text-to-Audio generation."""
    
    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        fad_model: str = "google/vggish",
        device: str = "cuda",
    ):
        """Initialize evaluator with CLAP and FAD.
        
        Args:
            clap_model: CLAP model identifier
            fad_model: Model to use for FAD computation
            device: Device to run models on
        """
        self.device = device
        
        log.info("Initializing TTA evaluator...")
        self.clap_scorer = CLAPScore(model_name=clap_model, device=device)
        self.fad_scorer = FADScore(model_name=fad_model, device=device)
        log.info("TTA evaluator initialized")
    
    def evaluate(
        self,
        generated_audio_dir: Path,
        metadata_path: Path,
        reference_audio_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        text_column: str = "caption",
        filename_column: str = "filename",
        batch_size: int = 8,
        compute_fad: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation.
        
        Args:
            generated_audio_dir: Directory with generated audio files
            metadata_path: Path to CSV with metadata (captions, filenames)
            reference_audio_dir: Directory with reference audio (for FAD)
            output_dir: Directory to save evaluation results
            text_column: Column name for text prompts in metadata
            filename_column: Column name for audio filenames in metadata
            batch_size: Batch size for CLAP computation
            compute_fad: Whether to compute FAD score (requires reference_audio_dir)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        log.info("Starting TTA evaluation...")
        
        # Load metadata
        log.info(f"Loading metadata from {metadata_path}")
        df = pd.read_csv(metadata_path)
        
        # Get audio paths and texts
        audio_paths = [generated_audio_dir / fname for fname in df[filename_column]]
        texts = df[text_column].tolist()
        
        # Filter out missing audio files
        valid_indices = [i for i, path in enumerate(audio_paths) if path.exists()]
        if len(valid_indices) < len(audio_paths):
            log.warning(
                f"Found {len(audio_paths) - len(valid_indices)} missing audio files. "
                "Evaluating only existing files."
            )
            audio_paths = [audio_paths[i] for i in valid_indices]
            texts = [texts[i] for i in valid_indices]
        
        log.info(f"Evaluating {len(audio_paths)} audio samples")
        
        results = {}
        
        # Compute CLAP scores
        log.info("Computing CLAP scores...")
        clap_results = self.clap_scorer.compute_score(
            audio_paths=audio_paths,
            texts=texts,
            batch_size=batch_size,
        )
        results.update(clap_results)
        log.info(f"CLAP Score (mean): {clap_results['clap_score_mean']:.4f}")
        
        # Compute FAD score if reference audio is provided
        if compute_fad and reference_audio_dir is not None:
            log.info("Computing FAD score...")
            reference_audio_dir = Path(reference_audio_dir)
            
            # Get all audio files from reference directory
            reference_paths = []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                reference_paths.extend(list(reference_audio_dir.glob(ext)))
            
            if not reference_paths:
                log.warning(f"No reference audio files found in {reference_audio_dir}")
            else:
                log.info(f"Found {len(reference_paths)} reference audio files")
                fad_results = self.fad_scorer.compute_score(
                    generated_audio_paths=audio_paths,
                    reference_audio_paths=reference_paths,
                    batch_size=batch_size,
                )
                results.update(fad_results)
                log.info(f"FAD Score: {fad_results['fad_score']:.4f}")
        elif compute_fad:
            log.warning("FAD computation requested but no reference_audio_dir provided")
        
        # Save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = output_dir / "evaluation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, indent=2, fp=f)
            log.info(f"Evaluation results saved to {results_path}")
            
            # Save per-sample CLAP scores if available
            df_results = df.copy()
            df_results["clap_score"] = results["clap_scores"]
            df_results.to_csv(output_dir / "per_sample_scores.csv", index=False)
        
        log.info("TTA evaluation completed")
        return results
