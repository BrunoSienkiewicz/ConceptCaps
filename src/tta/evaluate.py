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
                    sampling_rate=sr,
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
    
    def __init__(self, model_name: str = "laion/clap-htsat-unfused", device: str = "cuda"):
        """Initialize FAD model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run the model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        log.info(f"Loading FAD model: {model_name}")
        
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
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
            for i in tqdm(range(0, len(audio_paths), batch_size), desc="Extracting features for FAD"):
                batch_paths = audio_paths[i:i + batch_size]
                
                # Load and process audio
                audios = []
                for idx, audio_path in enumerate(batch_paths):
                    try:
                        audio, sr = librosa.load(audio_path, sr=48000, mono=True)
                        audios.append(audio)
                    except Exception as e:
                        log.warning(f"Failed to load {audio_path}: {e}")
                
                if not audios:
                    continue
                
                inputs = self.processor(
                    audio=audios,
                    sampling_rate=48000,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                
                outputs = self.model.get_audio_features(**inputs)
                features.append(outputs.cpu().numpy())
        
        if not features:
            return np.array([])
            
        return np.vstack(features)
    
    def calculate_score(
        self,
        generated_audio_paths: List[Path],
        reference_audio_paths: List[Path],
        batch_size: int = 8,
    ) -> float:
        """Compute FAD score."""
        log.info(f"Extracting features from {len(generated_audio_paths)} generated samples...")
        gen_features = self.extract_features(generated_audio_paths, batch_size)
        if len(gen_features) == 0:
            log.warning("No features extracted from generated audio")
            return float('inf')

        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        log.info(f"Extracting features from {len(reference_audio_paths)} reference samples...")
        ref_features = self.extract_features(reference_audio_paths, batch_size)
        if len(ref_features) == 0:
            log.warning("No features extracted from reference audio")
            return float('inf')

        mu_ref = np.mean(ref_features, axis=0)
        sigma_ref = np.cov(ref_features, rowvar=False)
        
        return self._calculate_frechet_distance(mu_gen, sigma_gen, mu_ref, sigma_ref)
        
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # product might be almost singular
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            log.warning(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                log.warning(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


class TTAEvaluator:
    """Comprehensive evaluator for Text-to-Audio generation."""
    
    def __init__(
        self,
        clap_model: str = "laion/clap-htsat-unfused",
        fad_model: str = "laion/clap-htsat-unfused",
        device: str = "cuda",
    ):
        """Initialize evaluator with CLAP and FAD.
        
        Args:
            clap_model: CLAP model identifier
            fad_model: FAD model identifier (defaults to using CLAP embeddings if same)
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

        # Compute FAD scores
        if compute_fad and reference_audio_dir is not None:
            log.info("Computing FAD score...")
            reference_audio_dir = Path(reference_audio_dir)
            reference_paths = []
            for ext in ['*.wav', '*.mp3', '*.flac']:
                reference_paths.extend(list(reference_audio_dir.glob(ext)))
            
            if reference_paths:
                fad_score = self.fad_scorer.calculate_score(
                    generated_audio_paths=audio_paths,
                    reference_audio_paths=reference_paths,
                    batch_size=batch_size,
                )
                results["fad_score"] = fad_score
                log.info(f"FAD Score: {fad_score:.4f}")
            else:
                log.warning(f"No reference audio files found in {reference_audio_dir}")
        elif compute_fad:
             log.warning("FAD computation requested but reference_audio_dir not provided.")
        
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
