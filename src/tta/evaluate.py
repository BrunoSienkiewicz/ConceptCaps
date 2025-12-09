from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from fadtk import FAD
from transformers import ClapModel, ClapProcessor

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
        
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
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
                
                # Process inputs
                audio_inputs = self.processor(
                    audios=audios,
                    return_tensors="pt",
                    sampling_rate=48000,
                    padding=True,
                ).to(self.device)
                
                text_inputs = self.processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                
                # Get embeddings
                audio_embeds = self.model.get_audio_features(**audio_inputs)
                text_embeds = self.model.get_text_features(**text_inputs)
                
                # Normalize embeddings
                audio_embeds = audio_embeds / audio_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                batch_similarities = (audio_embeds * text_embeds).sum(dim=-1)
                similarities.extend(batch_similarities.cpu().numpy().tolist())
        
        similarities = np.array(similarities)
        
        return {
            "clap_score_mean": float(similarities.mean()),
            "clap_score_std": float(similarities.std()),
            "clap_score_min": float(similarities.min()),
            "clap_score_max": float(similarities.max()),
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
        log.info("TTA evaluator initialized")
    
    def evaluate(
        self,
        generated_audio_dir: Path,
        metadata_path: Path,
        output_dir: Optional[Path] = None,
        text_column: str = "caption",
        filename_column: str = "filename",
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation.
        
        Args:
            generated_audio_dir: Directory with generated audio files
            metadata_path: Path to CSV with metadata (captions, filenames)
            reference_audio_dir: Directory with reference audio (for FAD)
            background_stats_path: Path to precomputed background stats (for FAD)
            output_dir: Directory to save evaluation results
            text_column: Column name for text prompts in metadata
            filename_column: Column name for audio filenames in metadata
            batch_size: Batch size for CLAP computation
            
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
        
        # Save results
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = output_dir / "evaluation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, indent=2, fp=f)
            log.info(f"Evaluation results saved to {results_path}")
            
            # Save per-sample CLAP scores if available
            if "clap_scores" in results:
                df_results = df.copy()
                df_results["clap_score"] = results["clap_scores"]
                df_results.to_csv(output_dir / "per_sample_scores.csv", index=False)
        
        log.info("TTA evaluation completed")
        return results


def compute_audio_quality_metrics(
    audio_paths: List[Path],
) -> Dict[str, float]:
    """Compute basic audio quality metrics.
    
    Args:
        audio_paths: List of paths to audio files
        
    Returns:
        Dictionary with audio quality metrics
    """
    import librosa
    
    durations = []
    rms_energies = []
    zero_crossing_rates = []
    
    for audio_path in tqdm(audio_paths, desc="Computing audio quality metrics"):
        try:
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Duration
            duration = librosa.get_duration(y=audio, sr=sr)
            durations.append(duration)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            rms_energies.append(rms.mean())
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zero_crossing_rates.append(zcr.mean())
            
        except Exception as e:
            log.warning(f"Failed to process {audio_path}: {e}")
            continue
    
    return {
        "avg_duration": float(np.mean(durations)),
        "avg_rms_energy": float(np.mean(rms_energies)),
        "avg_zero_crossing_rate": float(np.mean(zero_crossing_rates)),
        "duration_std": float(np.std(durations)),
    }
