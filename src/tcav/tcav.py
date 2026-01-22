"""TCAV (Testing with Concept Activation Vectors) implementation for
interpretability."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.svm import SVC

from src.constants import DEFAULT_NUM_CAV_RUNS, MIN_CAV_ACCURACY
from src.tcav.model import MusicGenreClassifier
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TCAV:
    """Testing with Concept Activation Vectors for model interpretability.

    TCAV quantifies the importance of user-defined concepts for a classifier's
    predictions by learning concept activation vectors (CAVs) and measuring
    directional derivatives of predictions along these vectors.

    Args:
        model: Trained genre classifier with accessible bottleneck layer.
        device: Device to run computations on.

    Example:
        >>> tcav = TCAV(model, device="cuda")
        >>> concept_acts = tcav.get_activations(concept_audio)
        >>> random_acts = tcav.get_activations(random_audio)
        >>> cav_result = tcav.train_cav(concept_acts, random_acts)
        >>> tcav_score = tcav.compute_tcav_score(cav_result["cav"], target_audio, target_class=0)
    """

    def __init__(
        self, model: MusicGenreClassifier, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.device = device

    def get_activations(
        self, audio: torch.Tensor, batch_size: int = 32
    ) -> np.ndarray:
        """Extract bottleneck layer activations from audio samples.

        Args:
            audio: Audio tensor of shape (n_samples, n_features).
            batch_size: Batch size for processing.

        Returns:
            Activations array of shape (n_samples, bottleneck_dim).
        """
        activations = []

        for i in range(0, len(audio), batch_size):
            batch = audio[i : i + batch_size].to(self.device)
            with torch.no_grad():
                _, bottleneck = self.model(batch)
            activations.append(bottleneck.cpu().numpy())

        return np.vstack(activations)

    def get_directional_derivatives(
        self,
        audio: torch.Tensor,
        cav: np.ndarray,
        target_class: int,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Compute directional derivatives of predictions along CAV direction.

        Args:
            audio: Audio tensor to compute derivatives for.
            cav: Concept activation vector.
            target_class: Target class index for gradient computation.
            batch_size: Batch size for processing.

        Returns:
            Array of directional derivatives.
        """
        audio = audio.to(self.device)
        cav_tensor = torch.from_numpy(cav).float().to(self.device)

        derivatives = []

        for i in range(0, len(audio), batch_size):
            batch = audio[i : i + batch_size]
            batch.requires_grad = True

            logits, bottleneck = self.model(batch)
            target_logits = logits[:, target_class]

            grads = torch.autograd.grad(
                outputs=target_logits.sum(),
                inputs=bottleneck,
                create_graph=False,
            )[0]

            directional_deriv = torch.matmul(grads, cav_tensor)
            derivatives.append(directional_deriv.detach().cpu().numpy())

        return np.concatenate(derivatives)

    def train_cav(
        self,
        concept_acts: np.ndarray,
        random_acts: np.ndarray,
        num_runs: int = DEFAULT_NUM_CAV_RUNS,
        test_size: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """Train Concept Activation Vector using linear SVM.

        Args:
            concept_acts: Activations from concept examples.
            random_acts: Activations from random/counter examples.
            num_runs: Number of independent training runs for stability.
            test_size: Fraction of data for testing.

        Returns:
            Dictionary containing 'cav' (mean vector), 'accuracy' (mean test accuracy),
            and 'std' (standard deviation of accuracy across runs).
        """
        cavs, scores = [], []

        for run in range(num_runs):
            # Combine data
            X = np.vstack([concept_acts, random_acts])
            y = np.hstack(
                [np.ones(len(concept_acts)), np.zeros(len(random_acts))]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=run, stratify=y
            )

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_normalized = normalize(X_train_scaled, norm="l2")
            X_test_normalized = normalize(X_test_scaled, norm="l2")

            clf = SVC(
                kernel="linear",
                C=0.1,
                class_weight="balanced",
                random_state=run,
            )
            clf.fit(X_train_normalized, y_train)

            # CAV is coefficient of trained SVM
            cav = clf.coef_[0]
            cav = cav / (np.linalg.norm(cav) + 1e-8)

            y_pred = clf.predict(X_test_normalized)
            acc = accuracy_score(y_test, y_pred)

            # Only accept CAVs with reasonable performance
            if acc > MIN_CAV_ACCURACY:
                cavs.append(cav)
                scores.append(acc)

        # If no good CAVs found, return failure
        if len(cavs) == 0:
            log.warning(f"No CAVs with accuracy > {MIN_CAV_ACCURACY} found")
            return {
                "cav": None,
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "num_successful_runs": 0,
            }

        # Average CAVs and renormalize
        avg_cav = np.mean(cavs, axis=0)
        avg_cav = avg_cav / (np.linalg.norm(avg_cav) + 1e-8)

        return {
            "cav": avg_cav,
            "accuracy": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
            "num_successful_runs": len(scores),
        }

    def compute_tcav_score(
        self, activations: np.ndarray, cav: np.ndarray
    ) -> float:
        """Compute TCAV score as fraction of samples positively aligned with
        CAV.

        Args:
            activations: Bottleneck activations of shape (n_samples, bottleneck_dim).
            cav: Concept activation vector.

        Returns:
            TCAV score between 0 and 1.
        """
        acts_normalized = normalize(activations, norm="l2")
        cav_normalized = cav / (np.linalg.norm(cav) + 1e-8)
        similarities = np.dot(acts_normalized, cav_normalized)
        return float(
            np.mean(similarities > 0.1)
        )  # Threshold for positive alignment
