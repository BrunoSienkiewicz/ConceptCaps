from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.svm import SVC

from src.tcav.model import MusicGenreClassifier
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class TCAV:
    def __init__(self, model: MusicGenreClassifier, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def get_activations(
        self, audio: torch.Tensor, batch_size: int = 32
    ) -> np.ndarray:
        """Extract bottleneck activations in batches."""
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
        """Compute directional derivatives along CAV direction."""
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
        num_runs: int = 10,
        test_size: float = 0.2,
    ) -> Dict:
        """Train CAV using SVM."""

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
            if acc > 0.60:  # Better than random + margin
                cavs.append(cav)
                scores.append(acc)

        # If no good CAVs found, return failure
        if len(cavs) == 0:
            log.warning("Warning: No CAVs with accuracy > 0.60 found")
            return {
                "cav": None,
                "accuracy": 0,
                "accuracy_std": 0,
                "num_successful_runs": 0,
            }

        # Average CAVs and renormalize
        avg_cav = np.mean(cavs, axis=0)
        avg_cav = avg_cav / (np.linalg.norm(avg_cav) + 1e-8)

        return {
            "cav": avg_cav,
            "accuracy": np.mean(scores),
            "accuracy_std": np.std(scores),
            "num_successful_runs": len(scores),
        }

    def compute_tcav_score(
        self, activations: np.ndarray, cav: np.ndarray
    ) -> float:
        """Compute TCAV score.

        Args:
            activations: Bottleneck activations
            cav: Concept activation vector
        """
        acts_normalized = normalize(activations, norm="l2")
        cav_normalized = cav / (np.linalg.norm(cav) + 1e-8)
        similarities = np.dot(acts_normalized, cav_normalized)
        return np.mean(similarities > 0.1)  # Threshold for positive alignment
