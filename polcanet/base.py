from itertools import combinations
from random import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn as nn
from torch.nn import functional as F


class SimpleChild(nn.Module):
    """Simple recursive autoencoder"""

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 w: float = 1.0,  # Weight for this child's loss
                 child_config: dict = None):  # Configuration for next child if any
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.w = w

        # Create child if configuration provided
        self.child = None
        if child_config is not None:
            self.child = SimpleChild(**child_config)

    def compute_recursive_loss(self, x):
        """Compute reconstruction loss recursively"""
        # Current level computation

        z = self.encoder(x)

        r = self.decoder(z)

        # Compute weighted reconstruction loss for current level
        current_loss = self.w * F.mse_loss(r, x)

        # Add child's reconstruction loss if it exists
        if self.child is not None:
            current_loss = current_loss + self.child.compute_recursive_loss(z)

        return current_loss


class LossConflictAnalyzer:
    def __init__(self, loss_names: List[str] = None, rate=0.1, conflict_threshold: float = 0.1):
        """
        Initializes the LossConflictAnalyzer class.

        Args:
            loss_names (List[str], optional): List of loss names. Defaults to None.
            rate (float, optional): Rate of updates as a probability. Defaults to 0.1.
            conflict_threshold (float, optional): Threshold for conflict detection. Defaults to 0.1.
        """
        self.pairwise_similarities = {}
        self.pairwise_conflicts = {}
        self.pairwise_interactions = {}
        self.total_conflicts = 0
        self.total_interactions = 0
        self.loss_names = loss_names if loss_names else []
        # rate of updates as a probability <1 if a random number is less than this rate the analysis will be done
        # this is to limit the heavy computation of the analysis during training
        self.rate = rate
        self.conflict_threshold = conflict_threshold
        self.reset_statistics()
        self.enabled = True

    def reset_statistics(self):
        self.total_interactions = 0
        self.total_conflicts = 0
        self.pairwise_interactions = {}
        self.pairwise_conflicts = {}
        self.pairwise_similarities = {}

    def step(self, model, losses: List[torch.Tensor]) -> None:
        # if len(losses) != len(self.loss_names):
        #     raise ValueError("Number of losses must match number of loss names")

        # rate test
        if not self.enabled or np.random.rand() > self.rate:
            return

        grads = self.compute_grad(model, losses)
        self.analyze_conflicts(grads)

    @staticmethod
    def compute_grad(model, losses: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        grads = []
        for loss in losses:
            if loss == 0:
                grads.append(None)
                continue
            model.zero_grad()
            loss.backward(retain_graph=True)
            grad_dict = {name: param.grad.clone() for name, param in model.named_parameters() if
                         param.grad is not None}
            grads.append(grad_dict)
        return grads

    def analyze_conflicts(self, grads: List[Dict[str, torch.Tensor]]) -> None:
        """
        Analyze conflicts between gradients of different losses.

        Args:
            grads (List[Dict[str, torch.Tensor]]): List of gradient dictionaries for each loss.
        """
        num_losses = len(grads)

        for (i, j) in combinations(range(num_losses), 2):
            conflict_key = self.get_conflict_key(i, j)
            g_i, g_j = grads[i], grads[j]

            if g_i is not None and g_j is not None:
                similarity = self.compute_similarity(g_i, g_j)

                self.pairwise_interactions[conflict_key] = self.pairwise_interactions.get(conflict_key, 0) + 1
                self.pairwise_similarities[conflict_key] = self.pairwise_similarities.get(conflict_key, []) + [
                    similarity]

                if similarity < -self.conflict_threshold:
                    self.total_conflicts += 1
                    self.pairwise_conflicts[conflict_key] = self.pairwise_conflicts.get(conflict_key, 0) + 1

        self.total_interactions += 1

    @staticmethod
    def compute_similarity(grad1: Dict[str, torch.Tensor], grad2: Dict[str, torch.Tensor]) -> float:
        """
        Compute the cosine similarity between two gradients.
        """
        dot_product = 0
        norm1 = 0
        norm2 = 0
        for name in grad1.keys():
            if name in grad2:
                g1, g2 = grad1[name], grad2[name]
                dot_product += torch.sum(g1 * g2).item()
                norm1 += torch.sum(g1 * g1).item()
                norm2 += torch.sum(g2 * g2).item()

        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (np.sqrt(norm1) * np.sqrt(norm2))

    def get_conflict_key(self, i: int, j: int) -> Tuple[str, str]:
        return self.loss_names[i], self.loss_names[j]

    def report(self) -> Tuple[Dict[str, float], pd.DataFrame]:
        overall_conflict_rate = self.total_conflicts / max(1, self.total_interactions)

        report_dict = {'total_interactions': self.total_interactions, 'total_conflicts': self.total_conflicts,
                       'overall_conflict_rate': overall_conflict_rate, 'pairwise_data': {}}

        df_data = []

        for (loss1, loss2), interactions in self.pairwise_interactions.items():
            conflicts = self.pairwise_conflicts.get((loss1, loss2), 0)
            similarities = self.pairwise_similarities[(loss1, loss2)]
            avg_similarity = np.mean(similarities)
            conflict_rate = conflicts / interactions if interactions > 0 else 0

            if avg_similarity > self.conflict_threshold:
                relationship = "Strongly Cooperative"
            elif avg_similarity > 0:
                relationship = "Weakly Cooperative"
            elif avg_similarity > -self.conflict_threshold:
                relationship = "Weakly Conflicting"
            else:
                relationship = "Strongly Conflicting"

            report_dict['pairwise_data'][(loss1, loss2)] = {'interactions': interactions, 'conflicts': conflicts,
                                                            'conflict_rate': conflict_rate,
                                                            'avg_similarity': avg_similarity,
                                                            'relationship': relationship}

            df_data.append({'loss1': loss1, 'loss2': loss2, 'interactions': interactions, 'conflicts': conflicts,
                            'conflict_rate': conflict_rate, 'avg_similarity': avg_similarity,
                            'relationship': relationship})

        df = pd.DataFrame(df_data)
        if self.total_conflicts > 0:
            df = df.sort_values('avg_similarity').reset_index(drop=True)

        return report_dict, df


class PolcaNetLoss(nn.Module):
    """
    PolcaNetLoss is a class that extends PyTorch's Module class. It represents the loss function used for
    Principal Latent Components Analysis (Polca).

    Attributes:
        r (float): The weight for the reconstruction loss.
        c (float): The weight for the classification loss.
        alpha (float): The weight for the orthogonality loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        class_labels (list): The list of class labels.
        loss_names (dict): The dictionary of loss names.
        i (torch.Tensor): The normalized axis.
        loss_analyzer (LossConflictAnalyzer): The loss conflict analyzer.
    """

    def __init__(self, latent_dim, r=1., c=1., alpha=1e-2, beta=1e-2, gamma=1e-4, class_labels=None,
                 analyzer_rate=0.05, use_loss_analyzer=True, nest_loss=0):
        """
        Initialize PolcaNetLoss with the provided parameters.
        """
        super().__init__()
        self.latent_dim = latent_dim
        c = c if class_labels is not None else (0, 0)  # classification loss weight

        # build two dictionaries for the loss weights and the optional loss probabilities
        self.r = r if isinstance(r, tuple) else (r, 1)
        self.c = c if isinstance(c, tuple) else (c, 1)
        self.alpha = alpha if isinstance(alpha, tuple) else (alpha, 1)
        self.beta = beta if isinstance(beta, tuple) else (beta, 1)
        self.gamma = gamma if isinstance(gamma, tuple) else (gamma, 1)
        self.nest_loss = nest_loss if isinstance(nest_loss, tuple) else (nest_loss, 0)
        self.class_labels = class_labels
        self.use_loss_analyzer = use_loss_analyzer

        self.alpha_sign = 1
        if self.alpha[0] < 0.0:
            self.alpha_sign = -1
            self.alpha = abs(self.alpha[0]), self.alpha[1]

        self.loss_names = {"loss": "Total Loss"}
        self.loss_names.update({
            "rec": "Reconstruction Loss" if self.r[0] != 0 else None,
            "ort": "Orthogonality Loss" if self.alpha[0] != 0 else None,
            "com": "Center of Mass Loss" if self.beta[0] != 0 else None,
            "var": "Variance Reduction Loss" if self.gamma[0] != 0 else None,
            "nest": "Nested Loss" if self.nest_loss[0] != 0 else None,
            "class": "Classification Loss" if self.class_labels is not None else None,

        })
        self.loss_names = {k: v for k, v in self.loss_names.items() if v is not None}

        # precompute the normalized axis
        i = (torch.arange(latent_dim, dtype=torch.float32) ** 1.25) / latent_dim
        # register i as a buffer
        self.register_buffer("i", i)

        if self.use_loss_analyzer:
            self.loss_analyzer = LossConflictAnalyzer(loss_names=list(self.loss_names)[1:], rate=analyzer_rate)

    def forward(self, z, r, x, n_loss, yp=None, target=None, classifier_model=None):
        """
        Variables:
        z: latent representation
        r: reconstruction
        x: input data
        v: variance of latent representation

        Losses:
        l_rec: Reconstruction loss
        l_ort: Orthogonality loss
        l_com: Center of mass loss
        l_var: Variance regularization loss
        l_class: Classification loss

        """
        # squeeze x and r
        x = x.squeeze()
        r = r.squeeze()

        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)
        zw = z * w

        # random tests for the loss components
        t_rec = self.r[0] != 0 and self.r[1] > random()
        t_ort = self.alpha[0] != 0 and self.alpha[1] > random()
        t_com = self.beta[0] != 0 and self.beta[1] > random()
        t_var = self.gamma[0] != 0 and self.gamma[1] > random()
        t_class = self.c[0] != 0 and self.c[1] > random()

        # Robust reconstruction loss
        # The mask works by randomly zeroing out x% of the pixels in the input and the reconstruction.
        mask = torch.rand_like(r) > 0.1
        # Apply the mask to both the reconstruction and the original
        masked_r = r * mask
        masked_x = x * mask

        # Step 2: Compute L2 norm with dim=1 (since we flattened)
        # x and r can be either:
        # (batch, N) - vector case
        # (batch, N, M) - matrix case
        # (batch, C, H, W) - Images case
        # Step 1: Reshape to flatten all dimensions after batch
        # masked_x_flat = masked_x.reshape(masked_x.shape[0], -1)  # becomes (batch, N) or (batch, N*M)
        # masked_r_flat = masked_r.reshape(masked_r.shape[0], -1)  # becomes (batch, N) or (batch, N*M)
        # l_rec = torch.norm(masked_x_flat - masked_r_flat, p=2, dim=1).mean() if t_rec else torch.tensor(0.0, device=z.device,
        #                                                                                       requires_grad=True)

        l_rec = nn.functional.mse_loss(masked_r, masked_x) if t_rec else torch.tensor(0.0, device=z.device,
                                                                                              requires_grad=True)
        l_ort = self.orthogonality_loss(zw, alpha_sign=self.alpha_sign) if t_ort else 0
        l_com = self.center_of_mass_loss(w) if t_com else 0
        l_var = v.mean() if t_var else 0
        l_class = self.clustering_loss(zw, target) if t_class else 0
        # l_class = self.cross_entropy_loss(yp, target) if t_class else 0
        l_nest = n_loss if self.nest_loss[0] != 0 else 0

        # Combine losses
        loss = (
                self.r[0] * l_rec +
                self.c[0] * l_class +
                self.alpha[0] * l_ort +
                self.beta[0] * l_com +
                self.gamma[0] * l_var +
                self.nest_loss[0] * l_nest
        )

        # dict of losses:
        loss_dict = {
            "rec": (l_rec, self.r),
            "ort": (l_ort, self.alpha),
            "com": (l_com, self.beta),
            "var": (l_var, self.gamma),
            "class": (l_class, self.c),
            "nest": (l_nest, self.nest_loss),
        }
        aux_losses = [l for l, (weight, prob) in loss_dict.values() if weight != 0]

        return loss, aux_losses

    def center_of_mass_loss(self, w):
        """
        Center of mass loss
        Purpose: Concentrate information in earlier latent dimensions
        Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        """
        # Compute and normalize variance
        l_com = torch.mean(self.i * w, dim=0)
        return l_com

    @staticmethod
    def orthogonality_loss(z, eps=1e-8, alpha_sign=1):
        """ Orthogonality loss
        Purpose: Encourage latent dimensions to be uncorrelated
        Method: Penalize off-diagonal elements of the cosine similarity matrix
        """
        # Multiply by the weights and add a little additive noise to z for numerical stability.
        z = z + eps * torch.randn_like(z)
        # Normalize z along the batch dimension
        z_norm = F.normalize(z, p=2, dim=0)

        # Compute cosine similarity matrix
        s = torch.mm(z_norm.t(), z_norm)  # z_norm.t() @ z_norm = I
        s = s.clamp(-1 + eps, 1 - eps)  # clamp to avoid NaNs and numerical instability

        idx0, idx1 = torch.triu_indices(s.shape[0], s.shape[1], offset=1)  # indices of triu w/o diagonal
        cos_sim = s[idx0, idx1]
        if alpha_sign < 0:
            loss = torch.mean(1 - cos_sim.square())
        else:
            loss = torch.mean(cos_sim.square())

        return loss

    @staticmethod
    def cross_entropy_loss(yp, target):
        """
        """
        if len(target.shape) == 1 or (len(target.shape) == 2 and target.shape[1] == 1):
            # Binary and multi-class classification
            l_cross_ent = F.cross_entropy(yp, target.view(-1).long())
        elif len(target.shape) == 2 and target.shape[1] > 1:
            # Multi-label binary classification
            l_cross_ent = F.binary_cross_entropy_with_logits(yp, target.float())
        else:
            raise ValueError("Unsupported target shape")
        return l_cross_ent

    def clustering_loss(self, z, labels, eps=1e-8):
        # Use precomputed weight vector w
        z_norm = F.normalize(z, p=2, dim=1)

        # Use precomputed num_classes
        num_classes = len(self.class_labels)

        # Labels are already indices from 0 to N
        if labels.dim() == 1 or (labels.dim() == 2 and labels.shape[1] == 1):
            # Single-label case
            labels = labels.view(-1)
            label_mask = F.one_hot(labels, num_classes=num_classes).float()
        else:
            # Multi-label case
            label_mask = labels.float()  # Labels are multi-hot vectors

        # Compute class sizes
        class_sizes = label_mask.sum(dim=0)

        # Identify non-empty classes and get their indices
        non_empty_classes = class_sizes > 0
        non_empty_class_indices = non_empty_classes.nonzero(as_tuple=False).squeeze(1)

        # Check for invalid indices
        if non_empty_class_indices.numel() == 0:
            return 0

        # Filter label_mask and class_sizes to include only non-empty classes
        label_mask = label_mask[:, non_empty_class_indices]
        class_sizes = class_sizes[non_empty_class_indices]

        # Compute centroids
        centroids = torch.matmul(label_mask.t(), z_norm)
        centroids = centroids / class_sizes.unsqueeze(1)
        centroids = F.normalize(centroids, p=2, dim=1)

        # Compute similarities
        all_similarities = torch.matmul(z_norm, centroids.t())
        all_similarities = all_similarities.clamp(-1 + eps, 1 - eps)

        # Compute intra-class similarities (normalized per sample)
        num_labels_per_sample = label_mask.sum(dim=1, keepdim=True) + eps
        intra_class_similarities = ((all_similarities * label_mask).sum(dim=1, keepdim=True)) / num_labels_per_sample
        intra_class_loss = (1 - intra_class_similarities.square()).sum()

        # Compute inter-class similarities
        inter_class_similarities = all_similarities * (1 - label_mask)
        inter_class_loss = inter_class_similarities.square().sum()

        # Total loss
        total_loss = intra_class_loss + inter_class_loss

        # Normalize the loss by the number of classes and batch size
        batch_size = z_norm.size(0)
        normalized_loss = total_loss / (num_classes * batch_size)

        return normalized_loss


class EncoderWrapper(nn.Module):
    """
    Wrapper class for an encoder module.
    A Softsign activation function is applied to the output of the encoder.
    Optionally, a classifier can be added to the output of the encoder when class_labels are provided.
    """

    def __init__(self, encoder: nn.Module, latent_dim, class_labels=None):
        """
        Initializes the encoder wrapper with the encoder module and the factor scale parameter.
        :param encoder: The encoder module.
        """
        super().__init__()
        self.class_labels = class_labels
        self.latent_dim = latent_dim
        self.encoder = encoder
        layers = []

        layer = nn.Linear(latent_dim, latent_dim)
        with torch.no_grad():
            nn.init.orthogonal_(layer.weight)
            # Get the number of features (output dimension)
            n_features, n_inputs = layer.weight.shape
            # Scale the orthogonal matrix
            scales = torch.linspace(1.0, 0.1, n_features)
            scaling_factor = 0.5  # To keep initial activations within a reasonable range for Softsign
            layer.weight.mul_(scales.unsqueeze(1) * scaling_factor)

        layers.append(layer)
        layers.append(nn.Softsign())

        self.polca_bottleneck = nn.Sequential(*layers)

        if class_labels is not None:
            # Binary, multi-class and multi-label-binary classification
            self.classifier = nn.Linear(latent_dim, len(class_labels), bias=False)

    def forward(self, x):
        z = self.encoder(x)
        z = self.polca_bottleneck(z)
        if self.class_labels is not None:
            return z, self.classifier(z)

        return z
