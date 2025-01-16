import gc
from collections import defaultdict
from collections.abc import Sequence
from itertools import batched
from random import random
from typing import Union, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from IPython.core.display_functions import clear_output
from IPython.display import display
from ipywidgets.widgets import Output
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from polcanet.aencoders import FastTensorDataLoader
from polcanet.base import EncoderWrapper, SimpleChild, PolcaNetLoss
from polcanet.utils import custom_float_format


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network encoder
    that is used for Principal Latent Components Analysis (Polca).

    Parameters:
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        latent_dim (int): The number of latent dimensions.
        r (float): The weight for the reconstruction loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        c (float): The weight for the classification loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        alpha (float): The weight for the orthogonality loss.
        beta (float): The weight for the center of mass loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        gamma (float): The weight for the low variance loss, optionally if tuple the second number is the probability
         of evaluating the loss.
        class_labels (list): The list of class labels.
        analyzer_rate (float): The rate of updates for the loss conflict analyzer.
        std_noise (float): The standard deviation of noise to add to the input data.
        blend_prob (float): The probability of blending the latent space with the mean.
        use_loss_analyzer (bool): Whether to use the loss conflict analyzer.
        child_config (dict): The configuration for nested autoencoder.

    Attributes:
        device (str): The device to run the encoder on ("cpu" or "cuda").
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        latent_dim (int): The number of latent dimensions.
        class_labels (list): The list of class labels.
        polca_loss (polcanet.PolcaNetLoss): The loss function.
        mu (torch.Tensor): The mean values of the latent space.
        std (torch.Tensor): The standard deviation values of the latent space.
        std_noise (float): The standard deviation of noise to add to the input data.
        blend_prob (float): The probability of blending the latent space with the mean.
        use_loss_analyzer (bool): Whether to use the loss conflict analyzer.


    Methods:
        forward(x): Encodes the input and decodes the result.
        predict(in_x, w): Encodes and decodes the input.
        encode(in_x): Encodes the input and adds the mean to the latent space.
        decode(z, w): Decodes the latent space.
        r_error(in_x, w): Computes the reconstruction error.
        to_device(device): Moves the encoder to the given device.
        to(device): Moves the encoder to the given device.
        train_model(data, batch_size, num_epochs, report_freq, lr): Trains the model.
        fitter_data_loader(data_loader, num_epochs, lr): Fits the model using a DataLoader.
        fitter_in_memory(data, y, batch_size, num_epochs, lr): Fits the model using in-memory data.
        learn_step(x, optimizer, y): Performs a training step.
        val_step(x, y): Performs a validation step.
        add_noise(x): Adds noise to the input data when training.
        stochastic_latent_blend(z, mu, std, batch_prob, temperature): Transforms the latent space by
        probabilistically stochastically blending with the mean when training.


    Usage:
        data = np.random.rand(100, 784)
        latent_dim = 32
        model = PolcaNet(encoder, decoder, latent_dim=latent_dim, alpha=1.0, beta=1.0, gamma=1.0, class_labels=None)
        model.to_device("cuda")
        model.train_model(data, batch_size=256, num_epochs=10000, report_freq=20, lr=1e-3)
        model.r_error(data)
        latents, reconstructed = model.predict(data)

    """

    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 latent_dim: int,
                 r: float = 1.,
                 c: float = 1.,
                 alpha: float | Tuple[float, float] = 1e-2,
                 beta: float | Tuple[float, float] = 1e-2,
                 gamma: float | Tuple[float, float] = 1e-8,
                 class_labels=None,
                 analyzer_rate: float = 0.05,
                 std_noise: float = 0.0,
                 blend_prob: float = 0.5,
                 use_loss_analyzer: bool = True,
                 child_config: dict = None,
                 ):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = class_labels
        self.std_noise = std_noise
        self.blend_prob = blend_prob

        self.encoder = EncoderWrapper(encoder, latent_dim=latent_dim, class_labels=class_labels)
        self.decoder = decoder

        # Create child if configuration provided
        self.child = None
        if child_config is not None:
            self.child = SimpleChild(**child_config)

        if hasattr(self.encoder, "decoder"):
            # remove the attribute decoder from the encoder
            self.encoder.decoder = None
            del self.encoder.decoder

        if hasattr(self.decoder, "encoder"):
            # remove the attribute encoder from the decoder
            self.decoder.encoder = None
            del self.decoder.encoder

        self.use_loss_analyzer = use_loss_analyzer
        self.polca_loss = PolcaNetLoss(
            latent_dim=latent_dim,
            r=r,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            class_labels=class_labels,
            analyzer_rate=analyzer_rate,
            use_loss_analyzer=use_loss_analyzer,
            nest_loss=1 if self.child is not None else 0,

        )
        self.mu = None
        self.std = None

    def forward(self, x):
        z = self.encoder(x)
        r = self.decoder(z) if self.class_labels is None else self.decoder(z[0])
        return z, r

    def predict(self, in_x, mask=None, n_components=None):
        self.eval()
        z = self.encode(in_x)
        z = z[:, :n_components] if n_components is not None else z
        r = self.decode(z, mask=mask)
        return z, r

    def predict_batched(self, x, batch_size=1024, mask=None, n_components=None):
        results = [self.predict(np.array(batch), mask=mask, n_components=n_components) for batch in
                   batched(x, batch_size)]
        latents, reconstructions = map(np.concatenate, zip(*results))
        return latents, reconstructions

    def encode(self, in_x):
        self.eval()
        with torch.no_grad():
            if isinstance(in_x, torch.Tensor):
                x = in_x.detach().clone().to(self.device)
            else:
                x = torch.tensor(in_x, dtype=torch.float32, device=self.device, requires_grad=False)
            z = self.encoder(x)
            if self.class_labels is not None:
                return z[0].cpu().numpy()

        return z.cpu().numpy()

    def decode(self, z, mask=None):
        self.eval()
        with torch.no_grad():
            z = torch.as_tensor(z, dtype=torch.float32, device=self.device)

            # pad with mean values when the latent space is smaller than the expected size
            if z.size(1) < self.latent_dim:
                padding = self.mu[z.size(1):].unsqueeze(0).expand(z.size(0), -1)
                z = torch.cat([z, padding], dim=1)

            # pad with mean when mask is provided for the latent space
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
                if mask.size(0) < self.latent_dim:
                    mask = torch.nn.functional.pad(mask, (0, self.latent_dim - mask.size(0)))
                mask = mask.unsqueeze(0).expand(z.size(0), -1)
                z = mask * z + (1 - mask) * self.mu.unsqueeze(0)

            return self.decoder(z).cpu().numpy()

    def rec_error(self, in_x, mask=None, n_components=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, mask=mask, n_components=n_components)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def train_model(self, data, y=None, val_data=None, val_y=None, batch_size=None,
                    num_epochs=100, report_freq=10, lr=1e-3, weight_decay=1e-2, verbose=1, optimizer=None):
        """
        Train the model using the given data.
        """
        train_metrics = list()
        val_metrics = list()
        # Create the optimizer if not provided
        free_optimizer = False
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                          weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-6)
            free_optimizer = True

        fitter = self.fitter(optimizer=optimizer,
                             data=data,
                             y=y,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             val_data=val_data,
                             val_y=val_y,
                             )

        epoch_progress = tqdm(fitter,
                              desc="epoch", leave=True, total=num_epochs, mininterval=1.0, disable=bool(not verbose))
        metrics = {}
        v_metrics = None
        epoch = 0
        output_widget = Output()
        display(output_widget)
        df_train_val = None
        try:
            for epoch, metrics, v_metrics in epoch_progress:
                train_metrics.append({**metrics, **{"epoch": epoch + 1, "split": "train"}})
                if v_metrics is not None:
                    val_metrics.append({**v_metrics, **{"epoch": epoch + 1, "split": "val"}})

                if epoch % report_freq == 0 and bool(verbose):
                    # epoch_progress.set_postfix(metrics)

                    metrics.update(**{"epoch": epoch + 1, "split": "train"})
                    if v_metrics is not None:
                        v_metrics.update(**{"epoch": epoch + 1, "split": "val"})

                    metrics_list = [metrics, v_metrics]
                    new_df_train_val = pd.DataFrame.from_records(metrics_list).set_index("split")
                    new_df_train_val.drop(columns=["epoch"], inplace=True, errors="ignore")
                    if df_train_val is None:
                        df_train_val = new_df_train_val
                    else:
                        # we will put only in df_train_val the values from new_df_train_val that are nonzero.
                        df_train_val = new_df_train_val.where(new_df_train_val != 0, df_train_val)

                    with output_widget:
                        clear_output(wait=True)
                        print(df_train_val.map(custom_float_format).to_string())

        except KeyboardInterrupt:
            print("Training interrupted by user.")

        finally:

            # Print final metrics if available
            if bool(verbose) and metrics:
                print("\nFinal Train metrics at epoch:", epoch + 1)
                print("=" * 40)
                for k, v in metrics.items():
                    if k not in ["epoch", "split"]:
                        print(f"{self.polca_loss.loss_names[k]}: {v:.4g}")

                if v_metrics:
                    print("\nFinal Validation metrics at epoch:", epoch + 1)
                    print("=" * 40)
                    for k, v in v_metrics.items():
                        if k not in ["epoch", "split"]:
                            print(f"{self.polca_loss.loss_names[k]}: {v:.4g}")

        if free_optimizer:
            del optimizer

        del metrics
        del v_metrics
        del fitter
        del df_train_val
        gc.collect()
        torch.cuda.empty_cache()

        return None

    def prepare_training_data(self,
                              data: Union[torch.Tensor, np.ndarray, DataLoader],
                              y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              val_data: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              val_y: Optional[Union[torch.Tensor, np.ndarray]] = None,
                              batch_size: int = 256,
                              use_fast: bool = True
                              ) -> Tuple[DataLoader, Optional[DataLoader]]:
        def create_loader(data, labels=None, shuffle=True):
            data = torch.as_tensor(data, dtype=torch.float32, device=self.device)
            if labels is not None:
                labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)

            if use_fast:
                return FastTensorDataLoader(data,
                                            labels,
                                            batch_size=batch_size,
                                            shuffle=shuffle) if labels is not None else FastTensorDataLoader(data,
                                                                                                             batch_size=batch_size,
                                                                                                             shuffle=shuffle)
            else:
                dataset = TensorDataset(data, labels) if labels is not None else TensorDataset(data)
                return DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=0,
                                  pin_memory=False,
                                  drop_last=True,
                                  )

        # Prepare training data
        if isinstance(data, DataLoader):
            data_loader = data
        else:
            data_loader = create_loader(data, y, shuffle=True)

        # Prepare validation data
        val_loader = None
        if val_data is not None:
            if isinstance(val_data, DataLoader):
                val_loader = val_data
            else:
                val_loader = create_loader(val_data, val_y, shuffle=False)

        return data_loader, val_loader

    def fitter(self, optimizer, data, y=None, batch_size=512, num_epochs=100, val_data=None, val_y=None):
        self.train()
        is_data_loader = isinstance(data, torch.utils.data.DataLoader)
        if not is_data_loader:
            data = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float32)
            data = data.to(self.device)

            if val_data is not None:
                val_data = val_data if isinstance(val_data, torch.Tensor) else torch.tensor(val_data,
                                                                                            dtype=torch.float32)
                val_data = val_data.to(self.device)

            if y is not None:
                y = y if isinstance(y, torch.Tensor) else torch.tensor(y, dtype=torch.int64)
                y = y.to(self.device)

            if val_y is not None:
                val_y = val_y if isinstance(val_y, torch.Tensor) else torch.tensor(val_y, dtype=torch.int64)
                val_y = val_y.to(self.device)

            num_samples = len(data)
            adjusted_num_samples = (num_samples // batch_size) * batch_size

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            val_losses = defaultdict(list)

            if is_data_loader:
                batches = data
            else:
                indices = torch.randperm(num_samples)[:adjusted_num_samples]
                batches = range(0, adjusted_num_samples, batch_size)

            for batch in batches:
                if is_data_loader:
                    x = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if self.class_labels is not None else None
                else:
                    batch_indices = indices[batch:batch + batch_size]
                    x = data[batch_indices]
                    targets = y[batch_indices] if y is not None else None

                loss, aux_losses = self.learn_step(x=x, optimizer=optimizer, y=targets)
                losses["loss"].append(loss.item())
                for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                    l_value = 0 if not aux_losses[idx] else aux_losses[idx].item()
                    losses[name].append(l_value)

            metrics = {name: np.mean(losses[name]) for name in losses}

            val_metrics = None
            if val_data is not None:
                if is_data_loader:
                    val_batches = val_data
                    for val_batch in val_batches:
                        x = val_batch[0].to(self.device)
                        targets = val_batch[1].to(self.device) if self.class_labels is not None else None
                        val_loss, val_aux_losses = self.val_step(x=x, y=targets)
                        val_losses["loss"].append(val_loss.item())
                        for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                            l_value = 0 if not val_aux_losses[idx] else val_aux_losses[idx].item()
                            val_losses[name].append(l_value)
                else:
                    val_loss, val_aux_losses = self.val_step(x=val_data, y=val_y)
                    val_losses["loss"].append(val_loss.item())
                    for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                        l_value = 0 if not val_aux_losses[idx] else val_aux_losses[idx].item()
                        val_losses[name].append(l_value)

                val_metrics = {name: np.mean(val_losses[name]) for name in val_losses}

            yield epoch, metrics, val_metrics

        del losses, val_losses
        data = data.cpu()
        if val_data is not None:
            val_data = val_data.cpu()

    def run_epoch(self, data_loader, optimizer):

        losses = defaultdict(list)
        for batch in data_loader:
            x, targets = batch if isinstance(batch, Sequence) and len(batch) == 2 else (batch, None)
            x = x[0] if isinstance(x, Sequence) else x
            x = x.to(self.device)
            if targets is not None:
                targets = targets.to(self.device) if self.class_labels is not None else None

            # with torch.amp.autocast('cuda'):
            if optimizer is not None:
                loss, aux_losses = self.learn_step(x=x, optimizer=optimizer, y=targets)
            else:
                loss, aux_losses = self.val_step(x=x, y=targets)

            losses["loss"].append(loss.item())
            for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                l_value = 0 if not aux_losses[idx] else aux_losses[idx].item()
                losses[name].append(l_value)

        metrics = {name: np.mean(losses[name]) for name in losses}
        return metrics

    def learn_step(self, x, optimizer, y=None):
        self.train()
        # add small noise to x to make regularization and a little data augmentation x has values in 0,1
        x = self.add_noise(x) if self.std_noise > 0 else x
        z, r = self.forward(x)
        yp = z[1] if self.class_labels is not None else None
        z = z[0] if self.class_labels is not None else z

        with torch.no_grad():
            self.mu = torch.mean(z, dim=0) if self.mu is None else self.mu + 0.01 * (torch.mean(z, dim=0) - self.mu)
            self.std = torch.std(z, dim=0) if self.std is None else self.std + 0.01 * (torch.std(z, dim=0) - self.std)

        if random() < self.blend_prob:
            z = self.stochastic_latent_blend(z, self.mu, self.std, batch_prob=self.blend_prob, temperature=1.0)

        # Add child's loss if it exists
        nest_loss = self.child.compute_recursive_loss(z) if self.child is not None else 0
        classifier_module = self.encoder.classifier if self.class_labels is not None else None
        loss, aux_losses = self.polca_loss(z, r, x, nest_loss, yp=yp, target=y, classifier_model=classifier_module)

        # Analyze loss conflicts
        if self.use_loss_analyzer:
            self.polca_loss.loss_analyzer.step(self, aux_losses)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return loss, aux_losses

    def val_step(self, x, y=None):
        self.eval()
        with torch.no_grad():
            z, r = self.forward(x)
            yp = z[1] if self.class_labels is not None else None
            z = z[0] if self.class_labels is not None else z

            # Add child's loss if it exists
            nest_loss = self.child.compute_recursive_loss(z) if self.child is not None else 0
            classifier_module = self.encoder.classifier if self.class_labels is not None else None
            loss, aux_losses = self.polca_loss(z, r, x, nest_loss, yp=yp, target=y, classifier_model=classifier_module)

            return loss, aux_losses

    def add_noise(self, x):
        x = x + torch.randn_like(x) * self.std_noise  # add noise to x
        return x

    @staticmethod
    def stochastic_latent_blend(z, mu, std, batch_prob=0.5, temperature=1.0):
        """
        Transform z by probabilistically blending with mu values

        Args:
        z (torch.Tensor): Latent vectors of shape (batch_size, latent_dim)
        mu (torch.Tensor): Mean values of shape (latent_dim,)
        std (torch.Tensor): Standard deviation values of shape (latent_dim,)
        batch_prob (float): Probability of applying the transformation to each element in the batch
        temperature (float): Temperature for Gumbel-Softmax, controls the discreteness of the output

        Returns:
        torch.Tensor: Transformed z
        """
        device = z.device
        batch_size, latent_dim = z.shape

        # 1. Batch-level probability
        batch_probs = torch.full((batch_size, 1), batch_prob, device=device)
        batch_mask = torch.bernoulli(batch_probs)  # Still using bernoulli, but it's not in the computational graph

        # 2. Feature-level probability inversely proportional to std
        feature_probs = 1 / (std + 1e-6)
        feature_probs = feature_probs / feature_probs.sum()

        # Generate differentiable feature mask using Gumbel-Softmax
        feature_logits = torch.log(feature_probs.unsqueeze(0).expand(batch_size, -1))
        feature_mask = F.gumbel_softmax(feature_logits, tau=temperature, hard=False)

        # Combine batch and feature masks
        combined_mask = batch_mask * feature_mask

        # Apply the transformation as a soft blending
        z_transformed = combined_mask * mu + (1 - combined_mask) * z

        return z_transformed

    def save(self, path, device="cpu"):
        """save the whole model and params in a single pytorch file"""
        prev_device = self.device
        self.to(device)
        torch.save(self, path)
        self.to(prev_device)

    @classmethod
    def load(cls, path, device=None):
        """load the whole model and params from a single pytorch file"""
        if device is not None:
            algo = torch.load(path, map_location=device)
            algo.to(device)
        else:
            algo = torch.load(path)

        return algo

    def input_shape(self):
        return self.encoder.input_shape
