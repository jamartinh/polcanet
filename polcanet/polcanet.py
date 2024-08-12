from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class PolcaNetLoss(nn.Module):
    """
    PolcaNetLoss is a class that extends PyTorch's Module class. It represents the loss function used for
    Principal Latent Components Analysis (Polca).

    Attributes:
        alpha (float): The weight for the orthogonality loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        class_labels (list): The list of class labels.
        a (torch.Tensor): The normalized axis.
        loss_names (dict): The dictionary of loss names.


    """

    def __init__(self, latent_dim, alpha=1.0, beta=1.0, gamma=1.0, class_labels=None):
        """
        Initialize PolcaNetLoss with the provided parameters.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_labels = class_labels

        pos = (torch.arange(latent_dim, dtype=torch.float32) ** 2) / latent_dim
        self.register_buffer('a', pos)  # normalized axis

        self.loss_names = {"loss": "Total Loss",
                           "rec": "Reconstruction Loss",
                           "ort": "Orthogonality Loss",
                           "com": "Center of Mass Loss",
                           "var": "Variance Reduction Loss",
                           }
        if class_labels is not None:
            self.loss_names["class"] = "Classification Loss"

    def forward(self, z, r, x, yp=None, target=None):
        """
        Variables:
        z: latent representation
        r: reconstruction
        x: input data
        n: number of latent dimensions
        Z: normalized latent representation (L2 norm)
        S: cosine similarity matrix
        v: variance of latent representation
        w: normalized variance (L1 norm)
        e: energy (squared z**2) z being unit vectors

        Losses:
        L1: reconstruction loss
        L2: orthogonality loss
        L3: center of mass loss
        L4: variance reduction loss
        L: total combined loss
        """

        device = x.device
        # Reconstruction loss
        # Purpose: Ensure the model can accurately reconstruct the input data
        # Method: Mean Squared Error between input and reconstruction
        L_rec = F.mse_loss(r, x)
        L_ort = self.orthogonality_loss(z)
        L_com, v, w = self.center_of_mass_loss(z)
        L_var = v.mean() if self.gamma != 0 else 0
        L_class = self.classification_loss(yp, target)

        # Combine losses
        # Purpose: Balance all loss components for optimal latent representation
        # Method: Weighted sum of individual losses
        L = L_rec + L_class + self.alpha * L_ort + self.beta * L_com + self.gamma * L_var

        if self.class_labels is not None:
            return L, (L_rec, L_ort, L_com, L_var, L_class)

        return L, (L_rec, L_ort, L_com, L_var)

    def center_of_mass_loss(self, z):
        """
        Center of mass loss
        Purpose: Concentrate information in earlier latent dimensions
        Method: Minimize the weighted average of normalized variances, e.g., the center of mass.
        """
        # Compute and normalize variance and energy
        device = z.device
        v = torch.var(z, dim=0)
        w = F.normalize(v, p=1.0, dim=0)
        e = torch.mean(z ** 2, dim=0)
        e = F.normalize(e, p=1.0, dim=0)

        if self.beta != 0:
            L_com = torch.mean((w + e) * self.a, dim=0)
        else:
            L_com = 0  # torch.tensor([0], dtype=torch.float32, device=device)
        return L_com, v, w

    def classification_loss(self, yp, target):
        # If class labels are provided, compute classification loss
        if self.class_labels is not None:
            L_class = F.cross_entropy(yp, target)
        else:
            # device = yp.device
            L_class = 0  # torch.tensor([0], dtype=torch.float32, device=device)
        return L_class

    def orthogonality_loss(self, z):
        """ Orthogonality loss
        Purpose: Encourage latent dimensions to be uncorrelated
        Method: Penalize off-diagonal elements of the cosine similarity matrix
        """
        if self.alpha == 0:
            return 0

        Z = F.normalize(z, p=2, dim=0)
        S = torch.mm(Z.t(), Z)
        idx0, idx1 = torch.triu_indices(S.shape[0], S.shape[1], offset=1)  # indices of triu w/o diagonal
        S_triu = S[idx0, idx1]
        loss = torch.mean(S_triu.square())
        return loss

    @staticmethod
    def random_projection(z, output_dim=None):
        batch_dim, n_features = z.shape
        if output_dim is None:
            output_dim = n_features

        # Step 1: Generate a random projection matrix
        random_matrix = torch.randn(n_features, output_dim, device=z.device)

        # Step 2: Project the data
        z_proj = torch.matmul(z, random_matrix)

        # # Step 3: Optionally normalize the rows to make them unit vectors
        # z_proj = F.normalize(z_proj, p=2, dim=1)

        return z_proj

    def orthogonality_loss_pythagoras(self, z):
        """
            Computes the orthogonality loss based on the Pythagorean property after normalizing
            the feature vectors to unit vectors and scaling by 1/n, using squared difference.

            Args:
                z (torch.Tensor): A tensor of shape (batch_dim, feature_dim) representing
                                  the feature vectors produced by the network for each sample
                                  in the batch.

            Returns:
                torch.Tensor: A scalar tensor representing the orthogonality loss.

            Theoretical Background:
            ------------------------
            For a set of vectors {z1, z2, ..., zn} in an n-dimensional space, when the vectors are normalized
            to unit vectors and scaled by 1/n (where n is the feature dimension), the vectors are orthogonal if
            and only if the following Pythagorean property holds:

                ||(1/n) * (z1 + z2 + ... + zn)||^2 = 1

            where n is the feature dimension.

            This scaling simplifies the analysis because the expected norm of the sum of these scaled vectors
            should be 1 when the vectors are orthogonal.

            Implementation:
            ----------------
            Given a batch of feature vectors z of shape (batch_dim, feature_dim):

            1. Normalize each feature vector to be a unit vector using F.normalize.
            2. Scale the normalized vectors by 1/n.
            3. Compute the sum of the scaled feature vectors for each sample in the batch: z_sum.
            4. Compute the squared norm of the sum vector: ||z_sum||^2.
            5. The loss is the squared difference between this squared norm and 1, averaged over the batch.

            Example Usage:
            --------------
            z = torch.randn(32, 128)  # Batch of 32 samples, each with 128-dimensional features
            loss = orthogonality_loss_with_scaling_squared(z)
            """
        if self.alpha == 0:
            return torch.tensor([0], dtype=torch.float32, device=z.device)

        z_unit = F.normalize(z, p=2, dim=1)  # Normalize each feature vector to be a unit vector
        batch_size, n_features = z.size()

        # Compute the squared magnitude of the sum
        sum_squared_magnitude = torch.sum(torch.sum(z_unit, dim=0) ** 2)

        # Compute the sum of squared magnitudes
        sum_of_squared_magnitudes = torch.sum(torch.sum(z_unit ** 2, dim=1))

        # The loss is the difference between these two values
        # If features are orthogonal, this difference should be zero
        loss = torch.abs(sum_squared_magnitude - sum_of_squared_magnitudes)

        loss = loss / (batch_size * n_features)  # Normalize by total number of elements

        return loss


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network encoder
    that is used for Principal Latent Components Analysis (Polca).

    Attributes:
        alpha (float): The weight for the cross-correlation loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        device (str): The device to run the encoder on ("cpu" or "cuda").
        encoder (nn.Module): The encoder module.
        decoder (nn.Module): The decoder module.
        latent_dim (int): The number of latent dimensions.
        class_labels (list): The list of class labels.


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


    Usage:
        data = np.random.rand(100, 784)
        latent_dim = 32
        model = PolcaNet(encoder, decoder, latent_dim=latent_dim, alpha=1.0, beta=1.0, gamma=1.0, class_labels=None)
        model.to_device("cuda")
        model.train_model(data, batch_size=256, num_epochs=10000, report_freq=20, lr=1e-3)
        model.r_error(data)
        latents, reconstructed = model.predict(data)


    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_dim: int, alpha=1, beta=1, gamma=1,
                 class_labels=None):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_labels = class_labels

        self.encoder = EncoderWrapper(encoder, latent_dim=latent_dim, class_labels=class_labels)
        self.decoder = decoder

        if hasattr(self.encoder, "decoder"):
            # remove the attribute decoder from the encoder
            self.encoder.decoder = None
            del self.encoder.decoder

        if hasattr(self.decoder, "encoder"):
            # remove the attribute encoder from the decoder
            self.decoder.encoder = None
            del self.decoder.encoder

        self.polca_loss = PolcaNetLoss(latent_dim, alpha, beta, gamma, class_labels=class_labels)

    def forward(self, x):
        z = self.encoder(x)
        if self.class_labels is not None:
            r = self.decoder(z[0])
            return (z[0], z[1]), r
        else:
            r = self.decoder(z)
        return z, r

    def predict(self, in_x, mask=None, n_components=None):
        z = self.encode(in_x)
        if n_components:
            z = z[:, :n_components]

        r = self.decode(z, mask=mask)
        return z, r

    def encode(self, in_x):
        self.encoder.eval()
        if isinstance(in_x, torch.Tensor):
            x = in_x.detach().clone().to(self.device)
        else:
            x = torch.tensor(in_x, dtype=torch.float32, device=self.device, requires_grad=False)

        with torch.no_grad():
            z = self.encoder(x)
            if self.class_labels is not None:
                return z[0].cpu().numpy()

        return z.cpu().numpy()

    def decode(self, z, mask=None):
        self.encoder.eval()

        # Convert z to tensor if it's not already
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32, device=self.device)

        # Ensure z has the correct shape
        if z.shape[1] < self.latent_dim:
            difference = self.latent_dim - z.shape[1]
            padding = torch.zeros(z.shape[0], difference, device=self.device)
            z = torch.cat([z, padding], dim=1)

        # Apply mask if provided
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
            if len(mask) != self.latent_dim:
                mask = torch.cat([mask, torch.zeros(self.latent_dim - len(mask), device=self.device)])
            mask = mask.unsqueeze(0).expand(z.shape[0], -1)
            z = z * mask  # This effectively sets masked dimensions to zero

        with torch.no_grad():
            r = self.decoder(z)

        return r.cpu().numpy()

    def r_error(self, in_x, w=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, w)
        error = (in_x - r).reshape(in_x.shape[0], -1)
        return np.mean(error ** 2, axis=-1)

    def to_device(self, device):
        """Move the encoder to the given device."""
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def train_model(self, data, y=None, batch_size=None, num_epochs=100, report_freq=10, lr=1e-3, verbose=1):
        # Input validation helpers
        # if class labels exist, assure that y is provided
        if self.class_labels is not None and y is None:
            raise ValueError("Class labels are provided, but no target values are provided in train_model")

        if isinstance(data, (np.ndarray, torch.Tensor)):
            if batch_size is None:
                raise ValueError("batch_size must be provided when data is in memory")
            fitter = self.fitter_in_memory(data, y=y, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
        elif isinstance(data, DataLoader):
            fitter = self.fitter_data_loader(data, num_epochs=num_epochs, lr=lr)
        else:
            fitter = None

        assert fitter is not None, "Data must be either a DataLoader or a numpy array or torch tensor"

        epoch_progress = tqdm(fitter, desc="epoch", leave=True, total=num_epochs, miniters=report_freq,
                              disable=bool(not verbose))
        metrics = {}
        epoch = 0
        for epoch, metrics in epoch_progress:
            if epoch % report_freq == 0:
                epoch_progress.set_postfix(metrics)

        # pretty print final metrics stats
        if bool(verbose):
            print(f"Final metrics at epoch: {epoch + 1}")
            for k, v in metrics.items():
                print(f"{self.polca_loss.loss_names[k]}: {v:.4g}")

        return metrics["loss"]

    def fitter_data_loader(self, data_loader, num_epochs=100, lr=1e-3):
        self.train()
        # Create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in data_loader:
                if self.class_labels is not None:
                    x = batch[0].to(self.device)
                    y = batch[1].to(self.device)
                    # assure that the target is an integer tensor of 32 bits
                    y = y.to(torch.int64)
                else:
                    # sometimes even if no targets in data loader the batch is a tuple
                    if isinstance(batch, tuple):
                        x = batch[0].to(self.device)
                    else:
                        x = batch.to(self.device)
                    y = None

                loss, aux_losses = self.learn_step(x, optimizer=optimizer, y=y)
                losses["loss"].append(loss.item())
                for idx, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                    losses[name].append(aux_losses[idx].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

    def fitter_in_memory(self, data: np.ndarray | torch.Tensor, y=None, batch_size=512, num_epochs=100, lr=1e-3):
        num_samples: int = len(data)
        self.train()
        torch_y = None

        if self.class_labels is None:
            y = None

        if not isinstance(data, torch.Tensor):
            torch_data = torch.tensor(data, dtype=torch.float32).to(self.device)
            if self.class_labels is not None:
                torch_y = torch.tensor(y, dtype=torch.int64).to(self.device)
        else:
            torch_data = data.to(self.device)
            if self.class_labels is not None:
                torch_y = y.to(self.device)

        # Create the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            indices = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                x = torch_data[batch_indices]
                if self.class_labels is not None:
                    targets = torch_y[batch_indices]
                else:
                    targets = None

                loss, aux_losses = self.learn_step(x, optimizer, y=targets)
                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(list(self.polca_loss.loss_names)[1:]):
                    losses[name].append(aux_losses[aux_loss].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

    def learn_step(self, x, optimizer, y=None):
        z, r = self.forward(x)
        yp = z[1] if self.class_labels is not None else None
        z = z[0] if self.class_labels is not None else z

        loss, aux_losses = self.polca_loss(z, r, x, yp=yp, target=y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, aux_losses


class EncoderWrapper(nn.Module):
    """
    Wrapper class for an encoder module.
    A Softsign activation function is applied to the output of the encoder and there is an optional factor scale
    parameter that can be set to True to factor the scale of the latent vectors and return the unit vectors and
    the magnitudes as a tuple.
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
        self.post_encoder = nn.Sequential(nn.Linear(latent_dim, latent_dim, bias=False), nn.Softsign())
        if class_labels is not None:
            # create a simple perceptron layer
            self.classifier = nn.Linear(latent_dim, len(class_labels))

    def forward(self, x):
        z = self.encoder(x)
        z = self.post_encoder(z)
        if self.class_labels is not None:
            return z, self.classifier(z)

        return z
