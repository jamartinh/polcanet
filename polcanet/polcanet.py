from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde
from tabulate import tabulate
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm


class FakeScaler:
    def __init__(self):
        pass

    @staticmethod
    def transform(x):
        return x

    @staticmethod
    def inverse_transform(x):
        return x

    @staticmethod
    def fit(data):
        pass


class PolcaNet(nn.Module):
    """
    PolcaNet is a class that extends PyTorch's Module class. It represents a neural network model
    that is used for Principal Latent Components Analysis (Polca).

    Attributes:
        latent_dim (int): The dimensionality of the latent space.
        hidden_dim (int): The dimensionality of the hidden layers.
        num_layers (int): The number of layers in the network.
        alpha (float): The weight for the cross-correlation loss.
        beta (float): The weight for the center of mass loss.
        gamma (float): The weight for the low variance loss.
        device (str): The device to run the model on ("cpu" or "cuda").
        model (examples.example_aencoders.BaseAutoEncoder): The base autoencoder model.
        bias (nn.Parameter): A learnable constant added to the latent space.
        reconstruction_loss_fn (nn.MSELoss): The reconstruction loss function.
        scaler (examples.example_aencoders.MinMaxScalerTorch): The scaler used to normalize the data.
        std_metrics (torch.Tensor): The standard deviation of the latent space.
        mean_metrics (torch.Tensor): The mean of the latent space.
        r_mean_metrics (torch.Tensor): The mean of the reconstruction error.
        aux_loss_names (list): The names of the auxiliary losses.

    Methods:
        forward(x): Encodes the input, adds the mean to the latent space, and decodes the result.
        compute_cross_correlation_loss(latent): Computes the cross-correlation loss.
        compute_center_of_mass_loss(latent): Computes the center of mass loss.
        low_variance_loss(latent): Minimizes the variance in the batch.
        predict(in_x, w): Encodes and decodes the input.
        encode(in_x): Encodes the input and adds the mean to the latent space.
        decode(z, w): Decodes the latent space.
        update_metrics(in_x): Updates the metrics based on the input.
        inv_score(x, idx): Scores the energy of a given x and y.
        r_error(in_x, w): Computes the reconstruction error.
        score(in_x, w): Computes the reconstruction score.
        to_device(device): Moves the model to the given device.
        train_model(data, batch_size, num_epochs, report_freq, lr): Trains the model.
        fitter(data, batch_size, num_epochs, lr): Fits the model to the data.
        learn_on_batch(x, optimizer): Performs a learning step on a batch.
        compute_losses(z, r, x): Computes the losses.
    """

    def __init__(self, model, alpha=0.1, beta=0.1, gamma=0.01, scaler=None, device="cpu"):
        """
        Initialize PolcaNet with the provided parameters.
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = torch.device(device)

        self.bias = None
        self.model = model

        self.reconstruction_loss_fn = nn.MSELoss()
        self.scaler = scaler or FakeScaler()
        self.std_metrics = None
        self.mean_metrics = None
        self.r_mean_metrics = None

        self.aux_loss_names = ["decode", "orth", "com", "var"]

    def forward(self, x, mask=None):
        latent = self.model.encode(x)
        if self.bias is None:
            self.bias = nn.Parameter(torch.zeros(latent.shape[1]))
            self.to(self.device)
        latent = latent + self.bias  # Add self.bias to latent
        reconstruction = self.model.decode(latent)
        return latent, reconstruction

    def predict(self, in_x, mask=None):
        z = self.encode(in_x)
        r = self.decode(z, mask=mask)
        return z, r

    def encode(self, in_x):
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(in_x, dtype=torch.float32, device=self.device)
            x = self.scaler.transform(x)
            latent = self.model.encode(x)
            if self.bias is None:
                self.bias = nn.Parameter(torch.zeros(latent.shape[1]))
                self.to(self.device)
            z = latent + self.bias

        return z.detach().cpu().numpy()

    def decode(self, z, mask=None):
        self.model.eval()
        if mask is not None:
            if len(mask) != self.bias.shape[0]:
                mask = np.array(mask)
                mask = np.concatenate([mask, np.zeros(self.latent_dim - len(mask))])

            z = np.where(mask == 0, self.bias.detach().cpu().numpy(), z)
        with torch.no_grad():
            z = torch.tensor(z, dtype=torch.float32, device=self.device)
            r = self.model.decode(z)
            r = self.scaler.inverse_transform(r)
            r = r.detach().cpu().numpy()
        return r

    def update_metrics(self, in_x):
        if isinstance(in_x, torch.Tensor):
            in_x = in_x.detach().cpu().numpy()
        z = self.encode(in_x)
        r = self.decode(z)
        self.r_mean_metrics = np.mean((in_x - r) ** 2)
        self.std_metrics = np.std(z, axis=0)
        self.mean_metrics = np.mean(z, axis=0)

    def r_error(self, in_x, w=None):
        """Compute the reconstruction error"""
        z, r = self.predict(in_x, w)
        return np.mean((in_x - r) ** 2, axis=1)

    def score(self, in_x, w=None):
        """Compute the reconstruction score"""
        return np.abs(self.r_error(in_x, w) - self.r_mean_metrics)

    def to_device(self, device):
        """Move the model to the given device."""
        self.device = device
        self.to(device)

    def to(self, device):
        self.device = device
        super().to(device)
        self.model.to(device)
        return self

    def train_model(self, data, batch_size=512, num_epochs=100, report_freq=10, lr=1e-3):
        fitter = self.fitter(data, batch_size=batch_size, num_epochs=num_epochs, lr=lr)
        epoch_progress = tqdm(fitter, desc="epoch", leave=True, total=num_epochs)
        metrics = {}
        epoch = 0
        for epoch, metrics in epoch_progress:
            if epoch % report_freq == 0:
                epoch_progress.set_postfix(metrics)

        # pretty print final metrics stats
        print(f"Final metrics at epoch: {epoch}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4g}")

    def fitter(self, data, batch_size=512, num_epochs=100, lr=1e-3):

        self.to(self.device)

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Create DataLoader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        with torch.no_grad():
            # Make some initializations
            _, _ = self.forward(data[0:2].to(self.device))

        # Create an optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(num_epochs):
            losses = defaultdict(list)
            for batch in dataloader:
                x = batch[0].to(self.device)
                loss, aux_losses = self.learn_on_batch(x, optimizer)
                losses["loss"].append(loss.item())
                for aux_loss, name in enumerate(self.aux_loss_names):
                    losses[name].append(aux_losses[aux_loss].item())

            metrics = {name: np.mean(losses[name]) for name in losses}
            yield epoch, metrics

        self.update_metrics(data)

    def learn_on_batch(self, x, optimizer):
        z, r = self.forward(x)
        loss, aux_losses = self.compute_losses(z, r, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, aux_losses

    @staticmethod
    def compute_cross_correlation_loss(latent):
        """Compute the cross correlation loss"""
        z = latent
        rnd = 1e-12 * torch.randn(size=z.shape, device=latent.device)
        x = z + rnd
        x_corr = torch.corrcoef(x.T)
        ll = x_corr.shape[0]
        idx = torch.triu_indices(ll, ll, offset=1)  # indices of triu w/o diagonal
        x_triu = x_corr[idx[0], idx[1]]
        corr_matrix_triu = x_triu
        corr_loss = torch.mean(corr_matrix_triu.pow(2))
        return corr_loss

    @staticmethod
    def average_pairwise_orthogonality(latent: torch.Tensor) -> torch.Tensor:
        """
        Compute the average pairwise orthogonality across all latent in a batch.

        Args:
        latent (torch.Tensor): Input tensor of shape (batch_dim, n_features)

        Returns:
        torch.Tensor: Average pairwise orthogonality measure for each batch
        """
        # Normalize the latent
        normalized_vectors = latent / torch.norm(latent, dim=1, keepdim=True)

        # Compute pairwise cosine similarities
        cosine_similarities = torch.matmul(normalized_vectors, normalized_vectors.transpose(0, 1))

        # Set diagonal to zero to exclude self-similarities
        mask = torch.eye(cosine_similarities.shape[0], device=latent.device).bool()
        cosine_similarities = cosine_similarities.masked_fill(mask, 0)

        # Compute orthogonality measure: |cos(theta)|
        orthogonality = torch.abs(cosine_similarities)

        # Compute average, excluding the diagonal zeros
        n = latent.shape[0]
        average_orthogonality = orthogonality.sum() / (n * (n - 1))

        return average_orthogonality

    @staticmethod
    def compute_center_of_mass_loss(latent):
        """Calculate center of mass loss"""
        axis = torch.arange(0, latent.shape[1], device=latent.device, dtype=torch.float32) ** 2
        std_latent = torch.var(latent, dim=0)

        w = nn.functional.normalize(std_latent, p=1.0, dim=0)
        com = w * axis / axis.shape[0]  # weight the latent space
        loss = torch.mean(com)
        return loss

    @staticmethod
    def low_variance_loss(latent, decay_rate=0.5):
        """Encourage exponential decay of variances"""
        var_latent = torch.var(latent, dim=0)

        # Create exponential decay target
        target_decay = torch.exp(-decay_rate * torch.arange(latent.shape[1], device=latent.device, dtype=torch.float32))

        # Normalize both to sum to 1 for fair comparison
        var_latent_norm = var_latent / torch.sum(var_latent)
        target_decay_norm = target_decay / torch.sum(target_decay)

        return torch.nn.functional.mse_loss(var_latent_norm, target_decay_norm)

    def compute_losses(self, z, r, x):
        # reconstruction loss
        l1 = self.reconstruction_loss_fn(r, x)
        # correlation loss
        # l2 = self.compute_cross_correlation_loss(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32,
        #                                                                                  device=x.device)
        l2 = self.average_pairwise_orthogonality(z) if self.alpha != 0 else torch.tensor([0], dtype=torch.float32,
                                                                                         device=x.device)
        # ordering loss
        l3 = self.compute_center_of_mass_loss(z) if self.beta != 0 else torch.tensor([0], dtype=torch.float32,
                                                                                     device=x.device)
        # low variance loss
        l4 = self.low_variance_loss(z) if self.gamma != 0 else torch.tensor([0], dtype=torch.float32, device=x.device)

        mean_loss = (torch.mean(z, dim=0) - self.bias).pow(2).sum()
        loss = l1 + self.alpha * l2 + self.beta * l3 + self.gamma * l4 + mean_loss

        return loss, (l1, l2, l3, l4)

    def analyze_latent_space(self, data=None, latents=None):
        """
        Perform a comprehensive text-based analysis of the latent space for a specialized autoencoder
        that concentrates variance in the first dimensions and aims to decorrelate features by orthogonalization
        of the latent space.

        Parameters:
        - model: The autoencoder model instance
        - data: Input data (numpy array), optional if latents are provided
        - latents: Latent representations (numpy array), optional if data is provided
        """
        if latents is None and data is None:
            raise ValueError("Either latents or data must be provided")

        if latents is None:
            latents = self.encode(data)

        n_components = latents.shape[1]

        # Compute the correlation matrix and variances
        corr = np.corrcoef(latents.T)
        variances = np.var(latents, axis=0)

        # Off-diagonal correlations
        off_diag_corr = corr[np.triu_indices(n_components, k=1)]

        # Explained variance ratio
        explained_variance_ratio = variances / np.sum(variances)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Compute specialized metrics
        variance_concentration = np.sum(variances * np.arange(n_components, 0, -1)) / (np.sum(variances) * n_components)
        decorrelation_metric = 1 - np.mean(np.abs(off_diag_corr))

        # Print report
        print("\n" + "=" * 50)
        print("Latent Space Analysis Report".center(50))
        print("=" * 50 + "\n")

        print("1. General Information")
        print("-" * 30)
        print(f"Number of latent components: {n_components}")
        print(f"Total variance in latent space: {np.sum(variances):.4f}")
        print()

        print("2. Variance Analysis")
        print("-" * 30)
        variance_table = [["First component", f"{explained_variance_ratio[0]:.4f}"],
                          ["First 5 components", f"{np.sum(explained_variance_ratio[:5]):.4f}"],
                          ["Components for 95% variance", f"{np.argmax(cumulative_variance_ratio >= 0.95) + 1}"],
                          ["Variance Concentration Metric", f"{variance_concentration:.4f}"]]
        print(tabulate(variance_table, headers=["Metric", "Value"]))
        print("\nVariance Concentration Interpretation:")
        if variance_concentration > 0.8:
            print("Excellent concentration of variance in earlier dimensions.")
        elif variance_concentration > 0.6:
            print("Good concentration of variance, but there might be room for improvement.")
        else:
            print("Poor concentration of variance. The model may need adjustment.")
        print()

        print("3. Orthogonality Analysis")
        print("-" * 30)
        corr_table = [["Mean absolute off-diagonal", f"{np.mean(np.abs(off_diag_corr)):.4f}"],
                      ["Median absolute off-diagonal", f"{np.median(np.abs(off_diag_corr)):.4f}"],
                      ["Max absolute off-diagonal", f"{np.max(np.abs(off_diag_corr)):.4f}"],
                      ["Proportion of |Orthogonality| > 0.1", f"{np.mean(np.abs(off_diag_corr) > 0.1):.4f}"],
                      ["Orthogonality Success Metric", f"{decorrelation_metric:.4f}"]]
        print(tabulate(corr_table, headers=["Metric", "Value"]))
        print("\nOrthogonality Interpretation:")
        if decorrelation_metric > 0.9:
            print("Excellent orthogonality of features.")
        elif decorrelation_metric > 0.7:
            print("Good orthogonality, but there might be room for improvement.")
        else:
            print("Poor orthogonality. The model may need adjustment.")
        print()

        print("4. Detailed Component Analysis")
        print("-" * 30)
        top_n = min(10, n_components)  # Analyze top 10 components or all if less than 10
        component_table = []
        for i in range(top_n):
            component_table.append([i + 1, f"{explained_variance_ratio[i]:.4f}", f"{cumulative_variance_ratio[i]:.4f}",
                                    f"{np.mean(np.abs(corr[i, i + 1:])):.4f}"])
        print(tabulate(component_table, headers=["Component", "Variance Ratio", "Cumulative Variance",
                                                 "Mean |Correlation| with Others"]))
        print()

    def show_correlation_matrix(self, latents=None, data=None):
        if latents is None and data is None:
            raise ValueError("Either latents or data must be provided")

        if latents is None:
            latents, reconstructed = self.predict(data)

        # Compute the correlation matrix
        x_corr = np.corrcoef(latents.T)
        corr = pd.DataFrame(x_corr)

        # Set up the matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create the heatmap
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

        # Add colorbar for reference
        fig.colorbar(cax)

        # Add title
        plt.title("Correlation Matrix of Latent Components", fontsize=16)

        # Show the plot
        plt.show()

    def plot_scatter_corr_matrix(self, latents=None, data=None, n_components=5, max_samples=5000):

        if latents is None and data is None:
            raise ValueError("Either latents or data must be provided")

        if latents is None:
            latents, reconstructed = self.predict(data)

        df = pd.DataFrame(latents)

        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        num_vars = min(n_components, latents.shape[1])
        fig, axes = plt.subplots(num_vars, num_vars, figsize=((num_vars * 1), (num_vars * 1)))

        # Loop over the indices to create scatter plots for the lower triangle
        for i in range(1, num_vars):
            for j in range(i):
                x = df.iloc[:, j]
                y = df.iloc[:, i]

                axes[i - 1, j].scatter(x, y, alpha=0.5, s=1)
                if i == num_vars - 1:
                    axes[i - 1, j].set_xlabel(f'$x_{j}$')
                if j == 0:
                    axes[i - 1, j].set_ylabel(f'$x_{i}$')

                    # Remove axis ticks
                axes[i - 1, j].set_xticks([])
                axes[i - 1, j].set_yticks([])
                axes[i - 1, j].spines['top'].set_visible(False)
                axes[i - 1, j].spines['right'].set_visible(False)

        # Hide the upper triangle and diagonal subplots
        for i in range(num_vars):
            for j in range(num_vars):
                if i <= j:
                    axes[i - 1, j].axis('off')

        plt.suptitle("Scatter plot matrix of latent components")
        plt.tight_layout()
        plt.show()

    def plot_stdev_pct(self):
        x_plot = np.arange(1, self.std_metrics.shape[0] + 1)
        y_plot = 100 * (self.std_metrics / np.sum(self.std_metrics))
        str_texts = [f"{round(t, 2):.1f}%" for t in y_plot.tolist()]

        plt.plot(x_plot, y_plot, "o-")
        for _x, _y, _s in zip(x_plot, y_plot, str_texts):
            plt.text(_x, _y, _s)
        plt.title("Stdev percentage")
        plt.show()

    def plot_cumsum_variance(self, data):

        latents, reconstructed = self.predict(data)
        inputs = data
        arr_x = np.zeros((latents.shape[1], latents.shape[1]))
        idx = np.tril_indices(latents.shape[1])
        arr_x[idx[0], idx[1]] = 1

        total_var = np.var(inputs)

        errors = []
        variances = []
        for i in range(latents.shape[1]):
            w = arr_x[i, :]
            latents = self.encode(inputs)
            reconstructed = self.decode(latents, w)
            error = np.mean(np.mean((inputs - reconstructed) ** 2, axis=1))
            errors.append(error)
            total_var_approx = np.var(reconstructed)
            cumulative_percent_variance = np.clip(((total_var_approx / total_var) * 100.0), 0, 100)
            variances.append(cumulative_percent_variance)
            print(f"({i + 1}) reconstruction error: {error:.4f}, "
                  f"variance: {cumulative_percent_variance:.1f}%, "
                  f"with {i + 1:6d}  active latent components")

        errors = np.array(errors)
        norm_errors = 100 * (errors / (np.sum(errors)))
        cumulative_percent_variances = np.array(variances)

        plt.plot(norm_errors, "o-", label="reconstruction mse")
        plt.title("Percentage error reduction by adding more components")

        plt.plot(cumulative_percent_variances, "o-", label="explained variance")
        plt.title("Cummulative ptc variances")
        plt.legend()
        plt.show()

    def analyze_latent_feature_importance(self, data):
        """
        Compute and visualize latent feature importance based on variance.

        Parameters:
        - model: The autoencoder model instance
        - data: Input data (numpy array)

        Returns:
        - feature_importance: Array of importance scores (variances) for each latent feature
        """
        # Encode the data to get latent representations
        latents = self.encode(data)

        # Compute feature importance as variance
        feature_importance = np.var(latents, axis=0)

        # Normalize the importance scores
        feature_importance_normalized = feature_importance / np.sum(feature_importance)

        # Plot results
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(feature_importance_normalized)), feature_importance_normalized)
        plt.xlabel('Latent Feature Index')
        plt.ylabel('Normalized Variance')
        plt.title('Latent Feature Importance (Based on Variance)')
        plt.xticks(range(0, len(feature_importance_normalized), max(1, len(feature_importance_normalized) // 10)))
        plt.grid(True)

        # Add a trend line
        z = np.polyfit(range(len(feature_importance_normalized)), feature_importance_normalized, 1)
        p = np.poly1d(z)
        plt.plot(range(len(feature_importance_normalized)), p(range(len(feature_importance_normalized))), "r--",
                 alpha=0.8)

        plt.show()

        # Plot cumulative importance
        cumulative_importance = np.cumsum(feature_importance_normalized)
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'bo-')
        plt.xlabel('Number of Latent Features')
        plt.ylabel('Cumulative Normalized Variance')
        plt.title('Cumulative Latent Feature Importance')
        plt.grid(True)
        plt.show()

        # Print some statistics
        print(f"First feature importance: {feature_importance_normalized[0]:.4f}")
        print(f"Last feature importance: {feature_importance_normalized[-1]:.4f}")
        print(
            f"Ratio of first to last feature importance: {feature_importance_normalized[0] / feature_importance_normalized[-1]:.4f}")
        print(f"Number of features explaining 80% of the variance: {np.argmax(cumulative_importance >= 0.8) + 1}")

        # Compute and print the Intrinsic Dimension
        intrinsic_dim = 2 * np.sum(feature_importance) ** 2 / np.sum(feature_importance ** 2)
        print(f"Intrinsic Dimension: {intrinsic_dim:.2f}")

    def analyze_reconstruction_error(self, data, n_samples=10000):
        """
        Analyze reconstruction errors, handling both low and high-dimensional data.

        Parameters:
        - model: The autoencoder model instance
        - data: Input data (numpy array)
        - n_samples: Number of samples to use for error distribution estimation (if data is larger)
        """
        # Encode and decode the data
        latents = self.encode(data)
        reconstructed = self.decode(latents)

        # Compute reconstruction errors (mean squared error per sample)
        errors = np.mean((data - reconstructed) ** 2, axis=tuple(range(1, data.ndim)))

        # Down sample if necessary
        if errors.shape[0] > n_samples:
            indices = np.random.choice(errors.shape[0], n_samples, replace=False)
            errors = errors[indices]

        # Plot histogram of errors
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, density=True, alpha=0.7)
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Density')
        plt.title('Distribution of Reconstruction Errors')

        # Try to plot KDE if the number of samples allows
        if errors.shape[0] > 1:
            try:
                kde = gaussian_kde(errors)
                x_range = np.linspace(errors.min(), errors.max(), 1000)
                plt.plot(x_range, kde(x_range), 'r-', lw=2)
            except np.linalg.LinAlgError:
                print("Warning: Couldn't compute KDE due to singular matrix. Showing histogram only.")

        # Add vertical line for mean error
        mean_error = np.mean(errors)
        plt.axvline(mean_error, color='g', linestyle='--', label=f'Mean Error: {mean_error:.4f}')

        # Add vertical line for median error
        median_error = np.median(errors)
        plt.axvline(median_error, color='r', linestyle=':', label=f'Median Error: {median_error:.4f}')

        plt.legend()
        plt.grid(True)
        plt.show()

        # Print some statistics
        print(f"Mean Reconstruction Error: {mean_error:.4f}")
        print(f"Median Reconstruction Error: {median_error:.4f}")
        print(f"Min Reconstruction Error: {np.min(errors):.4f}")
        print(f"Max Reconstruction Error: {np.max(errors):.4f}")
