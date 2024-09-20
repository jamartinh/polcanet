from itertools import combinations
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics.pairwise import cosine_similarity

from examples.notebooks.train import in_jupyterlab
from polcanet import utils as ut
from polcanet.utils import save_figure, save_df_to_csv


def analyze_reconstruction_error(model, data, n_samples=10000, save_fig: str = None):
    """
    Analyze reconstruction errors, handling both low and high-dimensional data.

    Parameters:
    - encoder: The autoencoder encoder instance
    - data: Input data (numpy array)
    - n_samples: Number of samples to use for error distribution estimation (if data is larger)
    """
    # Encode and decode the data
    latents = model.encode(data)
    reconstructed = model.decode(latents)

    # Compute reconstruction errors (mean squared error per sample)
    errors = np.mean((data - reconstructed) ** 2, axis=tuple(range(1, data.ndim)))

    # Down sample if necessary
    if errors.shape[0] > n_samples:
        indices = np.random.choice(errors.shape[0], n_samples, replace=False)
        errors = errors[indices]

    # Plot histogram of errors
    plt.figure()
    plt.hist(errors, bins=50, density=True, alpha=0.7, fill=False, color="black")
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')

    # Try to plot KDE if the number of samples allows
    if errors.shape[0] > 1:
        try:
            kde = gaussian_kde(errors)
            x_range = np.linspace(errors.min(), errors.max(), 1000)
            plt.plot(x_range, kde(x_range), 'k-', lw=1)
        except np.linalg.LinAlgError:
            print("Warning: Couldn't compute KDE due to singular matrix. Showing histogram only.")

    # Add vertical line for mean error
    mean_error = np.mean(errors)
    plt.axvline(mean_error, color='gray', linestyle='--', label=f'Mean Error: {mean_error:.4f}', lw=1)

    # Add vertical line for median error
    median_error = np.median(errors)
    plt.axvline(median_error, color='r', linestyle=':', label=f'Median Error: {median_error:.4f}', lw=1)

    plt.legend()
    plt.tight_layout()
    fig_name = save_fig or "reconstruction_error_distribution.pdf"
    save_figure(fig_name)
    plt.show()

    # Print some statistics
    print(f"Mean Reconstruction Error: {mean_error:.4f}")
    print(f"Median Reconstruction Error: {median_error:.4f}")
    print(f"Min Reconstruction Error: {np.min(errors):.4f}")
    print(f"Max Reconstruction Error: {np.max(errors):.4f}")


def plot_scatter_corr_matrix(model=None, latents=None, data=None, n_components=5, max_samples=1000,
                             save_fig: str = None):
    if latents is None and (data is None or model is None):
        raise ValueError("Either latents or model= and data= must be provided")

    if latents is None:
        latents, reconstructed = model.predict(data)

    if latents.shape[0] > max_samples:
        indices = np.random.choice(latents.shape[0], max_samples, replace=False)
        latents = latents[indices]

    cos_sim = cosine_similarity(latents.T)
    num_vars = min(n_components, latents.shape[1])
    plot_corr_scatter(cos_sim, latents, num_vars, save_fig=save_fig)


# Improved function to plot only the lower triangular part of the MI matrix
def plot_lower_triangular_mutual_information(mi_matrix, title='Pairwise Mutual Information', save_fig: str = None):
    threshold = 10

    # Mask for the upper triangle (to hide it) but keep the diagonal
    mask = np.triu(np.ones_like(mi_matrix, dtype=bool))

    # Setup the matplotlib figure
    fig, ax = plt.subplots()

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    # Get Matplotlib Grays color map
    cmap = plt.get_cmap('binary')

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(mi_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0.5, square=True, linewidths=.5,
                cbar_kws={"shrink": .5}, annot=(mi_matrix.shape[0] <= threshold), fmt='.2f', annot_kws={"size": 10})

    # Add title with improved formatting
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    # Tight layout for better spacing
    plt.tight_layout()
    fig_name = save_fig or "mutual_information_matrix.pdf"
    save_figure(fig_name)

    plt.show()


def calculate_mutual_information(latent_x):
    n_features = latent_x.shape[1]
    # Initialize an empty matrix to store MI values
    mi_matrix = np.zeros((n_features, n_features))

    def calculate_mi(x, i, j):
        _mi = mutual_info_regression(np.expand_dims(x[:, i], axis=1), x[:, j])[0]
        return i, j, _mi

    # Calculate pairwise MI in parallel
    results = Parallel(n_jobs=20)(delayed(calculate_mi)(latent_x, i, j) for i, j in combinations(range(n_features), 2))
    for i, j, mi in results:
        mi_matrix[i, j] = mi
        mi_matrix[j, i] = mi
    return mi_matrix


def plot_corr_scatter(corr_matrix, latents, n, save_fig: str = None):
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n - 1, n)

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    # Create the colormap
    sigma = 0.25  # Adjust this parameter to control the width of the black peak
    colors = plt.cm.binary(np.exp(-((np.linspace(-1, 1, 256)) ** 2) / (2 * sigma ** 2)))
    cmap = LinearSegmentedColormap.from_list("gaussian_black", colors)

    # Loop over the indices to create scatter plots for the lower triangle
    for i in range(1, n):
        for j in range(i):
            color = cmap((corr_matrix[i, j] + 1) / 2)
            # make color an array of len x
            color = np.array([color for _ in range(latents.shape[0])])
            x = latents[:, j]
            y = latents[:, i]
            axes[i - 1, j].scatter(x, y, s=1.0, c=color, rasterized=True)  # Make markers small
            # axes[i - 1, j].set_facecolor(color)
            axes[i - 1, j].axis('square')

            # Annotate correlation value if the number of variables is small
            if n <= 5:
                corr_text = f"{corr_matrix[i, j]:.2f}"
                legend = axes[i - 1, j].legend([corr_text], loc='best')
                frame = legend.get_frame()
                frame.set_facecolor('white')
                frame.set_edgecolor('black')
                frame.set_alpha(0.2)
                # Remove tick labels but show tick marks
                axes[i - 1, j].set_xticklabels([])
                axes[i - 1, j].set_yticklabels([])
            else:
                axes[i - 1, j].set_xticks([])
                axes[i - 1, j].set_yticks([])

            # Label the last row and first column
            if i == n - 1:
                axes[i - 1, j].set_xlabel(f'$x_{j}$')
            if j == 0:
                axes[i - 1, j].set_ylabel(f'$x_{i}$')

            # Set the legend for the Axes object

    # Hide the upper triangle and diagonal subplots
    for i in range(n):
        for j in range(n):
            if i < j:
                axes[i, j].axis('off')

    # Add color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04, shrink=.5)
    cbar.outline.set_edgecolor('none')  # Remove color bar border
    fig_name = save_fig or "scatter_correlation_matrix.pdf"
    save_figure(fig_name)
    # Show plot
    plt.show()


def plot_correlation_matrix(corr_matrix, threshold=10, save_fig: str = None):
    """
    Plots a correlation matrix. If the matrix size is below the threshold,
    it includes the correlation values in the cells; otherwise, it does not.

    Parameters:
    corr_matrix (pd.DataFrame): Correlation matrix to plot.
    threshold (int): Size threshold to decide if cell values should be printed.
    """
    # Check if the input is a DataFrame
    if not isinstance(corr_matrix, pd.DataFrame):
        raise ValueError("The input must be a pandas DataFrame.")

    # Mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Setup the matplotlib figure
    fig, ax = plt.subplots()

    # Generate a custom diverging colormap
    # cmap = sns.diverging_palette(220, 10, as_cmap=True)
    cmap = plt.get_cmap('binary')

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, vmin=0, center=0.5, square=True, linewidths=.5,
                cbar_kws={"shrink": .5}, annot=(corr_matrix.shape[0] <= threshold), fmt='.2f', annot_kws={"size": 10})

    # Add titles and labels
    ax.set_title('Cosine Similarity Matrix')
    plt.xticks(rotation=45, ha='right')

    # Tight layout for better spacing
    plt.tight_layout()
    fig_name = save_fig or "correlation_matrix.pdf"
    save_figure(fig_name)
    # Show plot
    plt.show()


def orthogonality_test_analysis(model, pca, data, num_samples=1000, n_components=10, save_figs: Tuple[str] = None):
    """
    Analyze the orthogonality of the latent features of the autoencoder.

    Parameters:
    - encoder: The autoencoder encoder instance
    - data: Input data (numpy array)
    - num_samples: Number of samples to test (default: 1000)
    - n_components: Number of components to plot in the scatter correlation matrix
    """
    # Select random samples from the data
    num_samples = min(num_samples, data.shape[0])
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    x_samples = data[indices]

    # Encode the samples
    latent_x = model.encode(x_samples)
    latent_pca = pca.transform(np.squeeze(x_samples.reshape(x_samples.shape[0], -1)))

    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(latent_x.T)

    # Extract the upper triangular part of the similarity matrix, excluding the diagonal
    upper_triangular_indices = np.triu_indices_from(cosine_sim, k=1)
    upper_triangular_values = np.abs(cosine_sim[upper_triangular_indices])

    # Reporting text with statistics
    # Create a report based on a DataFrame
    df_report = pd.DataFrame(columns=["Metric", "Value"])
    df_report.loc[len(df_report)] = ["Mean cosine similarity", np.mean(upper_triangular_values)]
    df_report.loc[len(df_report)] = ["Median cosine similarity", np.median(upper_triangular_values)]
    df_report.loc[len(df_report)] = ["Standard deviation of cosine similarity", np.std(upper_triangular_values)]
    df_report.loc[len(df_report)] = ["Max cosine similarity", np.max(upper_triangular_values)]
    df_report.loc[len(df_report)] = ["Min cosine similarity", np.min(upper_triangular_values)]
    display(df_report)
    # Save the report to a CSV file
    save_df_to_csv(df_report, "orthogonality_report.csv")

    # Plot cosine similarity matrix
    save_fig = save_figs[0] if save_figs else None
    plot_correlation_matrix(pd.DataFrame(cosine_sim).abs(), save_fig=save_fig)

    # plot mutual information matrix for model
    save_fig = save_figs[2] if save_figs else None
    mi_matrix = calculate_mutual_information(latent_x)

    # get the upper triangular part of the matrix
    upper_triangular = np.triu(mi_matrix, k=1)
    # create a report based on a DataFrame
    df_report = pd.DataFrame(columns=["Metric", "Value"])
    df_report.loc[len(df_report)] = ["Mean mutual information", np.mean(upper_triangular)]
    df_report.loc[len(df_report)] = ["Median mutual information", np.median(upper_triangular)]
    df_report.loc[len(df_report)] = ["Standard deviation of mutual information", np.std(upper_triangular)]
    df_report.loc[len(df_report)] = ["Max mutual information", np.max(upper_triangular)]
    df_report.loc[len(df_report)] = ["Min mutual information", np.min(upper_triangular)]
    display(df_report)
    # Save the report to a CSV file
    save_df_to_csv(df_report, "mutual_information_report_polca.csv")
    plot_lower_triangular_mutual_information(mi_matrix, title='Pairwise Mutual Information POLCA',
                                             save_fig="mutual_information_matrix_polca.pdf")

    # plot mutual information matrix for PCA
    save_fig = save_figs[3] if save_figs else None
    mi_matrix_pca = calculate_mutual_information(latent_pca)

    # get the upper triangular part of the matrix
    upper_triangular_pca = np.triu(mi_matrix_pca, k=1)
    # create a report based on a DataFrame
    df_report = pd.DataFrame(columns=["Metric", "Value"])
    df_report.loc[len(df_report)] = ["Mean mutual information", np.mean(upper_triangular_pca)]
    df_report.loc[len(df_report)] = ["Median mutual information", np.median(upper_triangular_pca)]
    df_report.loc[len(df_report)] = ["Standard deviation of mutual information", np.std(upper_triangular_pca)]
    df_report.loc[len(df_report)] = ["Max mutual information", np.max(upper_triangular_pca)]
    df_report.loc[len(df_report)] = ["Min mutual information", np.min(upper_triangular_pca)]
    display(df_report)
    # Save the report to a CSV file
    save_df_to_csv(df_report, "mutual_information_report_pca.csv")
    plot_lower_triangular_mutual_information(mi_matrix_pca, title='Pairwise Mutual Information PCA',
                                             save_fig="mutual_information_matrix_pca.pdf")

    # Plot scatter correlation matrix
    save_fig = save_figs[1] if save_figs else None
    plot_scatter_corr_matrix(model, latents=latent_x, n_components=n_components, save_fig=save_fig)


def variance_test_analysis(model, data, num_samples=1000, save_figs: Tuple[str] = None):
    """
    Analyze the variance concentration of the latent features of the autoencoder.

    Parameters:
    - encoder: The autoencoder encoder instance
    - data: Input data (numpy array)
    - num_samples: Number of samples to test (default: 1000)
    """
    num_samples = min(num_samples, data.shape[0])

    # Select random samples from the data
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    x_samples = data[indices]

    # Encode the samples
    latent_x = model.encode(x_samples)

    # Calculate variances of the latent features
    variances = np.var(latent_x, axis=0)

    # Calculate center of mass
    components = np.arange(1, len(variances) + 1)

    # Calculate exponential fit for variance
    normalized_variances = variances / np.sum(variances)
    exp_fit = np.exp(-components)
    exp_fit /= np.sum(exp_fit)

    # Plot variance distribution and exponential fit
    fig, ax = plt.subplots()
    ax.plot(components, normalized_variances, 'o-', label='Normalized Variances', color='black', alpha=0.7, linewidth=1,
            markersize=2)
    ax.plot(components, exp_fit, 'r--', label='Exponential ref')
    ax.set_title('Variance Distribution of latent space')
    ax.set_xlabel('Components')
    # ax.set_xticks(components, labels=[f"$x_{i}$" for i in components])
    ax.set_ylabel('Normalized Variance')
    # ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig_name = "variance_distribution.pdf"
    save_figure(fig_name)
    plt.show()

    # Plot cumulative variance
    # Compute cumulative explained variance ratio
    cumulative_variance = np.cumsum(normalized_variances)
    components_90 = np.argmax(cumulative_variance >= 0.9) + 1
    components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    fig, ax = plt.subplots()
    ax.plot(components, cumulative_variance, 'o-', label='Cumulative Variance', color='black', alpha=0.7, linewidth=1,
            markersize=2)
    ax.set_title('Cumulative Variance of latent space')
    ax.set_xlabel('Components')
    ax.set_ylabel('Cumulative Variance')
    # ax.set_xticks(components, labels=[f"$x_{i}$" for i in components])
    # ax.grid(True)

    # Add some reference lines
    total_components = len(variances)
    ax.axhline(y=0.9, color='r', linestyle='--', lw=1,
               label=f"90% by {(100 * components_90 / total_components):.1f}% ({components_90}) components")
    ax.axhline(y=0.95, color='g', linestyle='--', lw=1,
               label=f"95% by {(100 * components_95 / total_components):.1f}% ({components_95}) components")
    ax.legend()
    plt.tight_layout()
    fig_name = "cumulative_variance.pdf"
    save_figure(fig_name)
    plt.show()

    # Analyze and plot variance concentration
    top_n = min(20, len(variances))  # Analyze top 20 components or all if less than 10

    # Plot variance concentration in top components
    fig, ax = plt.subplots()
    # get a twin axis
    ax2 = ax.twinx()

    max_value = np.max(normalized_variances[:top_n])
    ax.set_xlim(0, max_value + 0.15)
    bar = ax.barh(range(1, top_n + 1), normalized_variances[:top_n], fill=False, edgecolor='k', linewidth=1,
                  align="center")
    ax.bar_label(bar, fmt='%.2f', fontsize=8, padding=5)
    ax.set_title('Variance Concentration by component')
    ax.set_ylabel('Components')
    ax.set_xlabel('Normalized Variance')
    # increase the limit of x-axis
    ax.set_yticks(range(1, top_n + 1))
    ax2.set_yticks(range(1, top_n + 1), labels=[f'{v2:.2f}' for v2 in cumulative_variance[:top_n]], fontsize=8)
    ax2.set_ylim(ax.get_ylim())
    ax.invert_yaxis()  # labels read top-to-bottom
    ax2.invert_yaxis()  # labels read top-to-bottom

    plt.tight_layout()
    fig_name = "variance_concentration.pdf"
    save_figure(fig_name)
    plt.show()


def linearity_test_plot(model, x, y, alpha_min=-1, alpha_max=1, save_fig: str = None):
    model.eval()
    shape = x.shape
    num_samples = shape[0]
    x = x.view(shape[0], -1)
    y = y.view(shape[0], -1)

    def f(z):
        return model(z).detach().view(shape[0], -1)

    with torch.no_grad():
        # Test Additivity
        xx = f(x + y)
        fx_shape = xx.shape
        yy = f(x) + f(y)
        xx = xx.cpu().numpy().flatten()
        yy = yy.cpu().numpy().flatten()
        additivity_corr = np.corrcoef(xx, yy)[0, 1]

        # Test Homogeneity
        alpha = torch.linspace(alpha_min, alpha_max, steps=shape[0], device=x.device).unsqueeze(1)
        xx_alpha = f(alpha * x)  # Unsqueeze alpha to match x's dimensions
        alpha = alpha.repeat(1, fx_shape[1])
        yy_alpha = alpha * f(x)
        xx_alpha = xx_alpha.cpu().numpy().flatten()
        yy_alpha = yy_alpha.cpu().numpy().flatten()
        alpha_colors = alpha.cpu().numpy().flatten()
        homogeneity_corr = np.corrcoef(xx_alpha, yy_alpha)[0, 1]

        # # Test Lipschitz condition  # # We calculate |f(x) - f(y)| and |x - y|
        # alpha = torch.linspace(alpha_min, alpha_max, steps=shape[0], device=x.device).unsqueeze(1)
        # f_x = f(x)
        # alpha_x = alpha.repeat(1, x.shape[1])
        #
        # f_y = f(x + alpha_x)
        # diff_fxy = torch.linalg.vector_norm(f_x - f_y, dim=1)
        # diff_xy = alpha.squeeze()  # diff_fxy = diff_fxy.cpu().numpy()
        # diff_xy = diff_xy.cpu().numpy()
        # lipschitz_corr = np.corrcoef(diff_fxy, diff_xy)[0, 1]

    # take a random sample of size at most 1000 of xx and yy since this is a heavy scatter plot
    indices = np.random.choice(xx.shape[0], min(5 * num_samples, xx.shape[0]), replace=False)
    xx = xx[indices]
    yy = yy[indices]
    add_colors = alpha_colors[indices]
    xx_alpha = xx_alpha[indices]
    yy_alpha = yy_alpha[indices]
    alpha_colors = alpha_colors[indices]

    # Plotting results
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Additivity plot
    ax1.scatter(xx, yy, c=add_colors, cmap='coolwarm', alpha=0.7, s=1, rasterized=True)
    min_val = min(xx.min(), yy.min())
    max_val = max(xx.max(), yy.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    ax1.set_xlabel(r"$f(x + y)$")
    ax1.set_ylabel(r"$f(x) + f(y)$")
    ax1.set_title(f"Additivity:\n{additivity_corr:.2f}")
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle=':', linewidth=0.5)
    ax1.set_aspect("equal")
    ax1.axis("square")

    # Homogeneity plot
    scatter = ax2.scatter(xx_alpha, yy_alpha, c=alpha_colors, cmap='coolwarm', alpha=0.7, s=1, rasterized=True)
    min_val = min(xx_alpha.min(), yy_alpha.min())
    max_val = max(xx_alpha.max(), yy_alpha.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    ax2.set_xlabel(r"$f(\alpha x)$")
    ax2.set_ylabel(r"$\alpha f(x)$")
    ax2.set_title(f"Homogeneity:\n{homogeneity_corr:.2f}")
    ax2.set_aspect("equal")
    ax2.axis("square")

    fig.colorbar(scatter, ax=ax2, fraction=0.04, label=r"$\alpha$ values")
    ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle=':', linewidth=0.5)
    plt.tight_layout()

    fig_name = save_fig or "linearity_properties.pdf"
    save_figure(fig_name)
    plt.show()

    # fig2, ax = plt.subplots(figsize=(5,5))
    # # # Lipschitz condition plot
    # #ax.scatter(diff_xy, diff_fxy, c='k', alpha=0.7, s=2, rasterized=True)
    # ax.plot(diff_xy, diff_fxy, c='k', alpha=0.7, lw=1)
    # ax.set_xlabel(r"$\epsilon$")
    # ax.set_ylabel(r"$||f(x) - f(x+\epsilon)||_2$")
    # ax.set_title(f"Lipschitz:\n{lipschitz_corr:.2f}")
    # plt.tight_layout()
    #
    # fig_name = "Lipschitz_condition.pdf"
    # save_figure(fig_name)
    # plt.show()

    return additivity_corr, homogeneity_corr, None


def get_data_samples(model, data, num_samples):
    """
    Get random samples from the data and encode them using the autoencoder.
    """
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    x_samples = data[indices]
    y_samples = data[np.random.choice(data.shape[0], num_samples, replace=False)]
    # Encode the samples
    latent_x = model.encode(x_samples)
    latent_y = model.encode(y_samples)
    return latent_x, latent_y


def linearity_tests_analysis(model, data, alpha_min=-1, alpha_max=1, num_samples=1000, save_fig: str = None):
    """
        Analyze the linearity properties of the autoencoder's decoder.

        Parameters:
        - encoder: The autoencoder encoder instance
        - data: Input data (numpy array)
        - num_samples: Number of samples to test (default: 100)
        """
    num_samples = min(num_samples, data.shape[0])
    latent_x, latent_y = get_data_samples(model, data, num_samples)
    latent_x = torch.tensor(latent_x, dtype=torch.float32, device=model.device)
    latent_y = torch.tensor(latent_y, dtype=torch.float32, device=model.device)
    linearity_test_plot(model.decoder, latent_x, latent_y, alpha_min=alpha_min, alpha_max=alpha_max, save_fig=save_fig)


def embedding_analysis(model, pca, data, targets, labels_dict, num_samples=5000):
    num_samples = min(num_samples, data.shape[0])
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    x_samples = data[indices]
    targets_samples = targets[indices]

    # Encode the samples
    latents = model.encode(x_samples)
    latent_pca = pca.transform(np.squeeze(x_samples.reshape(x_samples.shape[0], -1)))
    targets_samples = targets_samples if targets_samples.ndim == 1 or targets_samples.shape[
        1] == 1 else targets_samples[:, 0]
    ut.plot2d_analysis(latent_pca, targets_samples, title="PCA transform", labels_dict=labels_dict, legend=True)
    ut.plot2d_analysis(latents, targets_samples, title="POLCA-Net latent", labels_dict=labels_dict, legend=True)


def loss_interaction_analysis(model):
    report_dict, df = model.polca_loss.loss_analyzer.report()
    print(f"Loss Interaction Analysis Report:")
    print(f"Total interactions: {report_dict['total_interactions']}")
    print(f"Total conflicts: {report_dict['total_conflicts']}")
    print(f"Overall conflict rate: {report_dict['overall_conflict_rate']:.4f}")
    print(f"\nPairwise Statistics (sorted by similarity):")

    def color_cells(val, column):
        if column == 'relationship':
            colors = {"Strongly Cooperative": "color: green", "Weakly Cooperative": "color: green",
                      "Weakly Conflicting": "color: coral", "Strongly Conflicting": "color: red"}
            return colors.get(val, "")
        elif column == 'avg_similarity':
            if val > model.polca_loss.loss_analyzer.conflict_threshold:
                return 'color: green'
            elif val > 0:
                return 'color: green'
            elif val > -model.polca_loss.loss_analyzer.conflict_threshold:
                return 'color: coral'
            else:
                return 'color: red'
        return ''

    styled_df = df.style.apply(lambda x: [color_cells(xi, col) for xi, col in zip(x, x.index)], axis=1).format(
        {'interactions': '{:.0f}', 'conflicts': '{:.0f}', 'conflict_rate': '{:.4f}', 'avg_similarity': '{:.4f}'})

    if in_jupyterlab():
        display(styled_df)
    else:
        print(df)

    ut.set_fig_prefix("")
    save_df_to_csv(df, "loss_interaction_report.csv")
