from typing import Tuple

import cmasher as cmr
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from polcanet.utils import save_figure


# def plot_reconstruction_mask(model, data,n_components=None, save_fig: str = None):
#     latents, reconstructed = model.predict(data)
#     inputs = data
#     arr_x = np.zeros((latents.shape[1], latents.shape[1]))
#     idx = np.tril_indices(latents.shape[1])
#     arr_x[idx[0], idx[1]] = 1
#     errors = []
#     for i in range(latents.shape[1]):
#         w = arr_x[i, :]
#         latents = model.encode(inputs)
#         reconstructed = model.decode(latents, w)
#         error = np.mean((inputs - reconstructed) ** 2)
#         errors.append(error)
#
#     errors = np.array(errors)
#     norm_errors = 100 * (errors / (np.max(errors)))
#
#     plt.plot(norm_errors,
#              label="reconstruction mse",
#              color="black",
#              alpha=0.7,
#              linewidth=1,
#              marker=None,
#              )
#
#     plt.title("Percentage error reduction by adding successive components")
#     plt.legend()
#     plt.tight_layout()
#     fig_name = save_fig or "reconstruction_error_reduction.pdf"
#     save_figure(fig_name)
#     plt.show()


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


def plot_corr_scatter(corr_matrix, latents, n, save_fig: str = None):
    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(n - 1, n)

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0., hspace=0.)

    cmap = cmr.iceburn
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

    # Add a main title
    # plt.suptitle("Cosine Similarity Matrix of Latent Features")
    fig_name = save_fig or "scatter_correlation_matrix.pdf"
    save_figure(fig_name)
    # Show plot
    plt.show()


def analyze_latent_space(model, data=None, latents=None):
    """
    Perform a comprehensive text-based analysis of the latent space for a specialized autoencoder
    that concentrates variance in the first dimensions and aims to disentangle features by orthogonalization
    of the latent space.

    Parameters:
    - encoder: The autoencoder encoder instance
    - data: Input data (numpy array), optional if latents are provided
    - latents: Latent representations (numpy array), optional if data is provided
    """
    if latents is None and data is None:
        raise ValueError("Either latents or data must be provided")

    if latents is None:
        latents = model.encode(data)

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
        print("Poor concentration of variance. The encoder may need adjustment.")
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
        print("Poor orthogonality. The encoder may need adjustment.")
    print()

    print("4. Detailed Component Analysis")
    print("-" * 30)
    top_n = min(10, n_components)  # Analyze top 10 components or all if less than 10

    component_df = pd.DataFrame(
        columns=["Component", "Variance Ratio", "Cumulative Variance", "Mean |Correlation| with Others"])
    for i in range(top_n):
        corr_list = np.array([idx for idx in range(top_n) if idx != i])
        component_df.loc[i] = {
                "Component": i + 1,
                "Variance Ratio": f"{explained_variance_ratio[i]:.4f}",
                "Cumulative Variance": f"{cumulative_variance_ratio[i]:.4f}",
                "Mean |Correlation| with Others": f"{np.mean(np.abs(corr[i, corr_list])):.4f}"
        }

    # print(component_df.to_string())
    print(tabulate(component_df, headers="keys", tablefmt="grid", showindex=False))


def plot_correlation_matrix(corr_matrix, threshold=15, save_fig: str = None):
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
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=(corr_matrix.shape[0] <= threshold),
                fmt='.2f', annot_kws={"size": 10})

    # Add titles and labels
    ax.set_title('Cosine Similarity Matrix of Latent Features')
    plt.xticks(rotation=45, ha='right')

    # Tight layout for better spacing
    plt.tight_layout()
    fig_name = save_fig or "correlation_matrix.pdf"
    save_figure(fig_name)
    # Show plot
    plt.show()


def orthogonality_test_analysis(model, data, num_samples=1000, n_components=10, save_figs: Tuple[str] = None):
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

    # Calculate cosine similarity matrix
    cosine_sim = cosine_similarity(latent_x.T)

    # Extract the upper triangular part of the similarity matrix, excluding the diagonal
    upper_triangular_indices = np.triu_indices_from(cosine_sim, k=1)
    upper_triangular_values = cosine_sim[upper_triangular_indices]

    # Reporting text with statistics
    report = f"""
    Orthogonality Test Analysis
    ============================

    This report analyzes the orthogonality of the latent features generated by the autoencoder.
    We used a sample size of {num_samples} randomly selected data points for the analysis.

    The orthogonality of the features is assessed by minimizing the cosine distance between the 
    latent features. The cosine similarity values between the features are summarized below:

    - Mean cosine similarity: {np.mean(upper_triangular_values):.4f}
    - Max cosine similarity: {np.max(upper_triangular_values):.4f}
    - Min cosine similarity: {np.min(upper_triangular_values):.4f}
    """

    print(report)

    # Plot cosine similarity matrix
    save_fig = save_figs[0] if save_figs else None
    plot_correlation_matrix(pd.DataFrame(cosine_sim), threshold=15, save_fig=save_fig)

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
    n_components = data.shape[1]
    # Select random samples from the data
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    x_samples = data[indices]

    # Encode the samples
    latent_x = model.encode(x_samples)

    # Calculate variances of the latent features
    variances = np.var(latent_x, axis=0)

    # Calculate center of mass
    components = np.arange(1, len(variances) + 1)
    center_of_mass = np.sum(components * variances) / np.sum(variances)

    # Calculate exponential fit for variance
    normalized_variances = variances / np.sum(variances)
    exp_fit = np.exp(-components)
    exp_fit /= np.sum(exp_fit)

    # Reporting text with statistics
    report = f"""
    Variance Test Analysis
    =======================

    This report analyzes the variance concentration of the latent features generated by the autoencoder.
    We used a sample size of {num_samples} randomly selected data points for the analysis.

    The variance concentration of the features is assessed by minimizing the center of mass of the 
    latent space and fitting the variance distribution to an exponential distribution. The results 
    are summarized below:

    - Center of mass: {center_of_mass:.4f}
    - Variance fit to exponential distribution (sum of squared differences):
     {np.sum((normalized_variances - exp_fit) ** 2):.4f}
    """

    print(report)

    # Plot variance distribution and exponential fit
    fig, ax = plt.subplots()
    ax.plot(components, normalized_variances, 'o-', label='Normalized Variances', color='black', alpha=0.7, linewidth=1,
            markersize=2)
    ax.plot(components, exp_fit, 'r--', label='Exponential ref')
    ax.set_title('Variance Distribution of latent space')
    ax.set_xlabel('Components')
    ax.set_ylabel('Normalized Variance')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    fig_name = "variance_distribution.pdf"
    save_figure(fig_name)
    plt.show()

    # Plot cumulative variance
    # Compute cumulative explained variance ratio
    cumulative_variance = np.cumsum(normalized_variances)
    # plot_components_cdf(cumulative_variance, n_components, title='Cumulative Variance', ax=ax)
    # fig_name = save_fig or "cumulative_variance.pdf"
    # save_figure(fig_name)

    components_90 = np.argmax(cumulative_variance >= 0.9) + 1
    components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
    ax.plot(components, cumulative_variance, 'o-', label='Cumulative Variance', color='black', alpha=0.7, linewidth=1,
            markersize=2)
    ax.set_title('Cumulative Variance of latent space')
    ax.set_xlabel('Components')
    ax.set_ylabel('Cumulative Variance')
    ax.grid(True)

    # Add some reference lines
    total_components = len(variances)
    ax.axhline(y=0.9, color='r', linestyle='--', lw=1,
               label=f"90% by {(100 * components_90 / total_components):.1f}% ({components_90}) components")
    ax.axhline(y=0.95, color='g', linestyle='--', lw=1,
               label=f"95% by {(100 * components_95 / total_components):.1f}% ({components_95}) components")
    ax.legend()
    # set x-axis from 1 to n_components
    # ax.set_xlim(1, n_components)
    # set y-axis from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_box_aspect(2 / 3)
    fig_name = "cumulative_variance.pdf"
    save_figure(fig_name)
    plt.show()

    # Analyze and plot variance concentration
    top_n = min(20, len(variances))  # Analyze top 20 components or all if less than 10
    component_table = []
    for i in range(top_n):
        component_table.append([i + 1, f"{normalized_variances[i]:.4f}", f"{cumulative_variance[i]:.4f}"])

    # Plot variance concentration in top components
    fig, ax = plt.subplots()
    ax.barh(range(1, top_n + 1), normalized_variances[:top_n], fill=False, edgecolor='k', linewidth=1)

    ax.set_title('Variance Concentration of latent space')
    ax.set_ylabel('Components')
    ax.set_xlabel('Normalized Variance')
    # increase the limit of x-axis
    x_axis_lim = np.max(normalized_variances[:top_n])
    ax.set_xlim(0, x_axis_lim + 0.2)

    ax.set_yticks(range(1, top_n + 1), labels=range(1, top_n + 1))
    ax.invert_yaxis()  # labels read top-to-bottom
    # Add the value to each bar at the middle
    for i, (v1, v2) in enumerate(zip(normalized_variances[:top_n], cumulative_variance[:top_n])):
        # make the text color black an appearing as a percentage
        ax.text(x_axis_lim + 0.015, i + 1, f'{v1:.2f}', color='black', va='center', ha='left',
                fontsize=8)
        ax.text(x_axis_lim + 0.075, i + 1, f'{v2:.2f}', color='black', va='center', ha='left',
                fontsize=8)

    plt.tight_layout()
    fig_name = "variance_concentration.pdf"
    save_figure(fig_name)
    plt.show()


def linearity_test_plot_old(model, x, y, alpha_min=-1, alpha_max=1, save_fig: str = None):
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
        #     Generate alpha values between -2 and 2
        alpha = torch.linspace(alpha_min, alpha_max, steps=shape[0], device=x.device).unsqueeze(1)
        xx_alpha = f(alpha * x)  # Unsqueeze alpha to match x's dimensions
        alpha = alpha.repeat(1, fx_shape[1])
        yy_alpha = alpha * f(x)
        xx_alpha = xx_alpha.cpu().numpy().flatten()
        yy_alpha = yy_alpha.cpu().numpy().flatten()
        alpha_colors = alpha.cpu().numpy().flatten()
        homogeneity_corr = np.corrcoef(xx_alpha, yy_alpha)[0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    print(xx.shape)
    ax1.scatter(xx, yy, c=alpha_colors, cmap='coolwarm', alpha=0.7, s=1, rasterized=True)
    min_val = min(xx.min(), yy.min())
    max_val = max(xx.max(), yy.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    ax1.set_xlabel(r"$f(x + y)$")
    ax1.set_ylabel(r"$f(x) + f(y)$")
    ax1.set_title(f"Additivity\n correlation: {additivity_corr:.4f}")
    # # Add horizontal and vertical lines at 0
    ax1.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax1.axvline(x=0, color='k', linestyle=':', linewidth=0.5)
    ax1.set_aspect("equal")

    # Homogeneity plot
    print(xx_alpha.shape)
    scatter = ax2.scatter(xx_alpha, yy_alpha, c=alpha_colors, cmap='coolwarm', alpha=0.7, s=1, rasterized=True)
    min_val = min(xx_alpha.min(), yy_alpha.min())
    max_val = max(xx_alpha.max(), yy_alpha.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
    ax2.set_xlabel(r"$f(\alpha x)$")
    ax2.set_ylabel(r"$\alpha f(x)$")
    ax2.set_title(f"Homogeneity\n correlation: {homogeneity_corr:.4f}")
    ax2.set_aspect("equal")

    # # Add colorbar for alpha values
    fig.colorbar(scatter, ax=ax2, fraction=0.04, label=r"$\alpha$ values")

    # # Add horizontal and vertical lines at 0
    ax2.axhline(y=0, color='k', linestyle=':', linewidth=0.5)
    ax2.axvline(x=0, color='k', linestyle=':', linewidth=0.5)

    plt.tight_layout()
    fig_name = save_fig or "linearity_test.pdf"
    save_figure(fig_name)
    plt.show()
    return additivity_corr, homogeneity_corr


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

        # Test Lipschitz condition
        # We calculate |f(x) - f(y)| and |x - y|
        f_x = f(x)
        f_y = f(y)
        diff_fxy = torch.norm(f_x - f_y, dim=1)
        diff_xy = torch.norm(x - y, dim=1)
        diff_fxy = diff_fxy.cpu().numpy()
        diff_xy = diff_xy.cpu().numpy()
        lipschitz_corr = np.corrcoef(diff_fxy, diff_xy)[0, 1]

    # Plotting results
    #fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Additivity plot
    ax1.scatter(xx, yy, c=alpha_colors, cmap='coolwarm', alpha=0.7, s=1, rasterized=True)
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

    # # Lipschitz condition plot
    # ax3.scatter(diff_xy, diff_fxy, c='k', alpha=0.7, s=2, rasterized=True)
    # ax3.set_xlim(min(diff_xy),max(diff_xy))
    # ax3.set_xlabel(r"$|x - y|$")
    # ax3.set_ylabel(r"$|f(x) - f(y)|$")
    # ax3.set_title(f"Lipschitz:\n{lipschitz_corr:.2f}")
    # ax3.set_aspect("equal")
    # ax3.axis("square")

    plt.tight_layout()

    fig_name = save_fig or "linearity_test.pdf"
    save_figure(fig_name)
    plt.show()

    return additivity_corr, homogeneity_corr, lipschitz_corr


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
