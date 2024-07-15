import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate


def plot_stdev_pct(model):
    x_plot = np.arange(1, model.std_metrics.shape[0] + 1)
    y_plot = 100 * (model.std_metrics / np.sum(model.std_metrics))
    str_texts = [f"{round(t, 2):.1f}%" for t in y_plot.tolist()]

    plt.plot(x_plot, y_plot, "o-")
    for _x, _y, _s in zip(x_plot, y_plot, str_texts):
        plt.text(_x, _y, _s)
    plt.title("Stdev percentage")
    plt.show()


def plot_cumsum_variance(model, data):
    latents, reconstructed = model.predict(data)
    inputs = data
    arr_x = np.zeros((latents.shape[1], latents.shape[1]))
    idx = np.tril_indices(latents.shape[1])
    arr_x[idx[0], idx[1]] = 1
    errors = []
    for i in range(latents.shape[1]):
        w = arr_x[i, :]
        latents = model.encode(inputs)
        reconstructed = model.decode(latents, w)
        error = np.mean((inputs - reconstructed) ** 2)
        errors.append(error)

    errors = np.array(errors)
    norm_errors = 100 * (errors / (np.max(errors)))

    plt.plot(norm_errors,
             label="reconstruction mse",
             color="black",
             alpha=0.7,
             linewidth=1,
             marker=None,
             )

    plt.title("Percentage error reduction by adding successive components")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_latent_feature_importance(model, data):
    """
    Compute and visualize latent feature importance based on variance.

    Parameters:
    - encoder: The autoencoder encoder instance
    - data: Input data (numpy array)

    Returns:
    - feature_importance: Array of importance scores (variances) for each latent feature
    """
    # Encode the data to get latent representations
    latents = model.encode(data)

    # Compute feature importance as variance
    feature_importance = np.var(latents, axis=0)

    # Normalize the importance scores
    feature_importance_normalized = feature_importance / np.sum(feature_importance)

    # Plot results
    plt.figure()
    plt.bar(range(len(feature_importance_normalized)), feature_importance_normalized)
    plt.xlabel('Latent Feature Index')
    plt.ylabel('Normalized Variance')
    plt.title('Latent Feature Importance (Based on Variance)')
    plt.xticks(range(0, len(feature_importance_normalized), max(1, len(feature_importance_normalized) // 10)))

    # Add a trend line
    z = np.polyfit(range(len(feature_importance_normalized)), feature_importance_normalized, 1)
    p = np.poly1d(z)
    plt.plot(range(len(feature_importance_normalized)), p(range(len(feature_importance_normalized))), "r--", alpha=0.8)

    plt.show()

    # Plot cumulative importance
    cumulative_importance = np.cumsum(feature_importance_normalized)
    plt.figure()
    plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'bo-')
    plt.xlabel('Number of Latent Features')
    plt.ylabel('Cumulative Normalized Variance')
    plt.title('Cumulative Latent Feature Importance')
    plt.show()

    # Print some statistics
    print(f"First feature importance: {feature_importance_normalized[0]:.4f}")
    print(f"Last feature importance: {feature_importance_normalized[-1]:.4f}")
    print(f"Ratio of first to last feature importance:"
          f" {feature_importance_normalized[0] / feature_importance_normalized[-1]:.4f}")
    print(f"Number of features explaining 80% of the variance: {np.argmax(cumulative_importance >= 0.8) + 1}")

    # Compute and print the Intrinsic Dimension
    intrinsic_dim = 2 * np.sum(feature_importance) ** 2 / np.sum(feature_importance ** 2)
    print(f"Intrinsic Dimension: {intrinsic_dim:.2f}")


def analyze_reconstruction_error(model, data, n_samples=10000):
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
    plt.show()

    # Print some statistics
    print(f"Mean Reconstruction Error: {mean_error:.4f}")
    print(f"Median Reconstruction Error: {median_error:.4f}")
    print(f"Min Reconstruction Error: {np.min(errors):.4f}")
    print(f"Max Reconstruction Error: {np.max(errors):.4f}")


def plot_scatter_corr_matrix(model, latents=None, data=None, n_components=5, max_samples=5000):
    if latents is None and data is None:
        raise ValueError("Either latents or data must be provided")

    if latents is None:
        latents, reconstructed = model.predict(data)

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


def show_correlation_matrix(model, latents=None, data=None):
    if latents is None and data is None:
        raise ValueError("Either latents or data must be provided")

    if latents is None:
        latents, reconstructed = model.predict(data)

    # Compute the correlation matrix
    x_corr = np.corrcoef(latents.T)
    corr = pd.DataFrame(x_corr)

    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Create the heatmap
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)

    # Add colorbar for reference
    fig.colorbar(cax)

    # Add title
    plt.title("Correlation Matrix of Latent Components")

    # Show the plot
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
    component_table = []
    for i in range(top_n):
        component_table.append([i + 1, f"{explained_variance_ratio[i]:.4f}", f"{cumulative_variance_ratio[i]:.4f}",
                                f"{np.nanmean(np.abs(corr[i, i + 1:])):.4f}"])
    print(tabulate(component_table,
                   headers=["Component", "Variance Ratio", "Cumulative Variance", "Mean |Correlation| with Others"]))
    print()


def orthogonality_test_analysis(model, data, num_samples=1000, n_components=10):
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
    fig, ax = plt.subplots()
    cax = ax.matshow(cosine_sim, cmap='coolwarm', vmin=-1, vmax=1, alpha=0.8, interpolation='nearest', aspect='auto')
    fig.colorbar(cax)
    ax.set_title('Cosine Similarity Matrix of Latent Features')
    plt.show()

    # Plot scatter correlation matrix
    plot_scatter_corr_matrix(model, latents=latent_x, n_components=n_components)


def variance_test_analysis(model, data, num_samples=1000):
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
    ax.plot(components, normalized_variances, 'o-', label='Normalized Variances')
    ax.plot(components, exp_fit, 'r--', label='Exponential Fit')
    ax.set_title('Variance Distribution and Exponential Fit')
    ax.set_xlabel('Latent Components')
    ax.set_ylabel('Normalized Variance')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot cumulative variance
    cumulative_variance = np.cumsum(normalized_variances)
    fig, ax = plt.subplots()
    ax.plot(components, cumulative_variance, 'o-', label='Cumulative Variance')
    ax.set_title('Cumulative Variance of Latent Components')
    ax.set_xlabel('Latent Components')
    ax.set_ylabel('Cumulative Variance')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Analyze and plot variance concentration
    top_n = min(20, len(variances))  # Analyze top 20 components or all if less than 10
    component_table = []
    for i in range(top_n):
        component_table.append([i + 1, f"{normalized_variances[i]:.4f}", f"{cumulative_variance[i]:.4f}"])

    # Plot variance concentration in top components
    fig, ax = plt.subplots()
    ax.barh(range(1, top_n + 1), normalized_variances[:top_n], fill=False, edgecolor='k', linewidth=1)
    # set barh without fill

    ax.set_title('Variance Concentration in Top Components')
    ax.set_ylabel('Latent Components')
    ax.set_xlabel('Normalized Variance')
    # increase the limit of x-axis
    x_axis_lim = np.max(normalized_variances[:top_n])
    ax.set_xlim(0, x_axis_lim+0.2)

    ax.set_yticks(range(1, top_n + 1), labels=range(1, top_n + 1))
    ax.invert_yaxis()  # labels read top-to-bottom
    # Add the value to each bar at the middle
    for i, (v1,v2) in enumerate(zip(normalized_variances[:top_n], cumulative_variance[:top_n])):
        # make the text color black an appearing as a percentage
        ax.text(x_axis_lim+0.015, i + 1, f'{v1:.2f}', color='black', va='center', ha='left',
                fontsize=8)
        ax.text(x_axis_lim+0.075, i + 1, f'{v2:.2f}', color='black', va='center', ha='left',
                fontsize=8)

    plt.tight_layout()
    plt.show()


def linearity_tests_analysis(model, data, num_samples=100):
    """
    Analyze the linearity properties of the autoencoder.

    Parameters:
    - encoder: The autoencoder encoder instance
    - data: Input data (numpy array)
    - num_samples: Number of samples to test (default: 100)
    """
    # Select random samples from the data
    indices = np.random.choice(data.shape[0], num_samples, replace=False)
    x_samples = data[indices]
    y_samples = data[np.random.choice(data.shape[0], num_samples, replace=False)]

    # Encode the samples
    latent_x = model.encode(x_samples)
    latent_y = model.encode(y_samples)

    # Additive property test
    latent_x_plus_y = latent_x + latent_y
    decoded_latent_x_plus_y = model.decode(latent_x_plus_y)
    decoded_latent_x = model.decode(latent_x)
    decoded_latent_y = model.decode(latent_y)
    decoded_latent_x_plus_decoded_latent_y = decoded_latent_x + decoded_latent_y

    # Homogeneity property test with scalar alpha
    alpha_scalar = np.random.uniform(0.1, 2.0)  # Random scalar value
    latent_alpha_x_scalar = alpha_scalar * latent_x
    decoded_latent_alpha_x_scalar = model.decode(latent_alpha_x_scalar)
    alpha_decoded_latent_x_scalar = alpha_scalar * decoded_latent_x

    # Calculate differences for reporting
    differences_additive = np.abs(decoded_latent_x_plus_y - decoded_latent_x_plus_decoded_latent_y)
    differences_homogeneity_scalar = np.abs(decoded_latent_alpha_x_scalar - alpha_decoded_latent_x_scalar)

    # Reporting text with statistics
    report = f"""
    Linearity Tests Analysis
    =========================

    This report analyzes the linearity properties of the autoencoder. We used a sample size of
    {num_samples} randomly selected data points for the analysis.

    The linearity properties of the features are assessed through two tests: additive property
    and homogeneity property. The results are summarized below:

    1. Additive Property:
    ---------------------
    The additive property is tested to verify if:

    f(z_x + z_y) = f(z_x) + f(z_y)

    The differences between the left-hand side and the right-hand side of the equation are
    summarized below:

    - Mean difference: {np.mean(differences_additive):.4f}
    - Max difference: {np.max(differences_additive):.4f}
    - Min difference: {np.min(differences_additive):.4f}

    2. Homogeneity Property (Scalar alpha):
    ---------------------------------------
    The homogeneity property is tested to verify if:

    f(a.z_x) = a.f(z_x_)

    The differences between the left-hand side and the right-hand side of the equation are
    summarized below:

    - Mean difference: {np.mean(differences_homogeneity_scalar):.4f}
    - Max difference: {np.max(differences_homogeneity_scalar):.4f}
    - Min difference: {np.min(differences_homogeneity_scalar):.4f}
    """

    print(report)

    # Plot results
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(r"Autoencoder Linearity Tests")

    # Additive property plot
    for i in range(10):  # Plot first 10 samples
        axs[0, 0].scatter(decoded_latent_x_plus_y[i].flatten(), decoded_latent_x_plus_decoded_latent_y[i].flatten(),
                          alpha=0.5, s=1)
    max_val = max(decoded_latent_x_plus_y.max(), decoded_latent_x_plus_decoded_latent_y.max())
    min_val = min(decoded_latent_x_plus_y.min(), decoded_latent_x_plus_decoded_latent_y.min())
    axs[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[0, 0].set_title("Additive Property:\n " + r"$f(z_x + z_y) = f(z_x) + f(z_y)$")
    axs[0, 0].set_xlabel(r"$f(z_x + z_y)$")
    axs[0, 0].set_ylabel(r"$f(z_x) + f(z_y)$")
    # Add correlation legend to the plot
    axs[0, 0].text(0.5, 0.1,
                   f"Correlation: {np.corrcoef(decoded_latent_x_plus_y.flatten(),
                                               decoded_latent_x_plus_decoded_latent_y.flatten())[0, 1]:.4f}",
                   ha='center', va='center', transform=axs[0, 0].transAxes, fontsize=8)

    # Homogeneity property plot (scalar alpha)
    for i in range(10):  # Plot first 10 samples
        axs[0, 1].scatter(decoded_latent_alpha_x_scalar[i].flatten(), alpha_decoded_latent_x_scalar[i].flatten(),
                          alpha=0.5, s=1)
    max_val = max(decoded_latent_alpha_x_scalar.max(), alpha_decoded_latent_x_scalar.max())
    min_val = min(decoded_latent_alpha_x_scalar.min(), alpha_decoded_latent_x_scalar.min())
    axs[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axs[0, 1].set_title("Homogeneity Property:\n" + r"$f(\alpha z_x) = \alpha f(z_x)$")
    axs[0, 1].set_xlabel(r"$f(\alpha z_x)$")
    axs[0, 1].set_ylabel(r"$\alpha f(z_x)$")
    # Add correlation legend to the plot
    axs[0, 1].text(0.5, 0.1,
                   f"Correlation: {np.corrcoef(decoded_latent_alpha_x_scalar.flatten(),
                                               alpha_decoded_latent_x_scalar.flatten())[0, 1]:.4f}",
                   ha='center', va='center', transform=axs[0, 1].transAxes, fontsize=8)

    # Difference plots for additive property
    axs[1, 0].hist(differences_additive.flatten(), bins=50, density=True, alpha=0.7)
    axs[1, 0].set_title("Additive Property Differences")
    axs[1, 0].set_xlabel(r"$|f(z_x + z_y) - ( f(z_x) + f(z_y) )|$")
    axs[1, 0].set_ylabel("Density")

    # Difference plots for homogeneity property (scalar alpha)
    axs[1, 1].hist(differences_homogeneity_scalar.flatten(), bins=50, density=True, alpha=0.7)
    axs[1, 1].set_title(r"Homogeneity Property Differences (Scalar $\alpha$)")
    axs[1, 1].set_xlabel(r"$|f(\alpha z_x) - \alpha f(z_x)|$")
    axs[1, 1].set_ylabel("Density")

    plt.tight_layout()
    plt.show()
