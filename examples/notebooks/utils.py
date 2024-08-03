import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

from polcanet.polcanet_reports import save_figure, save_latex_table


def make_grid(images, nrow=8, padding=2, pad_value=0):
    """
    Arrange a batch of images into a grid.

    Parameters:
    - images (numpy.ndarray): A 3D or 4D array of shape (N, H, W) or (N, H, W, C) representing the batch of images.
        N is the number of images.
        H and W are the height and width of each image.
        C is the number of channels (1 for grayscale, 3 for RGB).
    - nrow (int): Number of images per row in the grid.
    - padding (int): Number of pixels to pad around each image in the grid.
    - pad_value (int or float): Value to use for the padding pixels.

    Returns:
    - grid (numpy.ndarray): A single image (2D or 3D array) representing the grid of images.
    """
    if images.ndim not in {3, 4}:
        raise ValueError("Images should be a 3D array with shape (N, H, W) or a 4D array with shape (N, H, W, C)")

    if images.ndim == 3:
        images = np.expand_dims(images, axis=-1)  # Convert (N, H, W) to (N, H, W, 1)

    N, H, W, C = images.shape
    ncol = int(np.ceil(N / nrow))

    grid_height = H * ncol + padding * (ncol + 1)
    grid_width = W * nrow + padding * (nrow + 1)

    grid = np.full((grid_height, grid_width, C), pad_value, dtype=images.dtype)

    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow

        y = padding + row * (H + padding)
        x = padding + col * (W + padding)
        grid[y: y + H, x: x + W, :] = image

    if C == 1:
        grid = grid.squeeze(axis=-1)  # Convert (H, W, 1) back to (H, W) if grayscale

    return grid


# Function to show a single image
def show_image(ax, img, cmap=None):
    cmap = cmap or "viridis"
    ax.imshow(img, cmap=cmap)
    ax.axis("off")  # Turn off axis


# Function to visualize output images horizontally
def visualise_reconstructed_images(reconstructed_list, title_list, cmap="gray", nrow=5, padding=0, save_fig: str = None):
    # Create a figure for all visualizations to be displayed horizontally
    fig, axs = plt.subplots(1, len(reconstructed_list),
                            figsize=(15, 15))  # Adjust number of subplots and size as needed22
    fig.subplots_adjust(wspace=0.01)
    for ax, reconstructed, title in zip(axs, reconstructed_list, title_list):
        reconstructed = np.squeeze(reconstructed)
        # reconstructed = reconstructed.clip(0, 1)
        # Create a grid of images for plotting
        grid = make_grid(reconstructed, nrow=nrow, padding=padding, pad_value=0)
        show_image(ax, grid, cmap=cmap)
        ax.set_title(title)
    fig_name = "reconstructed_images.pdf"
    save_figure(fig_name)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()


def calculate_metrics(original_images, reconstructed_images):
    """
    Calculate the quality metrics for a set of reconstructed images.

    Parameters:
    - original_images (numpy.ndarray): Array of original images with shape (N, H, W).
    - reconstructed_images (numpy.ndarray): Array of reconstructed images with shape (N, H, W).

    Returns:
    - avg_metrics (dict): Dictionary containing the average metrics:
        - 'Normalized Mean Squared Error': Average normalized MSE between original and reconstructed images (lower is better).
        - 'Peak Signal-to-Noise Ratio': Average PSNR between original and reconstructed images (higher is better).
        - 'Structural Similarity Index': Average SSIM between original and reconstructed images (higher is better).
    """
    metrics = {
            'Normalized Mean Squared Error': [],
            'Peak Signal-to-Noise Ratio': [],
            'Structural Similarity Index': []
    }

    for orig, recon in zip(original_images, reconstructed_images):
        nmse = mean_squared_error(orig, recon) / np.mean(np.square(orig))
        psnr = peak_signal_noise_ratio(orig, recon, data_range=255)
        ssim = structural_similarity(orig, recon, data_range=255)

        metrics['Normalized Mean Squared Error'].append(nmse)
        metrics['Peak Signal-to-Noise Ratio'].append(psnr)
        metrics['Structural Similarity Index'].append(ssim)

    # Calculate average metrics
    avg_metrics = {
            'Normalized Mean Squared Error': np.mean(metrics['Normalized Mean Squared Error']),
            'Peak Signal-to-Noise Ratio': np.mean(metrics['Peak Signal-to-Noise Ratio']),
            'Structural Similarity Index': np.mean(metrics['Structural Similarity Index'])
    }

    return avg_metrics


def get_images_metrics_table(original_images, reconstructed_sets):
    """
    Display a table of average quality metrics for multiple sets of reconstructed images.

    Parameters:
    - original_images (numpy.ndarray): Array of original images with shape (N, H, W).
    - reconstructed_sets (dict): Dictionary where keys are set names and values are arrays of reconstructed images.

    Returns:
    - metrics_table (pandas.DataFrame): DataFrame containing the average metrics for each set of reconstructed images.
    """
    metrics_list = []

    for set_name, reconstructed_images in reconstructed_sets.items():
        avg_metrics = calculate_metrics(original_images, reconstructed_images)
        avg_metrics['Method'] = set_name
        metrics_list.append(avg_metrics)

    metrics_table = pd.DataFrame(metrics_list)

    # Formatting the table
    metrics_table = metrics_table.round({
            'Normalized Mean Squared Error': 4,
            'Peak Signal-to-Noise Ratio': 4,
            'Structural Similarity Index': 4
    })

    # Move the 'Set' column to the first place
    cols = ['Method'] + [col for col in metrics_table.columns if col != 'Method']
    metrics_table = metrics_table[cols]

    return metrics_table


def get_pca(x, n_components=None, ax=None, title="", save_fig=None):
    total_pca = decomposition.PCA()
    total_pca.fit(np.squeeze(x.reshape(x.shape[0], -1)))
    n_components = n_components or np.prod(x[0].shape)
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(np.squeeze(x.reshape(x.shape[0], -1)))

    # Compute cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(total_pca.explained_variance_ratio_)
    plot_components_cdf(cumulative_variance_ratio, n_components, title, ax, save_fig)

    return pca


def plot_components_cdf(cumulative_variance_ratio, n_components, title="", ax=None, save_fig=None):
    # Number of components needed for 90% and 95% explained variance
    total_components = len(cumulative_variance_ratio)
    components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    # If ax is not provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(range(1, n_components + 1), cumulative_variance_ratio[:n_components], color="black",
            label="explained variance", lw=1)
    ax.set_xlabel(f'Number of components: {n_components} of {total_components}')
    ax.set_ylabel('Cumulative explained variance ratio')
    ax.set_title(title)
    ax.grid(True)
    # Add some reference lines
    ax.axhline(y=0.9, color='r', linestyle='--', lw=1,
               label=f"90% by {(100 * components_90 / total_components):.1f}% ({components_90}) components")
    ax.axhline(y=0.95, color='g', linestyle='--', lw=1,
               label=f"95% by {(100 * components_95 / total_components):.1f}% ({components_95}) components")
    ax.legend()
    ax.set_box_aspect(2 / 3)
    fig_name = "pca_explained_variance.pdf"
    save_figure(fig_name)
    if save_fig:
        plt.savefig(save_fig)


def image_metrics_table(experiment_data: dict):
    tables = []
    for k, (images, model, pca) in experiment_data.items():
        # Reconstruct the images using the autoencoder
        _, ae_reconstructed = model.predict(images)

        # Reconstruct the images by PCA
        pca_latents = pca.transform(images.reshape(images.shape[0], -1))
        pca_reconstructed = pca.inverse_transform(pca_latents)
        pca_reconstructed = pca_reconstructed.reshape(images.shape[0], images.shape[1], images.shape[2])
        original_images = np.squeeze(images)
        reconstructed_sets = {
                f"POLCA {k}": ae_reconstructed,
                f"PCA {k}": pca_reconstructed
        }

        item = get_images_metrics_table(original_images, reconstructed_sets)
        tables.append(item)

    df_table = pd.concat(tables).set_index("Method")
    save_latex_table(df_table, "image_metrics.tex")
    return df_table


def make_classification_report(model, pca, X, y):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if X.ndim > 2:
        X_train_pca = pca.transform(X_train.reshape(X_train.shape[0], -1))
        X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))
    else:
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

    print(X_train_pca.shape, X_test_pca.shape)

    # Transform the data using POLCA-Net
    X_train_polca = model.predict(X_train)[0][:, :pca.n_components]
    X_test_polca = model.predict(X_test)[0][:, :pca.n_components]
    print(X_train_polca.shape, X_test_polca.shape)

    # Define classifiers
    classifiers = {
            "Logistic Regression": LogisticRegression(),
            "Gaussian Naive Bayes": GaussianNB(),
            "Linear SVM": SVC(kernel="linear", probability=True),
            "Ridge Classifier": RidgeClassifier(),
            "Perceptron": Perceptron(),
    }

    # Train and evaluate classifiers on both PCA and POLCA-Net transformed datasets
    results = []

    for name, clf in classifiers.items():
        # Train on PCA
        clf.fit(minmax_scale(X_train_pca), y_train)
        y_pred_pca = clf.predict(minmax_scale(X_test_pca))
        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        report_pca = classification_report(y_test, y_pred_pca, output_dict=True)
        cm_pca = confusion_matrix(y_test, y_pred_pca)

        # Train on POLCA-Net
        clf.fit(minmax_scale(X_train_polca), y_train)
        y_pred_polca = clf.predict(minmax_scale(X_test_polca))
        accuracy_polca = accuracy_score(y_test, y_pred_polca)
        report_polca = classification_report(y_test, y_pred_polca, output_dict=True)
        cm_polca = confusion_matrix(y_test, y_pred_polca)

        # Append results
        results.append(
            {
                    "Classifier": name,
                    "Transformation": "PCA",
                    "Accuracy": accuracy_pca,
                    "Precision": report_pca["weighted avg"]["precision"],
                    "Recall": report_pca["weighted avg"]["recall"],
                    "F1-Score": report_pca["weighted avg"]["f1-score"],
                    "Confusion Matrix": cm_pca,
            }
        )

        results.append(
            {
                    "Classifier": name,
                    "Transformation": "POLCA",
                    "Accuracy": accuracy_polca,
                    "Precision": report_polca["weighted avg"]["precision"],
                    "Recall": report_polca["weighted avg"]["recall"],
                    "F1-Score": report_polca["weighted avg"]["f1-score"],
                    "Confusion Matrix": cm_polca,
            }
        )

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)

    # Display the main metrics table
    main_metrics_df = results_df.drop(columns=["Confusion Matrix"])
    main_metrics_df.round(2)

    df = main_metrics_df

    # Step 1: Pivot the DataFrame to separate PCA and POLCA-Net results
    df_metrics = df.pivot(index='Classifier', columns='Transformation',
                          values=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

    # Step 2: Perform the Wilcoxon Signed-Rank Test and Calculate Median Differences
    comparison_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    wilcoxon_results = {}

    for metric in comparison_metrics:
        pca_result = df[df["Transformation"] == "PCA"][metric]
        polca_result = df[df["Transformation"] == "POLCA"][metric]

        # Perform Wilcoxon Signed-Rank test
        stat, p_value = wilcoxon(pca_result.values, polca_result.values)

        # Calculate median difference
        median_diff = (pca_result - polca_result).median()

        # Determine which method is better
        if p_value < 0.05:
            if median_diff > 0:
                better_method = "PCA Better"
            else:
                better_method = "POLCA-Net Better"
        else:
            better_method = "No Significant Difference"

        # Store the results
        wilcoxon_results[metric] = {
                'Wilcoxon Test Statistic': stat,
                'P-Value': p_value,
                'Significant (p < 0.05)': f'Yes, {better_method} is better' if p_value < 0.05 else 'No better method'
        }

    # Convert the results to a DataFrame
    df_wilcoxon = pd.DataFrame(wilcoxon_results).T

    # Display the DataFrames
    print("Performance Metrics DataFrame:")
    display(df_metrics)
    save_latex_table(df_metrics, "classification_metrics.tex")

    print("\nWilcoxon Signed-Rank Test Results DataFrame:")
    display(df_wilcoxon)
    save_latex_table(df_wilcoxon, "wilcoxon_signed_test.tex")
    return df_metrics, df_wilcoxon


def plot_train_images(x, title, n=1, cmap="gray", save_fig=None):
    # Plot original and reconstructed signals for a sample
    fig, axes = plt.subplots(1, n)
    fig.subplots_adjust(wspace=0.01)
    im_list = list(range(n))
    for i in im_list:
        axes[i].imshow(x[i], cmap=cmap)
        if i == n // 2:
            axes[i].set_title(f"{title}")
        axes[i].axis("off")

    fig_name = "train_images.pdf"
    save_figure(fig_name)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()


def plot2d_analysis(X, y, title, legend=False):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X[:, 0], X[:, 1], label=y, cmap="tab10", c=y)
    ax.set_xlabel("component: 0")
    ax.set_ylabel("component 1")
    ax.axis("square")
    if legend:
        legend1 = plt.legend(*scatter.legend_elements(), title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.show()
    return fig, ax


class ExperimentInfoHandler:
    """
    This class is designed to control the information relative to the current experiment developed in the current file
     script where it is created.
     It will manage a folder name for the experiment which is located on the current working directory.
     Also, it will manage the experiment name, description and the random seed used in the experiment.
     This class will allow to get path names for saving images, generated data, and other files related to the
        experiment, for instance, text files containing latex tables with the results of the experiment.

    """

    def __init__(self, name: str, description: str, random_seed: int):
        self.experiment_name = name
        self.experiment_description = description
        self.random_seed = random_seed
        self.experiment_folder = Path.cwd() / f"{self.experiment_name}"
        self.create_experiment_folder()
        self.add_experiment_info_to_folder()

    def create_experiment_folder(self):
        self.experiment_folder.mkdir(exist_ok=True)

    def get_experiment_folder(self):
        return self.experiment_folder

    def get_name(self, str_name):
        # let the saver decide the extension and must be included in the figure name
        return self.experiment_folder / f"{str_name}"

    def add_experiment_info_to_folder(self):
        info = {
                "Experiment Name": self.experiment_name,
                "Experiment Description": self.experiment_description,
                "Random Seed": self.random_seed,
                "Date and Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Experiment Folder": str(self.experiment_folder),
                "Python Version": sys.version,
                "PyTorch Version": torch.__version__,
                "CUDA Version": torch.version.cuda,
                "CUDNN Version": torch.backends.cudnn.version(),
                "Device": torch.cuda.get_device_name(0),
                # Add the name type or class of the gpu if any
                "GPU Type": torch.cuda.get_device_capability(0),
                "Number of GPUs": torch.cuda.device_count(),
                "Current GPU": torch.cuda.current_device(),
                "GPU Available": torch.cuda.is_available(),
                "CPU Cores": os.cpu_count(),
                "CPU Threads": torch.get_num_threads()
        }
        with open(self.experiment_folder / "experiment_info.json", "w") as f:
            json.dump(info, f, indent=4)
