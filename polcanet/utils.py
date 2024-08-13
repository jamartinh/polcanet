import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scienceplots
import torch
import torch.nn as nn
from IPython.display import display
from PIL import Image
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from skimage import exposure
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from sklearn import decomposition
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

sp_path = scienceplots.scienceplots_path
plt.style.use(["science", "no-latex"])

# Query the current default figure size
current_fig_size = plt.rcParams["figure.figsize"]
print(f"Current default figure size: {current_fig_size}")

# Define a scalar factor
scalar_factor = 1.5

# Multiply the current figure size by the scalar factor
new_fig_size = [size * scalar_factor for size in current_fig_size]

# Set the new default figure size
plt.rcParams["figure.figsize"] = new_fig_size

print(f"New default figure size: {new_fig_size}")

SAVE_PATH = ""
SAVE_FIG = False
SAVE_FIG_PREFIX = ""
saved_figures = Counter()


def get_save_path():
    return SAVE_PATH


def set_save_path(path):
    global SAVE_PATH
    SAVE_PATH = Path(path)


def set_fig_prefix(prefix):
    global SAVE_FIG_PREFIX
    SAVE_FIG_PREFIX = prefix


def get_fig_prefix():
    if SAVE_FIG_PREFIX != "":
        return "_" + SAVE_FIG_PREFIX
    return ""


def get_save_fig():
    return SAVE_FIG


def set_save_fig(save_fig):
    global SAVE_FIG
    SAVE_FIG = save_fig


def save_figure(name):
    if get_save_fig():
        name = name.replace(".pdf", f"{get_fig_prefix()}_{saved_figures[name]}.pdf")
        plt.savefig(get_save_path() / Path(name), dpi=300, bbox_inches="tight")
        saved_figures[name] += 1


def save_latex_table(df, name):
    if get_save_fig():
        latex_table = df.reset_index().to_latex(index=False,
                                                # To not include the DataFrame index as a column in the table
                                                # caption="Comparison of ML Model Performance Metrics",
                                                # The caption to appear above the table in the LaTeX document
                                                # label="tab:model_comparison",
                                                # A label used for referencing the table within the LaTeX document
                                                # position="htbp",
                                                # The preferred positions where the table should be placed in the document ('here', 'top', 'bottom', 'page')
                                                # column_format="|l|l|l|l|",  # The format of the columns: left-aligned with vertical lines between them
                                                escape=False,
                                                # Disable escaping LaTeX special characters in the DataFrame
                                                float_format="{:0.4f}".format  # Formats floats to two decimal places
                                                )
        with open(get_save_path() / Path(name), "w") as f:
            f.write(latex_table)


def save_text(text, name):
    if get_save_fig():
        with open(get_save_path() / Path(name), "w") as f:
            f.write(text)


def save_df_to_csv(df, name):
    if get_save_fig():
        df.to_csv(get_save_path() / Path(name), index=False)


def normalize_array(array, value_range=None, scale_each=False):
    def norm_ip(arr, low, high):
        arr = np.clip(arr, low, high)  # Equivalent to clamp_
        arr = (arr - low) / max(high - low, 1e-5)  # Equivalent to sub_ and div_
        return arr

    def norm_range(arr, value_range):
        if value_range is not None:
            return norm_ip(arr, value_range[0], value_range[1])
        else:
            return norm_ip(arr, np.min(arr), np.max(arr))

    if scale_each:
        # Normalize each slice along the first dimension independently
        normalized_array = np.array([norm_range(t, value_range) for t in array])
    else:
        # Normalize the entire array as a whole
        normalized_array = norm_range(array, value_range)

    return normalized_array


def make_grid(images, nrow=8, padding=2, pad_value=0, normalize=True):
    """
    Arrange a batch of images into a grid.

    Parameters:
    - images (numpy.ndarray): A 3D or 4D array of shape (N, H, W) or (N, H, W, C) representing the batch of images.
        N is the number of images.
        H and W are the height and width of each image.
        C is the number of channels (1 for grayscale, 3 for RGB).
    - nrow (int): Number of images per row in the grid.
    - padding (int): Number of pixels to pad around each image in the grid.
    - pad_value (int or float or tuple): Value to use for the padding pixels. Should be a single value for grayscale, or a tuple of 3 values for RGB.

    Returns:
    - grid (numpy.ndarray): A single image (2D or 3D array) representing the grid of images.
    """
    if images.ndim not in {3, 4}:
        raise ValueError("Images should be a 3D array with shape (N, H, W) or a 4D array with shape (N, H, W, C)")

    if images.ndim == 3:
        images = np.expand_dims(images, axis=1)  # Convert (N, H, W) to (N, C=1, H, W)

    N, C, H, W = images.shape
    images = np.transpose(images, (0, 2, 3, 1))  # Convert from (N, C, H, W) to (N, H, W, C)

    if normalize:
        images = normalize_array(images, value_range=(0, 1), scale_each=False)

    ncol = int(np.ceil(N / nrow))

    grid_height = H * ncol + padding * (ncol + 1)
    grid_width = W * nrow + padding * (nrow + 1)

    # Handle padding value for different image channels
    if isinstance(pad_value, tuple) and len(pad_value) == 3 and C == 3:
        grid = np.full((grid_height, grid_width, C), pad_value, dtype=images.dtype)
    else:
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
    # if the images has channels first we should convert it to channels last
    # check if image is not in the range should be float32 [0,1] so we have to scale it

    if img.ndim == 3 and img.shape[0] in {1, 3}:
        img = np.transpose(img, (1, 2, 0))  # Convert (C, H, W) to (H, W, C)
        cmap = None  # Do not use cmap for RGB images

    if img.dtype != np.uint8:
        img = exposure.rescale_intensity(img, in_range='image', out_range=(0, 1))

    ax.imshow(img, cmap=cmap)
    ax.axis("off")  # Turn off axis


# Function to visualize output images horizontally
def visualise_reconstructed_images(reconstructed_list, title_list, cmap="gray", nrow=5, padding=0,
                                   save_fig: str = None):
    # Create a figure for all visualizations to be displayed horizontally
    fig, axs = plt.subplots(1, len(reconstructed_list), )
    fig.subplots_adjust(wspace=0.01)
    # if ax is non-iterable put it inside a list
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for ax, reconstructed, title in zip(axs, reconstructed_list, title_list):
        reconstructed = np.squeeze(reconstructed)
        # reconstructed = reconstructed.clip(0, 1)
        # Create a grid of images for plotting
        grid = make_grid(reconstructed, nrow=nrow, padding=padding, pad_value=0)
        show_image(ax, grid, cmap=cmap)
        if title:
            ax.set_title(title)

    plt.tight_layout()
    fig_name = save_fig or "reconstructed_images.pdf"
    save_figure(fig_name)
    plt.show()


def plot_reconstruction_comparison(model, pca, images, n_components=None, cmap="viridis", nrow=5, no_title=False,
                                   show_only_reconstruction=False):
    n_components = n_components or pca.n_components

    if n_components > pca.n_components:
        raise ValueError(f"Number of components should be less than or equal to {pca.n_components}")

    latents = model.encode(images)
    ae_reconstructed = model.decode(latents[:, :n_components])
    if ae_reconstructed.ndim != images.ndim:
        ae_reconstructed = ae_reconstructed.reshape(images.shape)
    r_pca = ReducedPCA(pca, n_components)
    # Reconstruct and visualize the images by PCA
    pca_latents = r_pca.transform(images.reshape(images.shape[0], -1))
    pca_reconstructed = r_pca.inverse_transform(pca_latents)
    pca_reconstructed = pca_reconstructed.reshape(images.shape)

    title_list = ["Original", "POLCA-Net", "PCA"] if not no_title else ["", "", ""]
    if show_only_reconstruction:
        reconstructed_list = [ae_reconstructed, pca_reconstructed]
    else:
        reconstructed_list = [images, ae_reconstructed, pca_reconstructed]
    visualise_reconstructed_images(reconstructed_list=reconstructed_list,
                                   title_list=title_list,
                                   nrow=nrow,
                                   cmap=cmap, )


def calculate_metrics(original_images, reconstructed_images):
    """
    Calculate the quality metrics for a set of reconstructed images.

    Parameters:
    - original_images (numpy.ndarray): Array of original images with shape (N, H, W).
    - reconstructed_images (numpy.ndarray): Array of reconstructed images with shape (N,C, H, W).

    Returns:
    - avg_metrics (dict): Dictionary containing the average metrics:
        - 'Normalized Mean Squared Error': Average normalized MSE between original and reconstructed images (lower is better).
        - 'Peak Signal-to-Noise Ratio': Average PSNR between original and reconstructed images (higher is better).
        - 'Structural Similarity Index': Average SSIM between original and reconstructed images (higher is better).
    """
    metrics = {'Normalized Mean Squared Error': [], 'Peak Signal-to-Noise Ratio': [], 'Structural Similarity Index': []}

    for orig, recon in zip(original_images, reconstructed_images):
        nmse = mean_squared_error(orig, recon) / np.mean(np.square(orig))
        psnr = peak_signal_noise_ratio(orig, recon, data_range=1)
        # check if the image has a channel dimension and pass the param to the structural similarity function
        channel_axis = None if recon.ndim == 2 else 0
        ssim = structural_similarity(orig, recon, data_range=1, channel_axis=channel_axis)

        metrics['Normalized Mean Squared Error'].append(nmse)
        metrics['Peak Signal-to-Noise Ratio'].append(psnr)
        metrics['Structural Similarity Index'].append(ssim)

    # Calculate average metrics
    avg_metrics = {'NMSE': np.mean(metrics['Normalized Mean Squared Error']),
                   'PSNR': np.mean(metrics['Peak Signal-to-Noise Ratio']),
                   'SSI': np.mean(metrics['Structural Similarity Index'])}

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
    # 'Normalized Mean Squared Error', 'Peak Signal-to-Noise Ratio', 'Structural Similarity Index'
    metrics_table = metrics_table.round(
        {'NMSE': 4, 'PSNR': 4, 'SSI': 4})

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
    plot_components_cdf(cumulative_variance_ratio, n_components, title, ax)
    fig_name = save_fig or "pca_explained_variance.pdf"
    save_figure(fig_name)

    return pca


def get_pca_torch(x: np.ndarray, n_components=None, ax=None, title="", save_fig=None, device="cuda"):
    total_pca = TorchPCA(device=device)
    total_pca.fit(x)
    n_components = n_components or x.shape[1]
    pca = TorchPCA(n_components=n_components, device=device)
    pca.fit(x)

    # Compute cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(total_pca.explained_variance_ratio_)
    plot_components_cdf(cumulative_variance_ratio, n_components, title, ax)
    fig_name = save_fig or "pca_explained_variance.pdf"
    save_figure(fig_name)

    return pca


def plot_components_cdf(cumulative_variance_ratio, n_components, title="", ax=None):
    # Number of components needed for 90% and 95% explained variance
    total_components = len(cumulative_variance_ratio)
    components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    # If ax is not provided, create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
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
    # set x-axis from 1 to n_components
    # ax.set_xlim(1, n_components)
    # set y-axis from 0 to 1
    ax.set_ylim(0, 1)
    ax.set_box_aspect(2 / 3)


def image_metrics_table(experiment_data: dict, n_components=None):
    tables = []

    for k, (images, model, pca) in experiment_data.items():
        # Reconstruct the images using the autoencoder
        n_comps = n_components or pca.n_components
        if n_comps > pca.n_components:
            raise ValueError(f"Number of components should be less than or equal to {pca.n_components}")

        latents = model.encode(images)
        ae_reconstructed = model.decode(latents[:, :n_comps])
        if ae_reconstructed.ndim != images.ndim:
            ae_reconstructed = ae_reconstructed.reshape(images.shape)
        r_pca = ReducedPCA(pca, n_comps)

        # Reconstruct the images by PCA
        pca_latents = r_pca.transform(images.reshape(images.shape[0], -1))
        pca_reconstructed = r_pca.inverse_transform(pca_latents)
        pca_reconstructed = pca_reconstructed.reshape(images.shape)
        original_images = np.squeeze(images)
        reconstructed_sets = {f"POLCA {k}": ae_reconstructed, f"PCA {k}": pca_reconstructed}

        item = get_images_metrics_table(original_images, reconstructed_sets)
        tables.append(item)

    df_table = pd.concat(tables).set_index("Method")
    display(df_table)
    save_latex_table(df_table, "image_metrics.tex")
    save_df_to_csv(df_table, "image_metrics.csv")
    return df_table


def make_classification_report(model, pca, X, y, n_components=None):
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    n_components = n_components or pca.n_components
    if n_components > pca.n_components:
        raise ValueError(f"Number of components should be less than or equal to {pca.n_components}")

    # Transform the data using PCA
    r_pca = ReducedPCA(pca, n_components)

    if X.ndim > 2:
        X_train_pca = r_pca.transform(X_train.reshape(X_train.shape[0], -1))
        X_test_pca = r_pca.transform(X_test.reshape(X_test.shape[0], -1))
    else:
        X_train_pca = r_pca.transform(X_train)
        X_test_pca = r_pca.transform(X_test)

    print("output shape from pca", X_train_pca.shape, X_test_pca.shape)

    # Transform the data using POLCA-Net
    X_train_polca = model.encode(X_train)[:, :n_components]
    X_test_polca = model.encode(X_test)[:, :n_components]
    print("output shape from POLCA", X_train_polca.shape, X_test_polca.shape)

    # Define classifiers
    classifiers = {"Logistic Regression": LogisticRegression(solver="saga", n_jobs=10),
                   "Gaussian Naive Bayes": GaussianNB(),
                   "Linear SVM": SVC(kernel="linear", probability=False),
                   "Ridge Classifier": RidgeClassifier(),
                   "Perceptron": Perceptron(n_jobs=10), }

    # Train and evaluate classifiers on both PCA and POLCA-Net transformed datasets
    results = []

    for name, clf in classifiers.items():
        # Train on PCA
        clf.fit(minmax_scale(X_train_pca), y_train)
        y_pred_pca = clf.predict(minmax_scale(X_test_pca))
        accuracy_pca = accuracy_score(y_test, y_pred_pca)
        matthews_correlation_pca = matthews_corrcoef(y_test, y_pred_pca)
        report_pca = classification_report(y_test, y_pred_pca, output_dict=True)
        # cm_pca = confusion_matrix(y_test, y_pred_pca)

        # Train on POLCA-Net
        clf.fit(minmax_scale(X_train_polca), y_train)
        y_pred_polca = clf.predict(minmax_scale(X_test_polca))
        accuracy_polca = accuracy_score(y_test, y_pred_polca)
        matthews_correlation_polca = matthews_corrcoef(y_test, y_pred_polca)
        report_polca = classification_report(y_test, y_pred_polca, output_dict=True)
        # cm_polca = confusion_matrix(y_test, y_pred_polca)

        # Append results
        results.append({"Classifier": name,
                        "Transformation": "PCA",
                        "Accuracy": accuracy_pca,
                        "Error rate": (1 - accuracy_pca) * 100,
                        # "Precision": report_pca["weighted avg"]["precision"],
                        # "Recall": report_pca["weighted avg"]["recall"],
                        "Matthews": matthews_correlation_pca,
                        "F1-Score": report_pca["weighted avg"]["f1-score"],
                        # "Confusion Matrix": cm_pca,
                        })

        results.append({"Classifier": name,
                        "Transformation": "POLCA",
                        "Accuracy": accuracy_polca,
                        "Error rate": (1 - accuracy_polca) * 100,
                        # "Precision": report_polca["weighted avg"]["precision"],
                        # "Recall": report_polca["weighted avg"]["recall"],
                        "Matthews": matthews_correlation_polca,
                        "F1-Score": report_polca["weighted avg"]["f1-score"],
                        # "Confusion Matrix": cm_polca,
                        })

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)

    # Display the main metrics table
    main_metrics_df = results_df  # .drop(columns=["Confusion Matrix"])
    main_metrics_df = main_metrics_df.round(2)

    df = main_metrics_df

    # Step 1: Pivot the DataFrame to separate PCA and POLCA-Net results
    df_metrics = df.pivot(index='Classifier', columns='Transformation',
                          values=['Accuracy',
                                  "Error rate",
                                  # 'Precision',
                                  # 'Recall',
                                  'Matthews',
                                  'F1-Score',
                                  ])

    # Step 2: Perform the Wilcoxon Signed-Rank Test and Calculate Median Differences
    comparison_metrics = [
            "Accuracy",
            "Error rate",
            # "Precision",
            # "Recall",
            "Matthews",
            "F1-Score"]
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
        wilcoxon_results[metric] = {'Wilcoxon Test': stat, 'P-Value': p_value,
                                    'Significant (p < 0.05)': f'Yes, {better_method} is better' if p_value < 0.05 else 'No better method'}

    # Convert the results to a DataFrame
    df_wilcoxon = pd.DataFrame(wilcoxon_results).T

    # Display the DataFrames
    print("Performance Metrics DataFrame:")
    display(df_metrics)
    save_latex_table(df_metrics, "classification_metrics.tex")
    save_df_to_csv(df_metrics, "classification_metrics.csv")

    print("\nWilcoxon Signed-Rank Test Results DataFrame:")
    display(df_wilcoxon)
    save_latex_table(df_wilcoxon, "wilcoxon_signed_test.tex")
    return df_metrics, df_wilcoxon


def plot_train_images(x, title="", n=1, cmap="gray", save_fig=None):
    # Plot original and reconstructed signals for a sample
    fig, axes = plt.subplots(1, n)
    fig.subplots_adjust(wspace=0.0)
    im_list = list(range(n))
    for i in im_list:
        # if the images has channels last we should convert it to channels first
        if x[i].ndim == 3 and x[i].shape[0] in {1, 3}:
            _x = np.transpose(x[i], (1, 2, 0))
        else:
            _x = x[i]
        # _x = normalize_array(_x, value_range=(0, 1), scale_each=False)
        axes[i].imshow(_x, cmap=cmap)
        if i == n // 2:
            if title:
                axes[i].set_title(f"{title}")
        axes[i].axis("off")

    fig_name = save_fig or "train_images.pdf"
    save_figure(fig_name)
    plt.show()


def plot2d_analysis(X, y, title, legend=False):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X[:, 0], X[:, 1], label=y, cmap="tab10", c=y, s=10, rasterized=True, alpha=0.75)
    ax.set_xlabel("component: 0")
    ax.set_ylabel("component 1")
    # ax.axis("equal")
    if legend:
        legend1 = plt.legend(*scatter.legend_elements(), title="Classes", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    save_figure(f"{title}_2d_analysis.pdf")
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
        self.experiment_folder = Path.cwd() / f"experiments/{self.experiment_name}"
        self.create_experiment_folder()
        self.add_experiment_info_to_folder()
        self.extra_args = dict()

    def create_experiment_folder(self):
        self.experiment_folder.mkdir(exist_ok=True)

    def get_experiment_folder(self):
        return self.experiment_folder

    def get_name(self, str_name):
        # let the saver decide the extension and must be included in the figure name
        return self.experiment_folder / f"{str_name}"

    def add_experiment_info_to_folder(self):
        info = {"Experiment Name": self.experiment_name, "Experiment Description": self.experiment_description,
                "Random Seed": self.random_seed, "Date and Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Experiment Folder": str(self.experiment_folder), "Python Version": sys.version,
                "PyTorch Version": torch.__version__, "CUDA Version": torch.version.cuda,
                "CUDNN Version": torch.backends.cudnn.version(), "Device": torch.cuda.get_device_name(0),
                # Add the name type or class of the gpu if any
                "GPU Type": torch.cuda.get_device_capability(0), "Number of GPUs": torch.cuda.device_count(),
                "Current GPU": torch.cuda.current_device(), "GPU Available": torch.cuda.is_available(),
                "CPU Cores": os.cpu_count(), "CPU Threads": torch.get_num_threads()}
        with open(self.experiment_folder / "experiment_info.json", "w") as f:
            json.dump(info, f, indent=4)

    def add_extra_args(self, **kwargs):
        # add extra args to the experiment json
        self.extra_args.update(kwargs)
        with open(self.experiment_folder / "experiment_info.json", "r") as f:
            info = json.load(f)
        info.update(self.extra_args)
        with open(self.experiment_folder / "experiment_info.json", "w") as f:
            json.dump(info, f, indent=4)


class TorchPCA(nn.Module):
    def __init__(self, n_components=None, center=True, device=None):
        """
        PCA implementation using PyTorch, with NumPy interface.

        Parameters:
        - n_components (int or None): Number of components to keep.
                                      If None, all components are kept.
        - center (bool): Whether to center the data before applying PCA.
        - device (str or torch.device or None): The device to run computations on. If None, defaults to 'cpu'.
        """
        super(TorchPCA, self).__init__()
        self.n_components = n_components
        self.center = center
        self.device = device if device is not None else torch.device('cpu')
        self.mean = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        """
        Fit the PCA model to X.

        Parameters:
        - X (np.ndarray): The data to fit, of shape (n_samples, n_features).
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.center:
            self.mean = X.mean(dim=0)
            X = X - self.mean

        # Perform PCA using torch.pca_lowrank
        U, S, V = torch.pca_lowrank(X, q=self.n_components)

        self.components_ = V  # Principal components
        explained_variance = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ = explained_variance
        total_variance = explained_variance.sum()
        self.explained_variance_ratio_ = explained_variance / total_variance

        # Convert back to numpy arrays for the external interface
        self.mean = self.mean.cpu().numpy() if self.mean is not None else None
        self.components_ = self.components_.cpu().numpy()
        self.explained_variance_ = self.explained_variance_.cpu().numpy()
        self.explained_variance_ratio_ = self.explained_variance_ratio_.cpu().numpy()

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.

        Parameters:
        - X (np.ndarray): New data to project, of shape (n_samples, n_features).

        Returns:
        - X_transformed (np.ndarray): The data in the principal component space.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.center and self.mean is not None:
            X = X - torch.tensor(self.mean, dtype=torch.float32).to(self.device)
        X_transformed = torch.matmul(X, torch.tensor(self.components_, dtype=torch.float32).to(self.device))
        return X_transformed.cpu().numpy()

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters:
        - X (np.ndarray): The data to fit and transform, of shape (n_samples, n_features).

        Returns:
        - X_transformed (np.ndarray): The data in the principal component space.
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space.

        Parameters:
        - X_transformed (np.ndarray): Data in the principal component space, of shape (n_samples, n_components).

        Returns:
        - X_original (np.ndarray): The data transformed back to the original space.
        """
        X_transformed = torch.tensor(X_transformed, dtype=torch.float32).to(self.device)
        X_original = torch.matmul(X_transformed,
                                  torch.tensor(self.components_, dtype=torch.float32).t().to(self.device))
        if self.center and self.mean is not None:
            X_original = X_original + torch.tensor(self.mean, dtype=torch.float32).to(self.device)
        return X_original.cpu().numpy()


class ReducedPCA:
    def __init__(self, pca, n_components):
        """
        Initialize ReducedPCA with a fitted PCA object and desired number of components.

        :param pca: Fitted PCA object
        :param n_components: Number of components to use (must be <= pca.n_components_)
        """
        if n_components > pca.n_components_:
            raise ValueError("n_components must be <= pca.n_components_")

        self.n_components = n_components
        self.components_ = pca.components_[:n_components]
        self.mean_ = pca.mean_
        self.explained_variance_ = pca.explained_variance_[:n_components]
        self.explained_variance_ratio_ = pca.explained_variance_ratio_[:n_components]

    def transform(self, X):
        """
        Apply dimensionality reduction to X using the reduced number of components.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array-like of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def inverse_transform(self, X_transformed):
        """
        Transform data back to its original space using the reduced number of components.

        :param X_transformed: Array-like of shape (n_samples, n_components)
        :return: Array-like of shape (n_samples, n_features)
        """
        return np.dot(X_transformed, self.components_) + self.mean_


def generate_2d_sinusoidal_data(N, M, num_samples):
    data = []
    for _ in range(num_samples):
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, M)
        xx, yy = np.meshgrid(x, y)

        # Random phase shifts for x and y directions
        phase_shift_x = np.random.uniform(0, 2 * np.pi)
        phase_shift_y = np.random.uniform(0, 2 * np.pi)

        # Random frequency multipliers for x and y directions
        freq_multiplier_x = np.random.uniform(0.5, 1.5)
        freq_multiplier_y = np.random.uniform(0.5, 1.5)

        # Generate sinusoidal data with random phase and frequency
        z = np.sin(2 * np.pi * freq_multiplier_x * xx + phase_shift_x) * np.cos(
            2 * np.pi * freq_multiplier_y * yy + phase_shift_y)
        data.append(z)

    return (1 + np.array(data).astype(np.float32)) / 2.0


def bent_function_image(N, M, a=1, b=1, c=1, d=1):
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    Z = np.cos(2 * np.pi * (a * X + b * Y)) + np.cos(2 * np.pi * (c * X - d * Y))
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min())
    return Z_norm


def generate_bent_images(N, M, num_samples, param_range=(0.1, 5)):
    """
    Generate multiple random bent function images.

    Parameters:
    n_images (int): Number of images to generate
    image_size (tuple): Size of each image as (height, width)
    param_range (tuple): Range for random parameters (min, max)

    Returns:
    numpy.ndarray: Array of shape (n_images, height, width) containing the bent function images
    """

    images = np.zeros((num_samples, N, M))

    for i in range(num_samples):
        # Generate random parameters
        a, b, c, d = np.random.uniform(*param_range, size=4)

        # Generate image
        images[i] = bent_function_image(N, M, a, b, c, d)

    return images


def convert_to_rgb_and_reshape(image_array):
    """
    Convert an array of images to RGB format and reshape it to (image_index, channels, W, H).

    Args:
    - image_array (numpy.ndarray): Array of images with shape (image_index, W, H, channels)

    Returns:
    - numpy.ndarray: Array of images converted to RGB format with shape (image_index, channels, W, H)
    """
    processed_images = []

    for img in image_array:
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img)

        # Convert to RGB if not already
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        # Convert back to numpy array and change shape to (channels, W, H)
        rgb_array = np.array(pil_img).transpose(2, 0, 1)

        # Append the processed image to the list
        processed_images.append(rgb_array)

    # Convert the list of processed images back to a numpy array
    processed_array = np.array(processed_images)

    return processed_array
