import json
import os
import sys
from datetime import datetime
from itertools import batched
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.core.display_functions import clear_output
from IPython.display import display
from PIL import Image
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
from sklearn.preprocessing import StandardScaler

import polcanet.mlutils
from polcanet.mlutils import ReducedPCA, run_classification_pipeline

try:
    import scienceplots

    sp_path = scienceplots.scienceplots_path
    plt.style.use(["science", "no-latex"])
except ImportError:
    scienceplots = None
    sp_path = None
    print("scienceplots style library not found, default matplotlib style will be used.")

# Query the current default figure size
current_fig_size = plt.rcParams["figure.figsize"]
# print(f"Current default figure size: {current_fig_size}")

# Define a scalar factor
scalar_factor = 1.5

# Multiply the current figure size by the scalar factor
new_fig_size = [size * scalar_factor for size in current_fig_size]

# Set the new default figure size
plt.rcParams["figure.figsize"] = new_fig_size

# print(f"New default figure size: {new_fig_size}")

my_style = {"text.usetex": False,
            "figure.autolayout": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
            "font.size": 14,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.titlesize": 14, }
plt.rcParams.update(my_style)

SAVE_PATH = ""
SAVE_FIG = False
SAVE_FIG_PREFIX = ""


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
        name = name.replace(".pdf", f"{get_fig_prefix()}.pdf")
        plt.savefig(get_save_path() / Path(name), dpi=300, bbox_inches="tight")


def save_latex_table(df, name):
    if get_save_fig():
        latex_table = df.reset_index().to_latex(index=False,
                                                escape=False,
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
        name = name.replace(".csv", f"{get_fig_prefix()}.csv")
        if ".csv" not in name:
            name = name + "_" + get_fig_prefix() + ".csv"
        df.to_csv(get_save_path() / Path(name), index=True)


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
def visualise_reconstructed_images(reconstructed_list, title_list, cmap="gray", nrow=5, padding=0, save_fig=None):
    # Create a figure for all visualizations to be displayed horizontally
    fig, axs = plt.subplots(1, len(reconstructed_list), figsize=(len(reconstructed_list) * 3.5, 3.5))
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

    fig_name = save_fig or "reconstructed_images.pdf"
    save_figure(fig_name)
    plt.show()


def plot_reconstruction_comparison(model, pca, images, n_components=None, mask=None, cmap="viridis", nrow=5,
                                   no_title=False,
                                   show_only_reconstruction=False):
    n_components = n_components or pca.n_components

    if n_components > pca.n_components:
        raise ValueError(f"Number of components should be less than or equal to {pca.n_components}")

    latents = model.encode(images)
    ae_reconstructed = model.decode(latents[:, :n_components], mask=mask)
    if ae_reconstructed.ndim != images.ndim:
        ae_reconstructed = ae_reconstructed.reshape(images.shape)
    r_pca = ReducedPCA(pca, n_components)
    # Reconstruct and visualize the images by PCA
    pca_latents = r_pca.transform(images.reshape(images.shape[0], -1))
    pca_reconstructed = r_pca.inverse_transform(pca_latents)
    pca_reconstructed = pca_reconstructed.reshape(images.shape)

    ae_nrmse = normalized_root_mse(images, ae_reconstructed)
    pca_nrmse = normalized_root_mse(images, pca_reconstructed)
    channel_axis = None if images[0].ndim == 2 else 0
    # ae_ssim = structural_similarity(images, ae_reconstructed, data_range=1, channel_axis=channel_axis)
    # pca_ssim = structural_similarity(images, pca_reconstructed, data_range=1, channel_axis=channel_axis)

    title_list = ["Original", f"POLCA-Net rmse: {ae_nrmse:.2f}", f"PCA rmse: {pca_nrmse:.2f}"] if not no_title else [
        "", "", ""]
    if show_only_reconstruction:
        reconstructed_list = [ae_reconstructed, pca_reconstructed]
    else:
        reconstructed_list = [images, ae_reconstructed, pca_reconstructed]
    visualise_reconstructed_images(reconstructed_list=reconstructed_list, title_list=title_list, nrow=nrow, cmap=cmap, )


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
        nmse = normalized_root_mse(orig, recon)
        psnr = peak_signal_noise_ratio(orig, recon, data_range=1)
        # check if the image has a channel dimension and pass the param to the structural similarity function
        channel_axis = None if orig.ndim == 2 else 0
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
    metrics_table = metrics_table.round({'NMSE': 4, 'PSNR': 4, 'SSI': 4})

    # Move the 'Set' column to the first place
    cols = ['Method'] + [col for col in metrics_table.columns if col != 'Method']
    metrics_table = metrics_table[cols]

    return metrics_table


def plot_components_cdf(pca, n_components, title="", save_fig=None):
    # Number of components needed for 90% and 95% explained variance
    # Compute cumulative explained variance ratio
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    total_components = len(cumulative_variance_ratio)
    components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

    fig, ax = plt.subplots(layout='constrained')
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
    ax.set_ylim(0, 1)
    ax.set_box_aspect(2 / 3)
    fig_name = save_fig or "pca_explained_variance.pdf"
    save_figure(fig_name)
    plt.show()


def image_metrics_table(experiment_data: dict, n_components=None, kind=None):
    tables = []

    for k, (images, model, pca) in experiment_data.items():
        # Reconstruct the images using the autoencoder
        n_comps = n_components or pca.n_components
        if n_comps > pca.n_components:
            raise ValueError(f"Number of components should be less than or equal to {pca.n_components}")

        # Transform the data using POLCA-Net but using batch_size since data could be large
        batch_size = min(1024, images.shape[0])
        latents, ae_reconstructed = model.predict_batched(images, batch_size=batch_size, n_components=n_comps)
        if ae_reconstructed.ndim != images.ndim:
            ae_reconstructed = ae_reconstructed.reshape(images.shape)

        # Reconstruct the images by PCA
        r_pca = ReducedPCA(pca, n_comps)
        pca_latents = r_pca.transform(images.reshape(images.shape[0], -1))
        pca_reconstructed = r_pca.inverse_transform(pca_latents)
        pca_reconstructed = pca_reconstructed.reshape(images.shape)
        original_images = np.squeeze(images)
        reconstructed_sets = {f"POLCA {k}": ae_reconstructed, f"PCA {k}": pca_reconstructed}

        item = get_images_metrics_table(original_images, reconstructed_sets)
        tables.append(item)

    df_table = pd.concat(tables).set_index("Method")
    display(df_table)
    prefix = "_" + kind if kind else ""
    save_df_to_csv(df_table, f"image_metrics{prefix}.csv")
    return df_table


def plot_train_images(x, title="", n=1, cmap="gray", save_fig=None):
    # Plot original and reconstructed signals for a sample
    fig, axs = plt.subplots(1, n)
    fig.subplots_adjust(wspace=0.0)

    for i, ax in enumerate(axs):
        _x = np.transpose(x[i], (1, 2, 0)) if x[i].ndim == 3 and x[i].shape[0] in {1, 3} else x[i]
        ax.imshow(_x, cmap=cmap)
        if i == n // 2 and title:
            ax.set_title(title)
        ax.axis("off")

    fig_name = save_fig or "train_images.pdf"
    save_figure(fig_name)
    plt.show()


def plot2d_analysis(X, y, title, c0=0, c1=1, labels_dict=None, legend=False):
    # Loop over each unique class to plot them separately
    y = np.squeeze(y)
    if y.ndim == 1:
        unique_classes = np.unique(y)
    else:
        unique_classes = np.unique(y[:, 0])

    fig, ax = plt.subplots()

    # Use a perceptually uniform colormap for better distinction
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))

    for cls, color in zip(unique_classes, colors):
        ix = np.where(y == cls)
        ax.scatter(X[ix, c0], X[ix, c1], s=5, alpha=0.25, color=color,
                   label=labels_dict[cls] if labels_dict else cls)

    # Add axis labels and title
    ax.set_xlabel(f"Component {c0}")
    ax.set_ylabel(f"Component {c1}")
    ax.set_title(title)
    # Remove top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add legend if requested
    # Add legend with solid color points
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color, markersize=10,
                                  label=labels_dict[cls] if labels_dict else cls)
                       for cls, color in zip(unique_classes, colors)]
    ax.legend(handles=legend_elements, title="Classes", loc='center left',
              bbox_to_anchor=(1, 0.5), frameon=False)

    # Save the figure
    save_figure(f"{title}_2d_analysis.pdf")

    # Show the plot
    plt.show()

    return fig, ax


def plot_class_distribution(y, labels_dict,
                            title: str = "Class Distribution", color_palette: str = "viridis",
                            show_percentages: bool = True, sort_by: str = "value", log_scale: bool = False):
    """
    Plots an improved class distribution for the dataset.

    Parameters:
    - y: Array-like, the target labels for the dataset.
    - labels_dict: Dictionary, mapping from label indices to class names.
    - title: String, the title of the plot.
    - figsize: Tuple, the size of the figure (width, height).
    - color_palette: String, the name of the color palette to use.
    - show_percentages: Boolean, whether to show percentages on bars.
    - sort_by: String, how to sort the bars ('value', 'count', or None for original order).
    - log_scale: Boolean, whether to use a logarithmic scale for the y-axis.
    """
    y = np.squeeze(y)
    # Convert to pandas Series for easier manipulation
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    # Count the number of instances for each class
    class_counts = y.value_counts()
    total_samples = len(y)

    # Convert label indices to class names and calculate percentages
    df = pd.DataFrame({'Class': [labels_dict[label] for label in class_counts.index], 'Count': class_counts.values,
                       'Percentage': (class_counts.values / total_samples) * 100})

    # Sort the dataframe
    if sort_by == 'value':
        df = df.sort_values('Class')
    elif sort_by == 'count':
        df = df.sort_values('Count', ascending=False)

    # Set up the plot
    plt.figure()
    ax = sns.barplot(x='Class', y='Count', hue='Class', data=df, palette=color_palette, edgecolor='black', legend=False)

    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Use log scale if specified
    if log_scale:
        ax.set_yscale('log')
        plt.ylabel('Count (log scale)', fontsize=12)

    # Annotate bars
    for i, row in df.iterrows():
        count = row['Count']
        percentage = row['Percentage']
        ax.text(i, count, f'{count}\n({percentage:.1f}%)', ha='center', va='bottom' if log_scale else 'top',
                fontsize=10, fontweight='bold')

    # Add a legend with total sample count
    plt.text(0.95, 0.95, f'Total Samples: {total_samples}', transform=ax.transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


class ExperimentInfoHandler:
    """
    This class is designed to control the information relative to the current experiment developed in the current file
     script where it is created.
     It will manage a folder name for the experiment which is located on the current working directory.
     Also, it will manage the experiment name, description and the random seed used in the experiment.
     This class will allow to get path names for saving images, generated data, and other files related to the
        experiment, for instance, text files containing latex tables with the results of the experiment.

    """

    def __init__(self, name: str, description: str, random_seed: int, experiment_dir: str = "experiments",
                 root_path=None):
        self.experiment_name = name
        self.experiment_description = description
        self.random_seed = random_seed
        self.root_path = root_path or Path.cwd()
        self.experiment_folder = Path(self.root_path) / "output" / f"{experiment_dir}/{self.experiment_name}"
        self.create_experiment_folder()
        self.add_experiment_info_to_folder()
        self.extra_args = dict()

    def create_experiment_folder(self):
        self.experiment_folder.mkdir(exist_ok=True, parents=True)

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
        self.extra_args.update({k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, tuple, dict))})
        with open(self.experiment_folder / "experiment_info.json", "r") as f:
            info = json.load(f)
        info.update(self.extra_args)
        with open(self.experiment_folder / "experiment_info.json", "w") as f:
            json.dump(info, f, indent=4)


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


class FakeDataset:
    def __init__(self):
        self.data = None
        self.target = None

    def __len__(self):
        return self.data.shape[0]


def bent(root, train=True, download=True, transform=None, n=32, m=32):
    # Generate 2D real bent function images data
    dataset = FakeDataset()
    if train:
        dataset.data = generate_bent_images(n, m, num_samples=2000) * 255

    else:
        dataset.data = generate_bent_images(n, m, num_samples=500) * 255

    dataset.targets = np.zeros(len(dataset.data))
    return dataset


def sinusoidal(root, train=True, download=True, transform=None, n=32, m=32):
    # Generate 2D sinusoidal data
    dataset = FakeDataset()
    if train:
        dataset.data = generate_2d_sinusoidal_data(n, m, num_samples=2000) * 255
    else:
        dataset.data = generate_2d_sinusoidal_data(n, m, num_samples=500) * 255

    dataset.targets = np.zeros(len(dataset.data))
    return dataset


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


def calculate_compressed_size_and_ratio(initial_size: int, compression_rate: float) -> (int, str):
    """
    Calculate the final size after compression and report the compression ratio.

    Parameters:
    initial_size (int): The original size before compression.
    compression_rate (float): The desired final compression rate (as a decimal).
                              For example, 0.5 means the data is compressed to 50% of the original size.

    Returns:
    tuple: A tuple containing the final size after compression (int) and
           the compression ratio as a string in the format "X:1".
    """
    if compression_rate <= 0 or compression_rate > 1:
        raise ValueError("Compression rate must be between 0 (exclusive) and 1 (inclusive).")

    # Calculate the final size after compression
    final_size = initial_size * compression_rate
    compressed_size = int(final_size)

    # Calculate the compression ratio
    compression_ratio = initial_size / compressed_size
    compression_ratio_str = f"{compression_ratio:.1f}:1"

    return compressed_size, compression_ratio_str


def make_classification_report(model, pca, X, y, X_test, y_test, n_components=None, kind=None):
    # Split the dataset into training and testing sets
    X_train, y_train = X, np.squeeze(y)
    X_test, y_test = X_test, np.squeeze(y_test)
    n_components = n_components or pca.n_components
    if n_components > pca.n_components:
        raise ValueError(f"Number of components should be less than or equal to {pca.n_components}")

    # Transform the data using PCA
    r_pca = ReducedPCA(pca, n_components)

    X_train_pca = r_pca.transform(X_train.reshape(X_train.shape[0], -1) if X.ndim > 2 else X_train)
    X_test_pca = r_pca.transform(X_test.reshape(X_test.shape[0], -1) if X_test.ndim > 2 else X_test)

    # Transform the data using POLCA-Net but using batch_size since data could be large
    min_batch_size = 2048  # adjust this value to fit the memory
    batch_size = min(min_batch_size, X_train.shape[0])
    # predict with predict_batched
    X_train_polca = model.predict_batched(X_train, batch_size=batch_size, n_components=n_components)[0]
    X_test_polca = model.predict_batched(X_test, batch_size=batch_size, n_components=n_components)[0]

    # Standardize all data splits using sklearn StandardScaler for each data and method
    pca_scaler = StandardScaler()
    pca_scaler.fit(X_train_pca)
    X_train_pca = pca_scaler.transform(X_train_pca)
    X_test_pca = pca_scaler.transform(X_test_pca)

    polca_scaler = StandardScaler()
    polca_scaler.fit(X_train_polca)
    X_train_polca = polca_scaler.transform(X_train_polca)
    X_test_polca = polca_scaler.transform(X_test_polca)

    # Run the classification pipeline
    results = run_classification_pipeline(X_train_pca, X_test_pca, X_train_polca, X_test_polca, y_train, y_test)
    # Create a DataFrame to display the results
    df = results

    df_metrics = df.pivot(index=['Classifier', "Split"], columns='Transformation',
                          values=['Accuracy',
                                  'Precision',
                                  'Recall',
                                  'F1-Score', ])

    # Display the DataFrames
    print("Classification Performance Metrics DataFrame:")
    display(df_metrics.round(4))
    kind = "_" + kind if kind else ""
    save_df_to_csv(df_metrics, f"classification_metrics{kind}.csv")

    return df_metrics, None


def perform_pca(X, n_components, title="PCA"):
    """Perform PCA on the training dataset."""
    total_pca, pca = polcanet.mlutils.get_pca(X, n_components=n_components)
    plot_components_cdf(total_pca, title=title, n_components=pca.n_components)
    return pca


# Custom float format function
def custom_float_format(x):
    if pd.isnull(x):
        return ""  # Handle NaN
    elif abs(x) >= 1e5 or abs(x) < 1e-5:
        return f"{x:.4e}"  # Scientific notation for large/small numbers
    else:
        return f"{x:.5f}"  # Fixed-point notation for other numbers
