import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


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
def show_image(ax, img):
    ax.imshow(img, cmap="viridis")
    ax.axis("off")  # Turn off axis


# Function to visualize output images horizontally
def visualise_reconstructed_images(reconstructed_list, title_list):
    # Create a figure for all visualizations to be displayed horizontally
    fig, axs = plt.subplots(1, len(reconstructed_list),
                            figsize=(15, 15))  # Adjust number of subplots and size as needed22
    fig.subplots_adjust(wspace=0.01)
    for ax, reconstructed, title in zip(axs, reconstructed_list, title_list):
        reconstructed = np.squeeze(reconstructed)
        # reconstructed = reconstructed.clip(0, 1)
        # Create a grid of images for plotting
        grid = make_grid(reconstructed, nrow=5, padding=0, pad_value=0)
        show_image(ax, grid)
        ax.set_title(title)
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


def display_metrics_table(original_images, reconstructed_sets):
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
