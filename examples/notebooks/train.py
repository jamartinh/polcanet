# Principal Orthogonal Latent Components Analysis Net (POLCA-Net)
import argparse
import gc
import os
import random
from pathlib import Path

import medmnist
import numpy as np
import torch
import torchinfo
from medmnist import INFO
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST

import polcanet.mlutils
import polcanet.reports as report
import polcanet.utils
import polcanet.utils as ut
from polcanet import PolcaNet
from polcanet.aencoders import ConvEncoder, LinearDecoder
from polcanet.utils import perform_pca

# Redefine plt.show() to do nothing
# plt.show = lambda: None

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train(params):
    """Train the POLCA-Net model on the given dataset."""
    setup_environment(params)

    exp = setup_experiment(params)

    X, X_test, y, y_test, labels = prepare_data(params)

    batch_size, epoch_list, lr_schedule = process_parameters_and_defaults(X, params)

    pca = perform_pca(X, n_components=params.n_components, title=f"PCA on {params.name} dataset")

    model = create_polca_model(X[0].shape, pca.n_components, labels, params)
    model.to(params.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_schedule[0], weight_decay=1e-1)

    if params.use_dataloader:
        train_with_dataloader(X, y, model, batch_size, lr_schedule, epoch_list, params.params, optimizer=optimizer)
    else:
        train_in_memory(X, y, model, batch_size, lr_schedule, epoch_list, optimizer=optimizer)

    analyze_and_report_results(X, X_test, y, y_test, model, pca, params)

    cleanup_resources(model)


def setup_environment(params):
    """Set up the environment and seeds for reproducibility."""
    device = params.device
    random_seed = int(params.seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)
    params.device = device
    params.seed = random_seed
    params.generator = g


def setup_experiment(params):
    """Configure the experiment and logging."""
    name_suffix = "_dataset_labels" if params.with_labels else "_dataset"
    name = f"{params.name}{name_suffix}"
    exp = ut.ExperimentInfoHandler(name=name,
                                   description=f"POLCA-Net on {name} dataset",
                                   random_seed=params.seed,
                                   experiment_dir=params.exp_dir)

    ut.set_save_fig(True)
    ut.set_save_path(str(exp.get_experiment_folder()))

    print(f"Experiment configured with seed: {params.seed}, name: {name}")
    print(f"Experiment folder: {exp.get_experiment_folder()}")
    print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

    return exp


def prepare_data(params):
    """Prepare and preprocess the dataset."""
    X = np.array(params.train_set.data, dtype=np.float32).squeeze()
    X_test = np.array(params.test_set.data, dtype=np.float32).squeeze()
    y = params.train_set.targets
    y_test = params.test_set.targets
    labels = list(params.labels_dict.keys()) if params.with_labels and params.labels_dict else None

    print(f"Data range: [{X.min()}, {X.max()}]")
    print(f"Train dataset: {X.shape}, Test dataset: {X_test.shape}")
    print(f"Train targets: {y.shape}, Test labels: {y_test.shape}")
    print(f"Unique targets: {labels}")
    print(f"Class labels: {params.labels_dict}")
    if len(np.unique(y)) > 1 and np.squeeze(y).ndim == 1:
        #  calculate and print information on class distribution and balance
        print("Class balance:")
        for label, name in params.labels_dict.items():
            count = np.sum(y == label)
            print(f"{name}: {count / len(y) * 100:.2f}%")

    print(f"Number of components: {params.n_components}")

    ut.set_fig_prefix("train")
    ut.plot_train_images(X, f"{params.name} train dataset images", cmap="gray", n=7)
    ut.set_fig_prefix("test")
    ut.plot_train_images(X_test, f"{params.name} test dataset images", cmap="gray", n=7)
    ut.set_fig_prefix("train")

    return X, X_test, y, y_test, labels


def process_parameters_and_defaults(dataset, params):
    """Process and set default parameters for training."""
    batch_size = params.batch_size or (512 if dataset.shape[0] >= 50000 else 256)
    dataset_size = dataset.shape[0]
    iterations_per_epoch = (dataset_size + batch_size - 1) // batch_size
    num_epochs = (params.n_updates + iterations_per_epoch - 1) // iterations_per_epoch

    image_height = dataset.shape[-1]
    params.n_components = params.n_components or max(
        (image_height * params.n_channels * params.n_components_multiplier) // 2, 8)

    lr_schedule = [1e-3, 1e-4, 1e-5] if params.n_channels == 1 else [1e-4, 1e-4, 1e-5]
    epoch_list = [num_epochs, num_epochs, num_epochs // 2] if params.epochs is None else [params.epochs] * 3

    return batch_size, epoch_list, lr_schedule


def create_polca_model(input_dim, latent_dim, labels, params):
    """Create and return the POLCA-Net model."""
    encoder = ConvEncoder(
        input_channels=params.n_channels,
        latent_dim=latent_dim,
        conv_dim=2,
        initial_channels=16 * params.n_channels,
        growth_factor=2,
        num_layers=5,
        act_fn=torch.nn.GELU,
        size=max(max(input_dim), 28),
    )

    decoder = LinearDecoder(
        latent_dim=latent_dim,
        input_dim=input_dim,
        hidden_dim=4 * 256 if not params.linear else 10 * 256,
        num_layers=4 if not params.linear else 2,
        act_fn=torch.nn.GELU if not params.linear else None,
        bias=False,
        output_act_fn=None,
    )

    r = 1.0
    c = 0.0
    alpha = 1e-2
    beta = 1e-2
    gamma = 1e-6
    delta = 0
    if params.with_labels:
        r = 1.0
        c = 1e-2
        alpha = 1
        beta = 1
        gamma = 1e-6
        delta = 0

    model = PolcaNet(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        r=r,  # reconstruction weight
        c=c,  # classification weight
        alpha=alpha,  # orthogonality loss weight
        beta=beta,  # variance sorting loss weight
        gamma=gamma,  # variance reduction loss weight
        delta=delta,  # instance orthogonality loss weight  *experimental*
        class_labels=labels,  # class labels for supervised in case labels is not None
    )

    print(model)
    summary = torchinfo.summary(
        model, (1, *input_dim), dtypes=[torch.float], verbose=1,
        col_width=16, col_names=["kernel_size", "output_size", "num_params"], row_settings=["var_names"]
    )

    ut.save_text(str(model), "model.txt")
    ut.save_text(str(summary), "model_summary.txt")

    return model


def train_with_dataloader(X, y, model, batch_size, lr_schedule, epoch_list, params, optimizer):
    """Train the model using DataLoader."""
    x_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.int64) if params.with_labels else None
    train_data_set = TensorDataset(x_torch, y_torch) if params.with_labels else TensorDataset(x_torch)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, generator=params.generator,
                                   num_workers=4, drop_last=True)

    for lr, epochs in zip(lr_schedule, epoch_list):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01
        model.train_model(train_data_loader, num_epochs=epochs, lr=lr, optimizer=optimizer)


def train_in_memory(X, y, model, batch_size, lr_schedule, epoch_list, optimizer):
    """Train the model using in-memory data."""
    for lr, epochs in zip(lr_schedule, epoch_list):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train_model(data=X, y=y, batch_size=batch_size, num_epochs=epochs, lr=lr, optimizer=optimizer)


def analyze_and_report_results(X, X_test, y, y_test, model, pca, params):
    """Analyze and report the results of the training."""

    ut.set_fig_prefix("")
    model.loss_analyzer.print_report()
    model.loss_analyzer.plot_correlation_matrix()

    for prefix, data, targets in [("train", X, y), ("test", X_test, y_test)]:
        ut.set_fig_prefix(prefix)
        # report.analyze_reconstruction_error(model, data)
        images = data[:25]
        ut.plot_reconstruction_comparison(model, pca, images, cmap="gray", nrow=5)
        report.orthogonality_test_analysis(model, pca, data)
        report.variance_test_analysis(model, data)
        report.linearity_tests_analysis(model, data, alpha_min=0, num_samples=200)

        experiment_data = {f"{params.name}": (data, model, pca)}
        ut.image_metrics_table(experiment_data, n_components=pca.n_components)

        report.embedding_analysis(model, pca, data, targets, params.labels_dict)

    if len(np.unique(y)) > 1:
        polcanet.utils.make_classification_report(model, pca, X, y, X_test, y_test, n_components=pca.n_components)


def cleanup_resources(model):
    """Clean up resources after training."""
    del model
    gc.collect()
    torch.cuda.empty_cache()


def get_dataset(name):
    """Load and preprocess a dataset from torchvision."""
    print(f"Loading dataset: {name}")
    dataset = datasets_dict[name]
    os.makedirs(Path(f"data/{name}").absolute(), exist_ok=True)
    train_set = dataset(root=f"data/{name}", train=True, download=True)
    test_set = dataset(root=f"data/{name}", train=False, download=True)

    # Convert to numpy
    if not isinstance(train_set.data, np.ndarray):
        train_set.data = train_set.data.numpy()
        test_set.data = test_set.data.numpy()

    if not isinstance(train_set.targets, (np.ndarray, list)):
        train_set.targets = train_set.targets.numpy()
        test_set.targets = test_set.targets.numpy()

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    # Adjust the data shape for channels first
    if train_set.data.ndim == 4:
        train_set.data = np.moveaxis(train_set.data, -1, 1)
        test_set.data = np.moveaxis(test_set.data, -1, 1)

    train_set.data = np.squeeze(train_set.data)
    test_set.data = np.squeeze(test_set.data)

    # Normalize the data
    train_set.data = train_set.data / 255.0
    test_set.data = test_set.data / 255.0

    return train_set, test_set


def get_medmnist_dataset(name, size=28):
    """Load and preprocess a dataset from medmnist."""
    print(f"Loading dataset: {name}")
    dataset = datasets_dict[name]
    os.makedirs(Path(f"data/{name}").absolute(), exist_ok=True)
    train_set = dataset(root=Path(f"data/{name}").absolute(), split='train', download=True, size=size)
    test_set = dataset(root=Path(f"data/{name}").absolute(), split='test', download=True, size=size)
    print(train_set)

    # Normalize the dataset format
    train_set.data = train_set.imgs
    test_set.data = test_set.imgs

    train_set.targets = np.array(train_set.labels)
    test_set.targets = np.array(test_set.labels)
    n_channels = 3 if 3 in set(train_set.data.shape) else 1

    if n_channels == 3:
        print("Converting to RGB")
        train_set.data = ut.convert_to_rgb_and_reshape(train_set.data)
        test_set.data = ut.convert_to_rgb_and_reshape(test_set.data)
        print(train_set.data.shape, test_set.data.shape)

    # Normalize the data pixel values to [0, 1]
    train_set.data = train_set.data / 255.0
    test_set.data = test_set.data / 255.0

    return train_set, test_set


def train_on_datasets(params):
    """Train the model on multiple datasets."""
    if not params.datasets:
        raise ValueError("No datasets provided")

    for name in params.datasets:
        if name in torch_datasets_dict:
            train_set, test_set = get_dataset(name)
            params.n_channels = train_set.data.shape[1] if train_set.data.ndim == 4 else 1
        else:
            train_set, test_set = get_medmnist_dataset(name, size=params.size)
            params.n_channels = INFO[name]['n_channels']

        print(f"Dataset: {name}\nTrain set: {len(train_set)}\nTest set: {len(test_set)}")
        print(f"Data shape, train: {train_set.data.shape}, test: {test_set.data.shape}")

        params.name = name
        params.train_set = train_set
        params.test_set = test_set
        params.labels_dict = dataset_dict_labels[name]
        if not params.labels_dict:
            params.with_labels = False

        train(params)
        gc.collect()


torch_datasets_dict = {
        "mnist": MNIST,
        "fmnist": FashionMNIST,
        "bent": ut.bent,
        "sinusoidal": ut.sinusoidal,
}

torch_dataset_dict_labels = {
        "mnist": {i: str(i) for i in range(10)},
        "fmnist": {
                0: "T-shirt/top",
                1: "Trouser",
                2: "Pullover",
                3: "Dress",
                4: "Coat",
                5: "Sandal",
                6: "Shirt",
                7: "Sneaker",
                8: "Bag",
                9: "Ankle boot", },
        "cifar10": {
                0: "airplane",
                1: "automobile",
                2: "bird",
                3: "cat",
                4: "deer",
                5: "dog",
                6: "frog",
                7: "horse",
                8: "ship",
                9: "truck", },
        "bent": None,
        "sinusoidal": None,
}

medmnist_datasets_dict = {
        "breastmnist": medmnist.BreastMNIST,
        "dermamnist": medmnist.DermaMNIST,
        "octmnist": medmnist.OCTMNIST,
        "organamnist": medmnist.OrganAMNIST,
        "organcmnist": medmnist.OrganCMNIST,
        "organsmnist": medmnist.OrganSMNIST,
        "pathmnist": medmnist.PathMNIST,
        "pneumoniamnist": medmnist.PneumoniaMNIST,
        "retinamnist": medmnist.RetinaMNIST,
        "bloodmnist": medmnist.BloodMNIST,
        "chestmnist": medmnist.ChestMNIST,
}

datasets_dict = {**torch_datasets_dict, **medmnist_datasets_dict}
dataset_dict_labels = dict()
dataset_dict_labels.update(torch_dataset_dict_labels)
for name in medmnist_datasets_dict:
    dataset_dict_labels[name] = {int(k): v for k, v in INFO[name]['label'].items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=[], nargs="+", help="datasets to train the model on")
    parser.add_argument("--epochs", type=int, default=None, help="number of epochs")
    parser.add_argument("--n_updates", type=int, default=20000, help="number of gradient updates")
    parser.add_argument("--device", type=str, default="cuda", help="device to train the model on")
    parser.add_argument("--seed", type=int, default=5, help="random seed")
    parser.add_argument("--with_labels", action="store_true", default=False, help="train with labels (LDA like)")
    parser.add_argument("--n_components", type=float, default=None, help="number or fraction of components")
    parser.add_argument("--n_components_multiplier", type=float, default=1, help="number of components multiplier")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--linear", action="store_true", default=False, help="use linear decoder")
    parser.add_argument("--list", action="store_true", default=False, help="list available datasets")
    parser.add_argument("--use_dataloader", action="store_true", default=False,
                        help="use DataLoaders instead of in-memory data")
    parser.add_argument("--size", type=int, default=28, choices=[28, 64, 128, 224],
                        help="MedMNIST image size")
    parser.add_argument("--exp_dir", type=str, default="experiments", help="experiments top directory")

    parameters = parser.parse_args()

    if parameters.list:
        for key in datasets_dict:
            print(key)

    for dataset_name in parameters.datasets:
        if dataset_name not in datasets_dict:
            raise ValueError(f"Dataset not available: {dataset_name}")

        train_on_datasets(parameters)
