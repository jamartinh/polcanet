# Principal Orthogonal Latent Components Analysis Net (POLCA-Net)
import argparse
import gc
import os
import random
from pathlib import Path
from pprint import pprint

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torchinfo
from matplotlib import pyplot as plt
from medmnist import INFO
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST

import polcanet.mlutils
import polcanet.reports as report
import polcanet.utils
import polcanet.utils as ut
from polcanet import PolcaNet
from polcanet.aencoders import ConvEncoder, LinearDecoder, LSTMDecoder
from polcanet.utils import perform_pca

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.backends.cudnn.benchmark = True


def in_jupyterlab():
    try:
        from IPython import get_ipython
        from ipywidgets.widgets import Output, Accordion
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        # Redefine plt.show() to do nothing
        plt.show = lambda: None
        return False


def train(params):
    """Train the POLCA-Net model on the given dataset."""
    # Create an accordion for the experiment with three parts: preamble, training, train, test each one with its respective output

    setup_environment(params)
    exp = setup_experiment(params)
    X, X_test, y, y_test, labels = prepare_data(params, exp)
    batch_size, epoch_list, lr_schedule = process_parameters_and_defaults(X, params)

    pca = perform_pca(X, n_components=params.n_components, title=f"PCA on {params.name} dataset")
    # Add more experiment data to Experiment Tracking

    exp_vars = {
            "pca_n_components": pca.n_components,
            "batch_size": batch_size,
            "epoch_list": tuple(epoch_list),
            "lr_schedule": tuple(lr_schedule),

    }
    exp.add_extra_args(**exp_vars)

    model = create_polca_model(experiment=exp, input_dim=X[0].shape, latent_dim=pca.n_components, labels=labels,
                               params=params)
    model.to(params.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_schedule[0], weight_decay=1e-2, betas=(0.9, 0.99), eps=1e-6)

    if params.use_dataloader:
        train_with_dataloader(X, y, model, batch_size, lr_schedule, epoch_list, params.params, optimizer=optimizer,
                              X_val=X_test, y_val=y_test)
    else:
        train_in_memory(X, y, model, batch_size, lr_schedule, epoch_list, optimizer=optimizer, X_val=X_test,
                        y_val=y_test)

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

    exp.add_extra_args(**vars(params))
    ut.set_save_fig(True)
    ut.set_save_path(str(exp.get_experiment_folder()))

    print(f"Experiment configured with seed: {params.seed}, name: {name}")
    print(f"Experiment folder: {exp.get_experiment_folder()}")
    print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

    return exp


def prepare_data(params, exp):
    """Prepare and preprocess the dataset."""
    X = np.array(params.train_set.data, dtype=np.float32).squeeze()
    X_test = np.array(params.test_set.data, dtype=np.float32).squeeze()
    y = params.train_set.targets
    y_test = params.test_set.targets
    labels = list(params.labels_dict.keys()) if params.with_labels and params.labels_dict else None

    pprint(f"Data range: [{X.min()}, {X.max()}]")
    pprint(f"Train dataset: {X.shape}, Test dataset: {X_test.shape}")
    pprint(f"Train targets: {y.shape}, Test labels: {y_test.shape}")
    pprint(f"Unique targets: {labels}")
    pprint(f"Class labels: {params.labels_dict}")
    if len(np.unique(y)) > 1 and np.squeeze(y).ndim == 1:
        #  calculate and print information on class distribution and balance
        pprint("Class balance:")
        for label, name in params.labels_dict.items():
            count = np.sum(y == label)
            pprint(f"{name}: {count / len(y) * 100:.2f}%")

    print(f"Number of components: {params.n_components}")

    # Add all the information to a dictionary and save it to the experiment using add_extra_args
    data_dict_exp = {
            "Data range": (float(X.min()), float(X.max())),
            "Train dataset shape": tuple((int(e) for e in X.shape)),
            "Test dataset shape": tuple((int(e) for e in X_test.shape)),
            "Train targets shape": tuple((int(e) for e in y.shape)),
            "Test targets shape": tuple((int(e) for e in y_test.shape)),
            "Unique targets": tuple(labels or (None,)),
    }
    exp.add_extra_args(**data_dict_exp)

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

    lr_schedule = (1e-3, 1e-4, 1e-5) if params.n_channels == 1 else (1e-4, 1e-4, 1e-5)
    if params.with_labels:
        lr_schedule = [1e-3] if params.n_channels == 1 else [1e-4]
    epoch_list = [num_epochs] * 3 if params.epochs is None else [params.epochs] * 3

    return batch_size, epoch_list, lr_schedule


def create_polca_model(experiment, input_dim, latent_dim, labels, params) -> PolcaNet:
    """Create and return the POLCA-Net model."""
    act_fn = nn.SiLU
    min_image_size = 28
    current_image_size = max(max(input_dim), min_image_size)
    encoder = ConvEncoder(
        input_channels=params.n_channels,
        latent_dim=latent_dim,
        conv_dim=2,
        initial_channels=32,
        growth_factor=2,
        num_layers=6,
        act_fn=act_fn,
        size=current_image_size,
    )

    # decoder = LinearDecoder(
    #     latent_dim=latent_dim,
    #     input_dim=input_dim,
    #     hidden_dim=12 * 256 if not params.linear else 15 * 256,
    #     num_layers=3 if not params.linear else 1,
    #     act_fn=nn.GELU if not params.linear else torch.nn.Identity,
    #     bias=False,
    # )

    decoder = LSTMDecoder(
        latent_dim=latent_dim,
        hidden_size=1024,
        output_dim=input_dim,
        num_layers=1,
        proj_size=None,
        bias=False,
    )

    r = params.r if params.r is not None else 1
    c = params.c if params.c is not None else 0
    alpha = params.alpha if params.alpha is not None else 1e-1
    beta = params.beta if params.beta is not None else 1e-1
    gamma = params.gamma if params.gamma is not None else 0
    std_noise = 0.0

    if params.with_labels:
        r = params.r if params.r is not None else 1
        c = params.c if params.c is not None else 1
        alpha = params.alpha if params.alpha is not None else 1e-1
        beta = params.beta if params.beta is not None else 1e-1
        gamma = params.gamma if params.gamma is not None else 0
        std_noise = 0.0

    extra_args = {
            "r": r,
            "c": c,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "std_noise": std_noise
    }
    experiment.add_extra_args(**extra_args)

    model = PolcaNet(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        r=r,  # reconstruction weight
        c=c,  # classification weight
        alpha=alpha,  # orthogonality loss weight
        beta=beta,  # variance sorting loss weight
        gamma=gamma,  # variance reduction loss weight
        class_labels=labels,  # class labels for supervised in case labels is not None
        std_noise=std_noise  # standard deviation of the noise
    )

    pprint(model)
    summary = torchinfo.summary(
        model, (1, *input_dim),
        dtypes=[torch.float],
        depth=1,
        verbose=1,
        col_width=16,
        col_names=["num_params"],
        row_settings=["var_names"],
    )
    pprint(summary)
    ut.save_text(str(model), "model.txt")
    ut.save_text(str(summary), "model_summary.txt")

    return model


def train_with_dataloader(X, y, model, batch_size, lr_schedule, epoch_list, params, optimizer, X_val=None, y_val=None):
    """Train the model using DataLoader."""
    x_torch = torch.tensor(X, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.int64) if params.with_labels else None
    train_data_set = TensorDataset(x_torch, y_torch) if params.with_labels else TensorDataset(x_torch)
    train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, generator=params.generator,
                                   num_workers=4, drop_last=True, pin_memory=True)

    for lr, epochs in zip(lr_schedule, epoch_list):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train_model(data=train_data_loader,
                          num_epochs=epochs,
                          lr=lr,
                          optimizer=optimizer,
                          val_data=X_val,
                          val_y=y_val,
                          )


def train_in_memory(X, y, model, batch_size, lr_schedule, epoch_list, optimizer, X_val=None, y_val=None):
    """Train the model using in-memory data."""
    for lr, epochs in zip(lr_schedule, epoch_list):
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        model.train_model(data=X,
                          y=y,
                          batch_size=batch_size,
                          num_epochs=epochs,
                          lr=lr, optimizer=optimizer,
                          val_data=X_val,
                          val_y=y_val,
                          )


def analyze_and_report_results(X, X_test, y, y_test, model, pca, params):
    """Analyze and report the results of the training."""

    report.loss_interaction_analysis(model)
    for prefix, data, targets in [("train", X, y), ("test", X_test, y_test)]:
        ut.set_fig_prefix(prefix)
        # report.analyze_reconstruction_error(model, data)
        images = data[:9]
        ut.plot_reconstruction_comparison(model, pca, images, cmap="gray", nrow=3)
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
    gc.collect()


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
    # print(train_set)

    # Normalize the dataset format
    train_set.data = train_set.imgs
    test_set.data = test_set.imgs

    train_set.targets = np.array(train_set.labels)
    test_set.targets = np.array(test_set.labels)
    n_channels = 3 if 3 in set(train_set.data.shape) else 1

    if n_channels == 3:
        # print("Converting to RGB")
        train_set.data = ut.convert_to_rgb_and_reshape(train_set.data)
        test_set.data = ut.convert_to_rgb_and_reshape(test_set.data)
        # print(train_set.data.shape, test_set.data.shape)

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

        # print(f"Dataset: {name}\nTrain set: {len(train_set)}\nTest set: {len(test_set)}")
        # print(f"Data shape, train: {train_set.data.shape}, test: {test_set.data.shape}")

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
        "tissuemnist": medmnist.TissueMNIST,
}

datasets_dict = {**torch_datasets_dict, **medmnist_datasets_dict}
dataset_dict_labels = dict()
dataset_dict_labels.update(torch_dataset_dict_labels)
for name in medmnist_datasets_dict:
    dataset_dict_labels[name] = {int(k): v for k, v in INFO[name]['label'].items()}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default=[], choices=list(dataset_dict_labels), nargs="+",
                        help="datasets to train the model on")
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
    # Add parameters c, r, alpha, beta , gamma for model creation
    parser.add_argument("--c", type=float, default=None, help="classification weight")
    parser.add_argument("--r", type=float, default=None, help="reconstruction weight")
    parser.add_argument("--alpha", type=float, default=None, help="orthogonality loss weight")
    parser.add_argument("--beta", type=float, default=None, help="variance sorting loss weight")
    parser.add_argument("--gamma", type=float, default=None, help="variance reduction loss weight")

    parameters = parser.parse_args()

    if parameters.list:
        for key in datasets_dict:
            print(key)

    for dataset_name in parameters.datasets:
        if dataset_name not in datasets_dict:
            raise ValueError(f"Dataset not available: {dataset_name}")

        train_on_datasets(parameters)
