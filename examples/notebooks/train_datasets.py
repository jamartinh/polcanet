# Principal Orthogonal Latent Components Analysis Net (POLCA-Net)
import argparse
import gc
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import medmnist
import numpy as np
import torch
import torchinfo
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10

import polcanet.reports as report
import polcanet.utils as ut
from polcanet import PolcaNet
from polcanet.aencoders import ConvEncoder, LinearDecoder

# Redefine plt.show() to do nothing
# plt.show = lambda: None

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train(params):
    device = params.device
    random_seed = int(params.seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    g = torch.Generator()
    g.manual_seed(random_seed)

    with_labels = params.with_labels
    if with_labels:
        name = f"{params.name}_dataset_labels"
    else:
        name = f"{params.name}_dataset"

    exp = ut.ExperimentInfoHandler(name=name, description=f"POLCA-Net on {name} dataset", random_seed=random_seed)
    ut.set_save_fig(True)
    ut.set_save_path(str(exp.get_experiment_folder()))
    # Print experiment information
    print(f"Experiment configured with seed:{random_seed}, name:{name}")
    print(f"Experiment folder: {exp.get_experiment_folder()}")
    print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

    # ### Load dataset
    train_dataset = params.train_set.data
    eval_dataset = params.test_set.data
    y = params.train_set.targets
    y_test = params.test_set.targets
    labels = np.unique(y) if with_labels else None

    X = np.array(train_dataset, dtype=np.float32)
    X = np.squeeze(X)
    X_test = np.array(eval_dataset, dtype=np.float32)
    X_test = np.squeeze(X_test)

    # print Data information
    print(f"Data range: [{X.min()}, {X.max()}]")
    print(f"Train dataset: {X.shape}, Test dataset: {X_test.shape}")
    print(f"Train labels: {y.shape}, Test labels: {y_test.shape}")
    print(f"Unique labels: {labels}")
    print(f"Number of components: {params.n_components}")

    ut.set_fig_prefix("train")
    ut.plot_train_images(X, f"{params.name} train dataset images", cmap="gray", n=7)
    ut.set_fig_prefix("test")
    ut.plot_train_images(X_test, f"{params.name} test dataset images", cmap="gray", n=7)
    ut.set_fig_prefix("train")

    # ### Fit standard sklearn PCA
    n_components = params.n_components
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
    pca = ut.get_pca(X, n_components=n_components, title=f"PCA on {params.name}", ax=axs, )
    X_pca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))
    X_pca_test = pca.transform(np.squeeze(X_test.reshape(X_test.shape[0], -1)))

    input_dim = X[0].shape
    latent_dim = pca.n_components

    # Create POLCA Net model
    model = create_polca_model(input_dim, latent_dim, labels, params)
    model.to(device)
    batch_size = params.batch_size

    lr_schedule = [1e-3, 1e-4, 1e-5] if params.n_channels == 1 else [1e-4, 1e-4, 1e-5]
    epoch_list = [2000, 2000, 1000]
    if params.use_dataloader:
        x_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.int64) if with_labels else None
        train_data_set =  TensorDataset(x_torch, y_torch) if with_labels else TensorDataset(x_torch)
        train_data_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, generator=g, num_workers=4)
        for lr, epochs in zip(lr_schedule, epoch_list):
            model.train_model(train_data_loader, num_epochs=epochs, report_freq=50, lr=lr)
    else:
        for lr, epochs in zip(lr_schedule, epoch_list):
            model.train_model(data=X, y=y, batch_size=batch_size, num_epochs=epochs, report_freq=50, lr=lr)

    # Evaluate loss interactions
    ut.set_fig_prefix("train")
    model.loss_analyzer.print_report()
    model.loss_analyzer.plot_correlation_matrix(figsize=None)

    # Evaluate results with several reports
    for prefix, data, pca_transform, targets in [("train", X, X_pca, y), ("test", X_test, X_pca_test, y_test)]:
        images = data[0:25]
        ut.set_fig_prefix(prefix)
        report.analyze_reconstruction_error(model, data)
        ut.plot_reconstruction_comparison(model, pca, images, cmap="gray", nrow=5)
        report.orthogonality_test_analysis(model, data)
        report.variance_test_analysis(model, data)
        report.linearity_tests_analysis(model, data, alpha_min=0, num_samples=200)

        experiment_data = {f"{name}": (data, model, pca,), }
        ut.image_metrics_table(experiment_data, n_components=pca.n_components, kind=prefix)

        latents, reconstructed = model.predict(data)
        ut.plot2d_analysis(pca_transform, targets, title="PCA transform", legend=True)
        ut.plot2d_analysis(latents, targets, title="POLCA-Net latent", legend=True)

        # Test Classification with two components on PCA vs POLCA Net
        if len(np.unique(y)) > 1:
            ut.make_classification_report(model, pca, data, targets, n_components=pca.n_components, kind=prefix)

    # Clear up
    del model
    gc.collect()
    torch.cuda.empty_cache()


def create_polca_model(input_dim, latent_dim, labels, params):
    act_fn = torch.nn.SiLU

    encoder = ConvEncoder(input_channels=params.n_channels,
                          latent_dim=latent_dim,
                          conv_dim=2,
                          initial_channels=16 * params.n_channels,
                          growth_factor=2,
                          num_layers=5,
                          act_fn=act_fn,
                          )

    decoder = LinearDecoder(
        latent_dim=latent_dim,
        input_dim=input_dim,
        hidden_dim=5 * 256,
        num_layers=4,
        act_fn=act_fn if not params.linear else torch.nn.Identity,
        bias=False,
    )

    model = PolcaNet(encoder=encoder,
                     decoder=decoder,
                     latent_dim=latent_dim,
                     r=1.0,  # reconstruction weight
                     c=1.0,  # classification weight
                     alpha=1e-2,  # orthogonality loss weight
                     beta=1e-2,  # variance sorting loss weight
                     gamma=1e-2,  # variance reduction loss weight
                     class_labels=labels,  # class labels for supervised in case labels is not None
                     )
    print(model)
    summary = torchinfo.summary(model, (1, *input_dim), dtypes=[torch.float], verbose=1, col_width=16,
                                col_names=["kernel_size", "output_size", "num_params"], row_settings=["var_names"], )
    ut.save_text(str(model), "model.txt")
    ut.save_text(str(summary), "model_summary.txt")
    return model


def train_on_datasets(params):
    for name in params.datasets:
        if name in torch_datasets_dict:
            train_set, test_set = get_dataset(name)
            params.n_channels = train_set.data.shape[1] if train_set.data.ndim == 4 else 1
        else:
            train_set, test_set = get_medmnist_dataset(name)
            params.n_channels = INFO[name]['n_channels']

        print(f"Dataset: {name}\nTrain set: {len(train_set)}\nTest set: {len(test_set)}")
        print(f"Data shape, train: {train_set.data.shape}, test: {test_set.data.shape}")

        params.name = name
        params.train_set = train_set
        params.test_set = test_set
        image_height = train_set.data.shape[-1]

        if params.n_components is None:
            params.n_components = (image_height * params.n_channels * params.n_components_multiplier) // 2
            params.n_components = max(int(params.n_components), 8)

        if params.batch_size is None:
            params.batch_size = 512 if train_set.data.shape[0] >= 50000 else 256

        train(params)
        gc.collect()


def get_dataset(name):
    print("Loading dataset: ", name)
    dataset = datasets_dict[name]
    os.makedirs(Path(f"data/{name}").absolute(), exist_ok=True)
    train_set = dataset(root=f"data/{name}", train=True, download=True)
    test_set = dataset(root=f"data/{name}", train=False, download=True)

    # convert to numpy
    if not isinstance(train_set.data, np.ndarray):
        train_set.data = train_set.data.numpy()
        test_set.data = test_set.data.numpy()

    if not isinstance(train_set.targets, (np.ndarray, list)):
        train_set.targets = train_set.targets.numpy()
        test_set.targets = test_set.targets.numpy()

    train_set.targets = np.array(train_set.targets)
    test_set.targets = np.array(test_set.targets)

    # adjust the data shape for channels first
    if train_set.data.ndim == 4:
        train_set.data = np.moveaxis(train_set.data, -1, 1)
        test_set.data = np.moveaxis(test_set.data, -1, 1)

    train_set.data = np.squeeze(train_set.data)
    test_set.data = np.squeeze(test_set.data)

    # normalize the data
    train_set.data = train_set.data / 255.0
    test_set.data = test_set.data / 255.0

    return train_set, test_set


def get_medmnist_dataset(name, size=28):
    print("Loading dataset: ", name)
    dataset = datasets_dict[name]
    os.makedirs(Path(f"data/{name}").absolute(), exist_ok=True)
    train_set = dataset(root=Path(f"data/{name}").absolute(), split='train', download=True, size=size)
    test_set = dataset(root=Path(f"data/{name}").absolute(), split='test', download=True, size=size)
    print(train_set)

    # normalize the dataset format
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

    # normalize the data pixel values to [0, 1]
    train_set.data = train_set.data / 255.0
    test_set.data = test_set.data / 255.0

    return train_set, test_set


torch_datasets_dict = {"mnist": MNIST,
                       "fmnist": FashionMNIST,
                       "cifar10": CIFAR10,
                       "bent": ut.bent,
                       "sinusoidal": ut.sinusoidal,
                       }

medmnist_datasets_dict = {"breastmnist": medmnist.BreastMNIST,
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

if __name__ == "__main__":
    # ### Load dataset

    print(f"Available datasets: {torch_datasets_dict.keys()}")
    print(f"Available medmnist datasets: {INFO.keys()}")

    # define console parameters
    args = argparse.ArgumentParser()
    args.add_argument("--datasets", type=str, nargs="+", help="datasets to train the model on")
    args.add_argument("--device", type=str, default="cuda", help="device to train the model on")
    args.add_argument("--seed", type=int, default=5, help="random seed")
    args.add_argument("--with_labels", action="store_true", default=False, help="train with labels (LDA like)")
    args.add_argument("--n_components", type=int, default=None, help="number of components")
    args.add_argument("--n_components_multiplier", type=int, default=1, help="number of components multiplier")
    args.add_argument("--batch_size", type=int, default=None, help="batch size")
    args.add_argument("--linear", action="store_true", default=False, help="use linear decoder")
    args.add_argument("--list", action="store_true", default=False, help="list available datasets")
    args.add_argument("--use_dataloader", action="store_true", default=False,
                      help="use DataLoaders instead on in memory data (in memory data is default)")
    parameters = args.parse_args()

    if parameters.list:
        for key, value in datasets_dict.items():
            print(key)

    for dataset_name in parameters.datasets:
        assert dataset_name in (medmnist_datasets_dict | torch_datasets_dict), f"dataset not available: {dataset_name}"
        # make a small dict with name, and class with the dataset name in parameters
        train_on_datasets(parameters)

    print("Done")
