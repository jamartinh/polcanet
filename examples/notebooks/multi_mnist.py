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
from torchvision.datasets import MNIST, FashionMNIST

import polcanet.reports as report
import polcanet.utils as ut
from polcanet import PolcaNet
from polcanet.aencoders import ConvEncoder, LinearDecoder

# Redefine plt.show() to do nothing
# plt.show = lambda: None

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train_x_mnist(params):
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
    print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

    # ### Load dataset
    train_dataset = params.train_set.data
    eval_dataset = params.test_set.data
    y = params.train_set.targets
    y_test = params.test_set.targets
    labels = np.unique(y) if with_labels else None

    X = np.array(train_dataset, dtype=np.float32)
    X = np.squeeze(X)
    print(X.min(), X.max())
    X_test = np.array(eval_dataset, dtype=np.float32)
    X_test = np.squeeze(X_test)
    print(train_dataset.shape, eval_dataset.shape, X.shape, X_test.shape, y.shape, y_test.shape, labels)

    ut.set_fig_prefix("train")
    ut.plot_train_images(X, f"{params.name} train dataset images", cmap="gray", n=7)
    ut.set_fig_prefix("test")
    ut.plot_train_images(X_test, f"{params.name} test dataset images", cmap="gray", n=7)

    # ### Fit standard sklearn PCA
    n_components = params.n_components
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
    pca = ut.get_pca(X, n_components=n_components, title=f"PCA on {params.name}", ax=axs, )
    Xpca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))

    # ### Fit POLCANet
    act_fn = torch.nn.SiLU
    input_dim = X[0].shape
    latent_dim = pca.n_components

    encoder = ConvEncoder(input_channels=params.n_channels,
                          latent_dim=latent_dim,
                          conv_dim=2,
                          initial_channels=8,
                          growth_factor=2,
                          num_layers=5,
                          act_fn=act_fn,
                          )

    if params.linear:
        decoder = LinearDecoder(latent_dim=latent_dim,
                                input_dim=input_dim,
                                hidden_dim=1024,
                                num_layers=2,
                                bias=False,
                                )
    else:
        decoder = LinearDecoder(latent_dim=latent_dim,
                                input_dim=input_dim,
                                hidden_dim=512 * params.n_channels,
                                num_layers=5,
                                act_fn=act_fn,
                                bias=False,
                                )

    model = PolcaNet(encoder=encoder,
                     decoder=decoder,
                     latent_dim=latent_dim,
                     alpha=0.1,  # orthogonality loss
                     beta=0.01,  # variance sorting loss
                     gamma=0.1,  # variance reduction loss
                     class_labels=labels,  # class labels for supervised in case labels is not None
                     )

    print(model)
    summary = torchinfo.summary(model, (1, *input_dim), dtypes=[torch.float], verbose=1, col_width=16,
                                col_names=["kernel_size", "output_size", "num_params"], row_settings=["var_names"], )
    ut.save_text(str(model), "model.txt")
    ut.save_text(str(summary), "model_summary.txt")

    model.to(device)
    batch_size = params.batch_size
    model.train_model(data=X, y=y, batch_size=batch_size, num_epochs=5000, report_freq=50, lr=1e-3)
    model.train_model(data=X, y=y, batch_size=batch_size, num_epochs=5000, report_freq=50, lr=1e-4)
    model.train_model(data=X, y=y, batch_size=batch_size, num_epochs=5000, report_freq=50, lr=1e-5)

    # Evaluate loss iteractions
    ut.set_fig_prefix("bent_train")
    model.loss_analyzer.print_report()
    model.loss_analyzer.plot_correlation_matrix(figsize=None)

    # ## Evaluate results
    for prefix, data in [("train", X), ("test", X_test)]:
        images = data[0:25]
        ut.set_fig_prefix(prefix)
        report.analyze_reconstruction_error(model, data)
        ut.plot_reconstruction_comparison(model, pca, images, cmap="gray", nrow=5)
        report.orthogonality_test_analysis(model, data)
        report.variance_test_analysis(model, data)
        report.linearity_tests_analysis(model, data, alpha_min=0, num_samples=200)

    latents, reconstructed = model.predict(X)
    ut.plot2d_analysis(Xpca, y, title="PCA transform", legend=True)
    ut.plot2d_analysis(latents, y, title="POLCA-Net latent", legend=True)

    # Test Classification with two components on PCA vs POLCA Net
    ut.make_classification_report(model, pca, X_test, y_test, n_components=pca.n_components)

    experiment_data = {f"{name}": (X_test, model, pca,), }
    ut.image_metrics_table(experiment_data)


def train_on_torch_datasets(params):
    # frst train without labels
    for name, dataset in params.datasets.items():
        train_set, test_set = get_torch_dataset(dataset, name)
        print(f"Dataset: {name}")
        print(f"Train set: {len(train_set)}")
        print(f"Test set: {len(test_set)}")
        params.name = name
        params.train_set = train_set
        params.test_set = test_set
        params.n_channels = 1
        params.n_components = 28  # Heuristic sqrt(image size) * n_channels
        if params.batch_size is None:
            if train_set.data.shape[0] >= 50000:
                params.batch_size = 512
            else:
                params.batch_size = 256
        train_x_mnist(params)
        gc.collect()


def train_on_medmnist_datasets(params):
    # frst train without labels
    for name in params.datasets:
        train_set, test_set = get_medmnist_dataset(name)
        print(f"Dataset: {name}")
        print(f"Train set: {len(train_set)}")
        print(f"Test set: {len(test_set)}")
        params.name = name
        params.train_set = train_set
        params.test_set = test_set
        params.n_channels = INFO[name]['n_channels']
        params.n_components = 28 * params.n_channels  # Heuristic sqrt(image size) * n_channels
        # adjust batch size when the train set if > 50000
        if params.batch_size is None:
            if train_set.data.shape[0] >= 50000:
                params.batch_size = 512
            else:
                params.batch_size = 256
        train_x_mnist(params)
        gc.collect()


def get_torch_dataset(dataset, name):
    print("Loading dataset: ", name)
    train_set = dataset(root=f"data/{name}", train=True, download=True, transform=None)
    test_set = dataset(root=f"data/{name}", train=False, download=True, transform=None)

    # normalize the data
    train_set.data = train_set.data.reshape(-1, 28, 28) / 255.0
    test_set.data = test_set.data.reshape(-1, 28, 28) / 255.0
    train_set.data = train_set.data.numpy()
    test_set.data = test_set.data.numpy()
    train_set.target = train_set.targets.numpy()
    test_set.target = test_set.targets.numpy()

    return train_set, test_set


def get_medmnist_dataset(name):
    print("Loading dataset: ", name)
    info = INFO[name]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = INFO[name]['n_channels']

    # load the data
    # assure Path(f"data/{name}").absolute() exists if not create if perhaps exists=True?
    # create directory if not exists
    os.makedirs(Path(f"data/{name}").absolute(), exist_ok=True)
    train_set = DataClass(root=Path(f"data/{name}").absolute(), split='train', transform=None, download=True, size=28)
    test_set = DataClass(root=Path(f"data/{name}").absolute(), split='test', transform=None, download=True, size=28)

    train_set.targets = train_set.labels
    test_set.targets = test_set.labels

    if n_channels == 3:
        print("Converting to RGB")
        train_set.imgs = ut.convert_to_rgb_and_reshape(train_set.imgs) / 255.0
        test_set.imgs = ut.convert_to_rgb_and_reshape(test_set.imgs) / 255.0
        print(train_set.imgs.shape, test_set.imgs.shape)
    else:
        train_set.imgs = train_set.imgs.reshape(-1, 28, 28) / 255.0
        test_set.imgs = test_set.imgs.reshape(-1, 28, 28) / 255.0

    # normalize the data
    train_set.data = train_set.imgs
    test_set.data = test_set.imgs

    return train_set, test_set


if __name__ == "__main__":
    # ### Load dataset
    torch_datasets_dict = {"mnist": MNIST,
                           "fmnist": FashionMNIST,
                           }

    medmnist_datasets_dict = {"breastmnist": "BreastMNIST",
                              "dermamnist": "Dermamnist",
                              "octmnist": "OCTMNIST",
                              "organamnist": "OrganAMNIST",
                              "organcmnist": "OrganCMNIST",
                              "organsmnist": "OrganSMNIST",
                              "pathmnist": "PathMNIST",
                              "pneumoniamnist": "PneumoniaMNIST",
                              "retinamnist": "RetinaMNIST",
                              "bloodmnist": "BloodMNIST",
                              "chestmnist": "ChestMNIST",
                              }
    print(f"Available datasets: {torch_datasets_dict.keys()}")
    print(f"Available medmnist datasets: {INFO.keys()}")

    # define console parameters
    args = argparse.ArgumentParser()
    args.add_argument("--datasets", type=str, nargs="+", help="datasets to train the model")
    args.add_argument("--device", type=str, default="cuda", help="device to train the model")
    args.add_argument("--seed", type=int, default=5, help="random seed")
    args.add_argument("--with_labels", action="store_true", default=False, help="train with labels")
    args.add_argument("--n_components", type=int, default=28, help="number of components")
    args.add_argument("--batch_size", type=int, default=None, help="batch size")
    args.add_argument("--linear", action="store_true", default=False, help="use linear decoder")
    parameters = args.parse_args()

    for name in parameters.datasets:
        assert name in (medmnist_datasets_dict | torch_datasets_dict), f"dataset not available: {name}"
        # make a small dict with name, and class with the dataset name in parameters
        if name in torch_datasets_dict:
            train_on_torch_datasets(parameters)
        else:
            train_on_medmnist_datasets(parameters)

    print("Done")
