# Principal Orthogonal Latent Components Analysis Net (POLCA-Net)
import argparse
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt

# Redefine plt.show() to do nothing
plt.show = lambda: None

import numpy as np
import torch
import torchinfo
from torchvision.datasets import MNIST, FashionMNIST

import polcanet.reports as report
import polcanet.utils as ut
from polcanet import PolcaNet
from polcanet.aencoders import ConvEncoder, LinearDecoder

import medmnist
from medmnist import INFO

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)


def train_x_mnist(params):
    device = params.device
    random_seed = params.seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

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
    mnist_train_set = params.train_set
    mnist_test_set = params.test_set

    train_dataset = mnist_train_set.data
    eval_dataset = mnist_test_set.data
    y = mnist_train_set.targets.numpy()
    y_test = mnist_test_set.targets.numpy()
    labels = np.unique(y) if with_labels else None

    X = np.array(train_dataset.numpy(), dtype=np.float32)
    X = np.squeeze(X)
    print(X.min(), X.max())
    X_test = np.array(eval_dataset.numpy(), dtype=np.float32)
    X_test = np.squeeze(X_test)
    print(train_dataset.shape, eval_dataset.shape, X.shape, X_test.shape, y.shape, y_test.shape, labels)

    ut.set_fig_prefix("train")
    ut.plot_train_images(X, f"{params.name} train dataset images", cmap="gray", n=7)
    ut.set_fig_prefix("test")
    ut.plot_train_images(X_test, f"{params.name} test dataset images", cmap="gray", n=7)

    # ### Fit standard sklearn PCA
    n_components = params.n_components  # Heuristic sqrt(image size)
    fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
    pca = ut.get_pca(X, n_components=n_components, title=f"PCA on {params.name}", ax=axs, )
    Xpca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))

    # ### Fit POLCANet
    N = X[0].shape[0]
    M = X[0].shape[1]
    act_fn = torch.nn.SiLU
    input_dim = (N, M)
    latent_dim = pca.n_components
    assert N == input_dim[0], "input_dim[0] should match first matrix dimension N"
    assert M == input_dim[1], "input_dim[1] should match second matrix dimension M"

    encoder = ConvEncoder(input_channels=params.n_channels,
                          latent_dim=latent_dim,
                          conv_dim=2,
                          initial_channels=8,
                          growth_factor=2,
                          num_layers=5,
                          act_fn=act_fn)

    decoder = LinearDecoder(latent_dim=latent_dim,
                            input_dim=input_dim,
                            hidden_dim=512,
                            num_layers=5,
                            act_fn=act_fn,
                            bias=False, )

    model = PolcaNet(encoder=encoder,
                     decoder=decoder,
                     latent_dim=latent_dim,
                     alpha=1.0,  # orthogonality loss
                     beta=1.0,  # variance sorting loss
                     gamma=1.0,  # variance reduction loss
                     class_labels=labels,  # class labels for supervised in case labels is not None
                     )

    print(model)
    summary = torchinfo.summary(model, (1, input_dim[0], input_dim[1]), dtypes=[torch.float], verbose=1, col_width=16,
                                col_names=["kernel_size", "output_size", "num_params"], row_settings=["var_names"], )
    ut.save_text(str(model), "model.txt")
    ut.save_text(str(summary), "model_summary.txt")

    model.to(device)
    model.train_model(data=X, y=y, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-3)

    model.train_model(data=X, y=y, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-4)

    model.train_model(data=X, y=y, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-5)

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
    for name, dataset in torch_datasets_dict.items():
        test_set, train_set = get_torch_dataset(dataset, name)
        print(f"Dataset: {name}")
        print(f"Train set: {len(train_set)}")
        print(f"Test set: {len(test_set)}")
        params.name = name
        params.train_set = train_set
        params.test_set = test_set
        params.n_channels = 1
        train_x_mnist(params)


def train_on_medmnist_datasets(params):
    # frst train without labels
    for name in params.datasets:
        test_set, train_set = get_medmnist_dataset(name)
        print(f"Dataset: {name}")
        print(f"Train set: {len(train_set)}")
        print(f"Test set: {len(test_set)}")
        params.name = name
        params.train_set = train_set
        params.test_set = test_set
        params.n_channels = INFO[name]['n_channels']
        train_x_mnist(params)


def get_torch_dataset(dataset, name):
    assert name in ["mnist", "fmnist"], f"dataset not yet available for testing POLCA Nett: {name}"
    train_set = dataset(root=f"data/{name}", train=True, download=True, transform=None)
    test_set = dataset(root=f"data/{name}", train=False, download=True, transform=None)

    # normalize the data
    train_set = train_set.data.reshape(-1, 28, 28) / 255.0
    test_set = test_set.data.reshape(-1, 28, 28) / 255.0

    return test_set, train_set


def get_medmnist_dataset(name):
    info = INFO[name]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info[name]['n_channels']
    assert name in ["breastmnist",
                    "dermamnist",
                    "organmnist_axial",
                    "organmnist_coronal",
                    "organmnist_sagittal",
                    "pneumoniamnist",
                    "retinamnist"], f"dataset not yet available for testing POLCA Nett: {name}"

    # load the data
    train_set = DataClass(root=f"data/{name}", split='train', transform=None, download=True, size=28)
    test_set = DataClass(root=f"data/{name}", split='test', transform=None, download=True, size=28)

    # normalize the data
    train_set = train_set.data.reshape(-1, n_channels, train_set.size, train_set.size) / 255.0
    test_set = test_set.data.reshape(-1, n_channels, test_set.size, test_set.size) / 255.0

    return test_set, train_set


if __name__ == "__main__":
    # ### Load dataset
    torch_datasets_dict = {"mnist": MNIST,
                           "fmnist": FashionMNIST,
                           }

    medmnist_datasets_dict = {"breastmnist": "BreastMNIST",
                              # "chestmnist": "ChestMNIST",
                              "dermamnist": "Dermamnist",
                              # "octmnist": "OCTMNIST",
                              "organamnist": "OrganMNIST",
                              "organcmnist": "OrganCMNIST",
                              "organsmnist": "OrganSMNIST",
                              # "pathmnist": "PathMNIST",
                              "pneumoniamnist": "PneumoniaMNIST",
                              "retinamnist": "RetinaMNIST",
                              "bloodmnist": "BloodMNIST",

                              }

    # define console parameters
    args = argparse.ArgumentParser()
    args.add_argument("--device", type=str, default="cuda", help="device to train the model")
    args.add_argument("--seed", type=int, default=42, help="random seed")
    args.add_argument("--with_labels", action="store_true", default=False, help="train with labels")
    args.add_argument("--n_components", type=int, default=28, help="number of components")
    # add a new argument as a list of datasets to run
    args.add_argument("--datasets", type=str, nargs="+", default=["mnist", "fmnist"],
                      help="datasets to train the model")

    args = args.parse_args()

    parameters = SimpleNamespace(device='cuda', seed=42, with_labels=False)

    # train without labels
    parameters.with_labels = False
    train_on_torch_datasets(parameters)

    # train with labels
    parameters.with_labels = True
    train_on_torch_datasets(parameters)
