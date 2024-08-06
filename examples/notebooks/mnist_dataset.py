# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] editable=true slideshow={"slide_type": ""}
# # **P**rincipal **O**rthogonal **L**atent **C**omponents **A**nalysis Net (POLCA-Net)
# -

# %load_ext autoreload
# %autoreload 2

# +
import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn


import scienceplots
plt.style.use(['science','no-latex'])

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


import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

from sklearn import datasets, decomposition

# + editable=true slideshow={"slide_type": ""}
from polcanet import LinearDecoder, PolcaNet, PolcaNetLoss
from polcanet.example_aencoders import ConvEncoder

# + editable=true slideshow={"slide_type": ""}
import polcanet.polcanet_reports as report

# + editable=true slideshow={"slide_type": ""}
import utils as ut
import random

random_seed = 5
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

exp = ut.ExperimentInfoHandler(
    name="MNIST_dataset",
    description="POLCA-Net on MNIST dataset",
    random_seed=random_seed,
)
report.set_save_fig(True)
report.set_save_path(str(exp.get_experiment_folder()))
print(f"Saving Images: {report.get_save_fig()}, saving in path: {report.get_save_path()}")
# -

# ### Load dataset

# + editable=true slideshow={"slide_type": ""}
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
mnist_trainset = MNIST(root="data/MNIST", train=True, download=True, transform=None)
mnist_testset = MNIST(root="data/MNIST", train=False, download=True, transform=None)
# -

train_dataset = mnist_trainset.data.reshape(-1, 28, 28) / 255.0
eval_dataset = mnist_testset.data.reshape(-1, 28, 28) / 255.0
y = mnist_trainset.targets.numpy()
y_test = mnist_testset.targets.numpy()
X = np.array(train_dataset.numpy(), dtype=np.float32)
X = np.squeeze(X)
X_test = np.array(eval_dataset.numpy(), dtype=np.float32)
X_test = np.squeeze(X_test)
train_dataset.shape, eval_dataset.shape, X.shape,X_test.shape, y.shape, y_test.shape

# + editable=true slideshow={"slide_type": ""}
report.set_fig_prefix("sin_train")
ut.plot_train_images(X, "MNIST train dataset images",cmap="gray", n=7)
report.set_fig_prefix("sin_test")
ut.plot_train_images(X_test, "MNIST test dataset images",cmap="gray", n=7)
# -

# ### Fit standard sklearn PCA

# + editable=true slideshow={"slide_type": ""}
n_components = 28 #  int(np.prod(X.shape[1:]) // 25)
fig, axs = plt.subplots(1,1,sharex=True, sharey=True,layout='constrained')
pca = ut.get_pca(X,n_components=n_components,title="PCA on MNIST",ax=axs,)
Xpca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))
plt.show()
# -

# ### Fit POLCANet

N = X[0].shape[0]
M = X[0].shape[1]

# + editable=true slideshow={"slide_type": ""}
act_fn = torch.nn.SiLU
input_dim = (N, M)
latent_dim = pca.n_components
assert N == input_dim[0], "input_dim[0] should match first matrix dimension N"
assert M == input_dim[1], "input_dim[1] should match second matrix dimension M"


encoder = ConvEncoder(
    input_channels=1,
    latent_dim=latent_dim,
    conv_dim=2,
    initial_channels=8,
    growth_factor=2,
    num_layers=5,
    act_fn=act_fn,
)

decoder = LinearDecoder(
    latent_dim=latent_dim,
    input_dim=input_dim,
    hidden_dim=512,
    num_layers=5,
    act_fn=act_fn,
    bias = False,
)

model = PolcaNet(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=0.0,  # variance reduction loss
    device=device,
    center=True,
    factor_scale=True,
)

report.save_text(str(model),"model.txt")
print(model)
# -

model.to(device)
model.train_model(data=X,batch_size=2*512, num_epochs=5000, report_freq=10, lr=1e-3)

# + jupyter={"outputs_hidden": false}
model.train_model(data=X,batch_size=2*512, num_epochs=5000, report_freq=10, lr=1e-4)

# + jupyter={"outputs_hidden": false}
model.train_model(data=X, batch_size=2*512, num_epochs=5000, report_freq=10, lr=1e-5)
# -

# ## Evaluate results

# + editable=true slideshow={"slide_type": ""}
report.set_fig_prefix("train")
report.analyze_reconstruction_error(model, X)
report.set_fig_prefix("test")
report.analyze_reconstruction_error(model, X_test)
# -

latents, reconstructed = model.predict(X)

# + editable=true slideshow={"slide_type": ""}
# Assuming images are properly defined as before
images = X[0:25]
report.set_fig_prefix("train")
ut.plot_reconstruction_comparison(model,pca,images,cmap="gray",nrow=5)
images = X_test[0:25]
report.set_fig_prefix("test")
ut.plot_reconstruction_comparison(model,pca,images,cmap="gray",nrow=5)
# -

report.set_fig_prefix("train")
report.orthogonality_test_analysis(model, X)
report.set_fig_prefix("test")
report.orthogonality_test_analysis(model, X_test)

report.set_fig_prefix("train")
report.variance_test_analysis(model, X)
report.set_fig_prefix("test")
report.variance_test_analysis(model, X_test)

report.set_fig_prefix("train")
report.linearity_tests_analysis(model, X)
report.set_fig_prefix("test")
report.linearity_tests_analysis(model, X_test)


def plot2d_analysis(X, y, title, legend=True):
    fig = plt.figure(1, figsize=(5, 5))
    ax = fig.add_subplot(111)

    for label in range(10):
        ax.scatter(X[y == label, 0], X[y == label, 1], label=label,alpha=0.7, s=1)
        ax.set_xlabel("component: 0")
        ax.set_ylabel("component 1")
    if legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.show()
    return fig, ax


o1 = widgets.Output()
o2 = widgets.Output()
with o1:
    _, _ = plot2d_analysis(Xpca, y, title="PCA transform", legend=True)
with o2:
    _, _ = plot2d_analysis(latents, y, title="POLCA-Net latent")
layout = widgets.Layout(grid_template_columns="repeat(2, 600px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# +
latents, reconstructed = model.predict(X)
vectors = []
labels = [str(i) for i in range(10)]
for c, label in enumerate(labels):
    vectors.append(np.sum(latents[y == c, :], axis=1))


plt.boxplot(vectors, tick_labels=labels)
plt.violinplot(vectors, showmeans=False, showmedians=True)
plt.suptitle("Polca Analysis of the summation of latent orthogonal components")
plt.show()

# +
import seaborn as sns

o1 = widgets.Output()
o2 = widgets.Output()


with o1:
    scores = model.score(X)
    sns.displot(scores, kde=True)
    plt.title("Last component with clean data")
    plt.show()

with o2:
    scores = model.score(X * (np.random.random(size=X.shape) - 0.5) * 1)
    sns.displot(scores, kde=True)
    plt.title("Last componet with uniform noise in data")
    plt.show()


layout = widgets.Layout(grid_template_columns="repeat(2, 500px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)
# -

# ## Test Classification with two components on PCA vs POLCA Net

# + editable=true slideshow={"slide_type": ""}
_ = ut.make_classification_report(model, pca, X, y)

# + editable=true slideshow={"slide_type": ""}
experiment_data = {
    "MNIST": (
        X_test,
        model,
        pca,
    ),
}
_ = ut.image_metrics_table(experiment_data)
# + editable=true slideshow={"slide_type": ""}


