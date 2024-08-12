# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # **P**rincipal **O**rthogonal **L**atent **C**omponents **A**nalysis Net (POLCA-Net)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from IPython.display import display
import ipywidgets as widgets
import matplotlib.pyplot as plt

# %%
import numpy as np
import torch
import torchinfo
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
import polcanet.reports as report
import polcanet.utils as ut
from polcanet import PolcaNet

# %%
import random

random_seed = 5
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
with_labels = True
if with_labels:
    name = "MNIST_dataset_labels"
else:
    name = "MNIST_dataset"
    
exp = ut.ExperimentInfoHandler(
    name=name,
    description="POLCA-Net on MNIST dataset",
    random_seed=random_seed,
)
ut.set_save_fig(True)
ut.set_save_path(str(exp.get_experiment_folder()))
print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

# %% [markdown]
# ### Load dataset

# %%
from torchvision.datasets import MNIST
mnist_trainset = MNIST(root="data/MNIST", train=True, download=True, transform=None)
mnist_testset = MNIST(root="data/MNIST", train=False, download=True, transform=None)

# %%
train_dataset = mnist_trainset.data.reshape(-1, 28, 28) / 255.0
eval_dataset = mnist_testset.data.reshape(-1, 28, 28) / 255.0
y = mnist_trainset.targets.numpy()
y_test = mnist_testset.targets.numpy()

labels = np.unique(y) if with_labels else None

X = np.array(train_dataset.numpy(), dtype=np.float32)
X = np.squeeze(X)
print(X.min(),X.max())
X_test = np.array(eval_dataset.numpy(), dtype=np.float32)
X_test = np.squeeze(X_test)
train_dataset.shape, eval_dataset.shape, X.shape,X_test.shape, y.shape, y_test.shape, labels

# %%
ut.set_fig_prefix("sin_train")
ut.plot_train_images(X, "MNIST train dataset images",cmap="gray", n=7)
ut.set_fig_prefix("sin_test")
ut.plot_train_images(X_test, "MNIST test dataset images",cmap="gray", n=7)

# %% [markdown]
# ### Fit standard sklearn PCA

# %%
n_components = 28 #  Heuristic sqrt(image size)
fig, axs = plt.subplots(1,1,sharex=True, sharey=True,layout='constrained')
pca = ut.get_pca(X,n_components=n_components,title="PCA on MNIST",ax=axs,)
Xpca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))
plt.show()

# %% [markdown]
# ### Fit POLCANet

# %%
N = X[0].shape[0]
M = X[0].shape[1]

# %%
from polcanet.aencoders import ConvEncoder, LinearDecoder

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
    bias=False,
)

model = PolcaNet(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=1.0,  # variance reduction loss
    class_labels=labels,  # class labels for supervised in case labels is not None
)

print(model)
summary = torchinfo.summary(
    model,
    (1, input_dim[0], input_dim[1]),
    dtypes=[torch.float],
    verbose=1,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"],
)
ut.save_text(str(model), "model.txt")
ut.save_text(str(summary), "model_summary.txt")

# %%
model.to(device)
model.train_model(data=X,y=y,batch_size=512, num_epochs=5000, report_freq=10, lr=1e-3)

# %%
model.train_model(data=X,y=y,batch_size=512, num_epochs=5000, report_freq=10, lr=1e-4)

# %%
model.train_model(data=X,y=y, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-5)

# %% [markdown]
# ## Evaluate results

# %%
ut.set_fig_prefix("train")
report.analyze_reconstruction_error(model, X)
ut.set_fig_prefix("test")
report.analyze_reconstruction_error(model, X_test)

# %%
latents, reconstructed = model.predict(X)

# %%
# Assuming images are properly defined as before
images = X[0:25]
ut.set_fig_prefix("train")
ut.plot_reconstruction_comparison(model,pca,images,cmap="gray",nrow=5)
images = X_test[0:25]
ut.set_fig_prefix("test")
ut.plot_reconstruction_comparison(model,pca,images,cmap="gray",nrow=5)

# %%
ut.set_fig_prefix("train")
report.orthogonality_test_analysis(model, X)
ut.set_fig_prefix("test")
report.orthogonality_test_analysis(model, X_test)

# %%
ut.set_fig_prefix("train")
report.variance_test_analysis(model, X)
ut.set_fig_prefix("test")
report.variance_test_analysis(model, X_test)

# %%
ut.set_fig_prefix("train")
report.linearity_tests_analysis(model, X,alpha_min=0, num_samples=200)
ut.set_fig_prefix("test")
report.linearity_tests_analysis(model, X_test,alpha_min=0, num_samples=200)

# %%
o1 = widgets.Output()
o2 = widgets.Output()
with o1:
    _, _ = ut.plot2d_analysis(Xpca, y, title="PCA transform", legend=True)
with o2:
    _, _ = ut.plot2d_analysis(latents, y, title="POLCA-Net latent",  legend=True)
layout = widgets.Layout(grid_template_columns="repeat(2, 600px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# %%
latents, reconstructed = model.predict(X)
vectors = []
labels = [str(i) for i in range(10)]
for c, label in enumerate(labels):
    vectors.append(np.sum(latents[y == c, :], axis=1))


plt.boxplot(vectors, tick_labels=labels)
plt.violinplot(vectors, showmeans=False, showmedians=True)
plt.suptitle("Polca Analysis of the summation of latent orthogonal components")
plt.show()

# %% [markdown]
# ## Test Classification with two components on PCA vs POLCA Net

# %%
_ = ut.make_classification_report(model, pca, X_test, y_test, n_components=pca.n_components)

# %%
experiment_data = {
    "MNIST": (
        X_test,
        model,
        pca,
    ),
}
_ = ut.image_metrics_table(experiment_data)

# %%

