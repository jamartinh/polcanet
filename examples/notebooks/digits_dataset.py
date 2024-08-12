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

# %% [markdown]
# # **P**rincipal **O**rthogonal **L**atent **C**omponents **A**nalysis Net (POLCA-Net)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display


# %%
import numpy as np
import torch
import torchinfo
from sklearn import datasets

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

exp = ut.ExperimentInfoHandler(
    name="digits_dataset_8",
    description="POLCA-Net on digits dataset",
    random_seed=random_seed,
)
ut.set_save_fig(True)
ut.set_save_path(str(exp.get_experiment_folder()))
print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

# %% [markdown]
# ### Load dataset

# %%
digits = datasets.load_digits()
X = digits.data / 16.0
y = digits.target
print(X.min(), X.max())
X.shape, y.shape

# %%
images = X.reshape(X.shape[0], 8, 8)
ut.set_fig_prefix("train")
ut.plot_train_images(images, "digits dataset images", n=10)

# %% [markdown]
# ### Fit standard sklearn PCA

# %%
n_components = 8
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout="constrained")
pca = ut.get_pca(X, ax=axs, title="PCA on the digits dataset", n_components=n_components)
plt.show()
Xpca = pca.transform(X)

# %% [markdown]
# ### Fit POLCANet

# %%
from polcanet.aencoders import DenseDecoder, DenseEncoder, LinearDecoder

ae_input = X
act_fn = torch.nn.SiLU
input_dim = (ae_input.shape[1],)
latent_dim = pca.n_components

encoder = DenseEncoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    num_layers=1,
    act_fn=act_fn,
    first_layer_size=256,
    # hidden_size=512,
)

decoder = LinearDecoder(
    latent_dim=latent_dim,
    input_dim=input_dim,
    hidden_dim=256,
    num_layers=3,
    act_fn=act_fn,
    bias=True,
)

model = PolcaNet(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=0.0,  # variance reduction loss
    device="cuda",
    center=True,
    factor_scale=True,
)
print(model)
summary = torchinfo.summary(
    model,
    (1, input_dim[0]),
    dtypes=[torch.float],
    verbose=1,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"],
)
ut.save_text(str(model), "model.txt")
ut.save_text(str(summary), "model_summary.txt")

# %%
model.to("cuda")
model.train_model(data=X, batch_size=256, num_epochs=10000, report_freq=20, lr=1e-3)

# %%
model.train_model(data=X, batch_size=256, num_epochs=5000, report_freq=20, lr=1e-4)

# %%
model.train_model(data=X, batch_size=256, num_epochs=5000, report_freq=20, lr=1e-5)

# %% [markdown]
# ## Evaluate results

# %%
report.analyze_reconstruction_error(model, X)

# %%
latents, reconstructed = model.predict(X)

# %%
N = 32
# Assuming images are properly defined as before
images_to_show = images[:N]
ut.visualise_reconstructed_images([images_to_show], ["Original"], cmap="gray", nrow=8)
for i in range(2, 9, 2):
    print("n_components:", i)
    ut.plot_reconstruction_comparison(
        model, pca, images_to_show, cmap="gray", nrow=8, n_components=i, no_title=True, show_only_reconstruction=True
    )

# %%
report.orthogonality_test_analysis(model, X)

# %%
report.variance_test_analysis(model, X)

# %%
report.linearity_tests_analysis(model, X)

# %% [markdown]
# ## Polca Net vs. PCA

# %%
o1 = widgets.Output()
o2 = widgets.Output()
with o1:
    _, _ = ut.plot2d_analysis(Xpca, y, title="PCA transform", legend=True)
with o2:
    _, _ = ut.plot2d_analysis(latents, y, title="POLCA-Net latent", legend=True)

layout = widgets.Layout(grid_template_columns="repeat(2, 600px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# %% [markdown]
# ## Test Classification with two components on PCA vs POLCA Net

# %%
_ = ut.make_classification_report(model, pca, X, y, n_components=pca.n_components)

# %%
experiment_data = {
    "digits": (
        images,
        model,
        pca,
    ),
}
_ = ut.image_metrics_table(experiment_data)

# %%
