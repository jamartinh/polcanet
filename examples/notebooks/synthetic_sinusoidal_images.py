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
#
# Study on simple random sinusoidal images vs. real valued bent functions (maximaly non-linear) images
# -

# ### Simple sinusoidal images:
# Generate sinusoidal data with random phase and frequency:
#
# $$Z = \sin(2\pi f_x x + \phi_x) \cos(2\pi f_y y + \phi_y)$$
#
# Where:
# - $f_x$ is the frequency multiplier for x
# - $f_y$ is the frequency multiplier for y
# - $\phi_x$ is the phase shift for x
# - $\phi_y$ is the phase shift for y
#
# ### Real valued Bent functions:
#
# $$Z = \cos(2\pi(aX + bY)) + \cos(2\pi(cX - dY))$$
#
# Where $a$, $b$, $c$, and $d$ are parameters controlling the function's behavior.
#
#

# ## Imports and Initialization

# %load_ext autoreload
# %autoreload 2

# +
import matplotlib.pyplot as plt

plt.style.use(['science', 'no-latex'])

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
import torchinfo
# -

from polcanet import PolcaNet
import polcanet.utils as ut
import polcanet.reports as report

# +
import random

random_seed = 5
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

exp = ut.ExperimentInfoHandler(
    name="SYNTH_dataset",
    description="POLCA-Net on Synthetic Sinudosial and Bent Images",
    random_seed=random_seed,
)
ut.set_save_fig(True)
ut.set_save_path(str(exp.get_experiment_folder()))
print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")
# -

# ## Generate Synthetic Sinudosial and Bent Images

# + editable=true slideshow={"slide_type": ""}
# Parameters
N = 32  # rows
M = 32  # cols
num_samples = 1000

# Generate 2D sinusoidal data
data_sin = ut.generate_2d_sinusoidal_data(N, M, num_samples=num_samples)
data_sin_test = ut.generate_2d_sinusoidal_data(N, M, num_samples=num_samples)

# Generate 2D real bent function images data
data_bent = ut.generate_bent_images(N, M, num_samples=5000)
data_bent_test = ut.generate_bent_images(N, M, num_samples=num_samples)
print(data_sin.min(), data_sin.max())
print(data_bent.min(), data_bent.max())
ut.set_fig_prefix("sin_train")
ut.plot_train_images(data_sin, "Sinusoidal images", cmap="viridis", n=5)
ut.set_fig_prefix("bent_train")
ut.plot_train_images(data_bent, "Bent images", cmap="viridis", n=5)

ut.set_fig_prefix("sin_test")
ut.plot_train_images(data_sin_test, "Sinusoidal test images", cmap="viridis", n=5)
ut.set_fig_prefix("bent_test")
ut.plot_train_images(data_bent_test, "Bent test images", cmap="viridis", n=5)
# -

# ### Perform PCA on datasets

# + editable=true slideshow={"slide_type": ""}
n_components = 8  # int((N*M)//100)
ut.set_fig_prefix("sin")
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
pca_sin = ut.get_pca(data_sin, ax=axs, title="PCA on Sinusoidal images", n_components=n_components)
plt.show()

n_components = int((N * M) / 40)
ut.set_fig_prefix("bent")
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
pca_bent = ut.get_pca(data_bent, ax=axs, title="PCA on Bent images", n_components=n_components)
plt.show()
# -

# ## POLCA-Net

# ### Train on Sinusoidal Images

# + editable=true slideshow={"slide_type": ""}
from polcanet.aencoders import ConvEncoder, LinearDecoder

ae_input = data_sin
act_fn = torch.nn.SiLU
input_dim = (N, M)
latent_dim = pca_sin.n_components
assert N == input_dim[0], "input_dim[0] should match first matrix dimension N"
assert M == input_dim[1], "input_dim[1] should match second matrix dimension M"

encoder_sin = ConvEncoder(
    input_channels=1,
    latent_dim=latent_dim,
    conv_dim=2,
    initial_channels=16,
    growth_factor=2,
    num_layers=5,
    act_fn=act_fn,
)

decoder_sin = LinearDecoder(
    latent_dim=latent_dim,
    input_dim=input_dim,
    hidden_dim=5 * 256,
    num_layers=5,
    act_fn=act_fn,
    bias=False,
)

model_sin = PolcaNet(
    encoder=encoder_sin,
    decoder=decoder_sin,
    latent_dim=latent_dim,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=0.0,  # variance reduction loss
    device="cuda",
    center=True,
    factor_scale=True,

)
print(model_sin)
summary = torchinfo.summary(
    model_sin,
    (1, *input_dim),
    dtypes=[torch.float],
    verbose=1,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"],
)
ut.save_text(str(model_sin), "model_sin.txt")
ut.save_text(str(summary), "model_sin_summary.txt")
# -

model_sin.to("cuda")
model_sin.train_model(data=data_sin, batch_size=512, num_epochs=10000, report_freq=20, lr=1e-3)

model_sin.train_model(data=data_sin, batch_size=512, num_epochs=10000, report_freq=20, lr=1e-4)

model_sin.train_model(data=data_sin, batch_size=512, num_epochs=10000, report_freq=20, lr=1e-5)

ut.set_fig_prefix("sin_train")
report.analyze_reconstruction_error(model_sin, data_sin, n_samples=1000)
ut.set_fig_prefix("sin_test")
report.analyze_reconstruction_error(model_sin, data_sin_test, n_samples=1000)

latents, reconstructed = model_sin.predict(data_sin)
data_sin.shape, reconstructed.shape, latents.shape

# + editable=true slideshow={"slide_type": ""}
ut.set_fig_prefix("sin_train")
images = data_sin[0:25]
ut.plot_reconstruction_comparison(model_sin, pca_sin, images, cmap="viridis", nrow=5)
ut.set_fig_prefix("sin_test")
images = data_sin_test[0:25]
ut.plot_reconstruction_comparison(model_sin, pca_sin, images, cmap="viridis", nrow=5)
# -

ut.set_fig_prefix("sin_train")
report.orthogonality_test_analysis(model_sin, data_sin)
ut.set_fig_prefix("sin_test")
report.orthogonality_test_analysis(model_sin, data_sin_test)

ut.set_fig_prefix("sin_train")
report.variance_test_analysis(model_sin, data_sin)
ut.set_fig_prefix("sin_test")
report.variance_test_analysis(model_sin, data_sin_test)

# + editable=true slideshow={"slide_type": ""}
ut.set_fig_prefix("sin_train")
report.linearity_tests_analysis(model_sin, data_sin, alpha_min=0, num_samples=200)
ut.set_fig_prefix("sin_test")
report.linearity_tests_analysis(model_sin, data_sin_test, alpha_min=0, num_samples=200)
# -

# ### Train on Bent Images

# +
from polcanet import PolcaNet

ae_input = data_bent
act_fn = torch.nn.SiLU
input_dim = (N, M)
latent_dim = pca_bent.n_components
assert N == input_dim[0], "input_dim[0] should match first matrix dimension N"
assert M == input_dim[1], "input_dim[1] should match second matrix dimension M"

encoder_bent = ConvEncoder(
    input_channels=1,
    latent_dim=latent_dim,
    conv_dim=2,
    initial_channels=16,
    growth_factor=2,
    num_layers=5,
    act_fn=act_fn,
)

decoder_bent = LinearDecoder(
    latent_dim=latent_dim,
    input_dim=input_dim,
    hidden_dim=5 * 256,
    num_layers=5,
    act_fn=act_fn,
    bias=False
)

model_bent = PolcaNet(
    encoder=encoder_bent,
    decoder=decoder_bent,
    latent_dim=latent_dim,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=0.0,  # variance reduction loss
    device="cuda",
    center=True,
    factor_scale=True,

)
print(model_bent)
summary = torchinfo.summary(
    model_bent,
    (1, *input_dim),
    dtypes=[torch.float],
    verbose=1,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"],
)
ut.save_text(str(model_bent), "model_bent.txt")
ut.save_text(str(summary), "model_bent_summary.txt")
# -

model_bent.to("cuda")
model_bent.train_model(data=data_bent, batch_size=512, num_epochs=5000, report_freq=20, lr=1e-3)

model_bent.train_model(data=data_bent, batch_size=512, num_epochs=10000, report_freq=20, lr=1e-4)

model_bent.train_model(data=data_bent, batch_size=512, num_epochs=10000, report_freq=20, lr=1e-5)

# + editable=true slideshow={"slide_type": ""}
ut.set_fig_prefix("bent_train")
report.analyze_reconstruction_error(model_bent, data_bent, n_samples=1000)
ut.set_fig_prefix("bent_test")
report.analyze_reconstruction_error(model_bent, data_bent_test, n_samples=1000)
# -

latents, reconstructed = model_bent.predict(data_bent)
data_bent.shape, reconstructed.shape, latents.shape

# + editable=true slideshow={"slide_type": ""}
ut.set_fig_prefix("sin_train")
images = data_bent[0:25]
ut.plot_reconstruction_comparison(model_bent, pca_bent, images, cmap="viridis", nrow=5)
ut.set_fig_prefix("sin_test")
images = data_bent_test[0:25]
ut.plot_reconstruction_comparison(model_bent, pca_bent, images, cmap="viridis", nrow=5)

# + editable=true slideshow={"slide_type": ""}
ut.set_fig_prefix("bent_train")
report.orthogonality_test_analysis(model_bent, data_bent)
ut.set_fig_prefix("bent_test")
report.orthogonality_test_analysis(model_bent, data_bent_test)
# -

ut.set_fig_prefix("bent_train")
report.variance_test_analysis(model_bent, data_bent)
ut.set_fig_prefix("bent_test")
report.variance_test_analysis(model_bent, data_bent_test)

ut.set_fig_prefix("bent_train")
report.linearity_tests_analysis(model_bent, data_bent, alpha_min=0, num_samples=200)
ut.set_fig_prefix("bent_test")
report.linearity_tests_analysis(model_bent, data_bent_test, alpha_min=0, num_samples=200)

# ## Test Overall

# + editable=true slideshow={"slide_type": ""}
experiment_data = {
        "Sinudoidal": (
                data_sin,
                model_sin,
                pca_sin,
        ),
        "Bent": (
                data_bent,
                model_bent,
                pca_bent,
        ),
}
_ = ut.image_metrics_table(experiment_data)
# -
