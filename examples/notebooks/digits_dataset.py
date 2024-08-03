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

import ipywidgets as widgets
import matplotlib.pyplot as plt
# +
from IPython.display import display
import scienceplots
type(scienceplots)
plt.style.use(["science", "no-latex"])

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
# -

import numpy as np
import torch
from sklearn import datasets

from polcanet import LinearDecoder, PolcaNet
from polcanet.example_aencoders import DenseEncoder

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
    name="digits_dataset",
    description="POLCA-Net on digits dataset",
    random_seed=random_seed,
)
report.set_save_fig(True)
report.set_save_path(str(exp.get_experiment_folder()))
print(f"Saving Images: {report.get_save_fig()}, saving in path: {report.get_save_path()}")
# -

# ### Load dataset

# + editable=true slideshow={"slide_type": ""}
digits = datasets.load_digits()
X = digits.data / 255
y = digits.target
images = X.reshape(X.shape[0], 8, 8)
ut.plot_train_images(images, "digits dataset images", n=10)
# -

# ### Fit standard sklearn PCA

# + editable=true slideshow={"slide_type": ""}
n_components = 32
fig, axs = plt.subplots(1, 1, sharex=True, sharey=True, layout='constrained')
pca = ut.get_pca(X, ax=axs, title="PCA on the digits dataset", n_components=n_components)
plt.show()
Xpca = pca.transform(X)
# -

# ### Fit POLCANet

# +
ae_input = X
act_fn = torch.nn.SiLU
input_dim = (ae_input.shape[1],)
latent_dim = pca.n_components

encoder = DenseEncoder(input_dim=input_dim, latent_dim=latent_dim, num_layers=1, act_fn=act_fn, first_layer_size=512,
                       # hidden_size=512,
                       )

decoder = LinearDecoder(latent_dim=latent_dim, input_dim=input_dim, hidden_dim=512, num_layers=2, act_fn=None,
                        bias=True, )

model = PolcaNet(encoder=encoder, decoder=decoder, latent_dim=latent_dim, alpha=0.1,  # ortgogonality loss
                 beta=0.1,  # variance sorting loss
                 gamma=0.0,  # variance reduction loss
                 device="cuda", center=True, factor_scale=True, )
model
# -

model.to("cuda")
model.train_model(data=X, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-3)

# + jupyter={"outputs_hidden": false}
model.train_model(data=X, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-4)

# + jupyter={"outputs_hidden": false}
model.train_model(data=X, batch_size=512, num_epochs=5000, report_freq=10, lr=1e-5)
# -

# ## Evaluate results

report.analyze_reconstruction_error(model, X)

latents, reconstructed = model.predict(X)

# +
# Assuming images are properly defined as before
N = 24
images_to_show = images[:N]
# Reconstruct and visualise the images using the autoencoder
_, ae_reconstructed = model.predict(X[:N])
ae_reconstructed = ae_reconstructed.reshape(images_to_show.shape)
# Reconstruct and visualize the imagaes by PCA
pca_latents = pca.transform(X[:N])
pca_reconstructed = pca.inverse_transform(pca_latents)
pca_reconstructed = pca_reconstructed.reshape(images_to_show.shape)

ut.visualise_reconstructed_images([images_to_show, ae_reconstructed, pca_reconstructed],
                                  title_list=["Original", "POLCA-Net reconstruction", "PCA reconstruction"],
                                  cmap="gray", nrow=6, )
# -

report.orthogonality_test_analysis(model, X)

report.variance_test_analysis(model, X)

report.linearity_tests_analysis(model, X)

# ## Polca Net vs. PCA

# +
o1 = widgets.Output()
o2 = widgets.Output()
with o1:
    _, _ = ut.plot2d_analysis(Xpca, y, title="PCA transform", legend=True)
with o2:
    _, _ = ut.plot2d_analysis(latents, y, title="POLCA-Net latent", legend=True)

layout = widgets.Layout(grid_template_columns="repeat(2, 600px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## Test Classification with two components on PCA vs POLCA Net

# + editable=true slideshow={"slide_type": ""}
_ = ut.make_classification_report(model, pca, X, y)

# + editable=true slideshow={"slide_type": ""}
experiment_data = {
    "digits": (
        X,
        model,
        pca,
    ),
}
_ = ut.image_metrics_table(experiment_data)
