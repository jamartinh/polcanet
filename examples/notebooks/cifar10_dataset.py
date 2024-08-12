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
import matplotlib.pyplot as plt

# %%
import numpy as np
import torch
import torchinfo
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
from polcanet import PolcaNet
import polcanet.reports as report
import polcanet.utils as ut

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
    name="cifar10_dataset",
    description="POLCA-Net on cifar10 dataset",
    random_seed=random_seed,
)
ut.set_save_fig(True)
ut.set_save_path(str(exp.get_experiment_folder()))
print(f"Saving Images: {ut.get_save_fig()}, saving in path: {ut.get_save_path()}")

# %% [markdown]
# ### Load dataset

# %%
from torchvision.datasets import CIFAR10
# Load CIFAR-10 dataset
cifar_trainset = CIFAR10(root="data/CIFAR10", train=True, download=True, transform=None)
cifar_testset = CIFAR10(root="data/CIFAR10", train=False, download=True, transform=None)

# %%
train_dataset = cifar_trainset.data / 255.0  #.reshape(-1, 32, 32, 3) / 255.0 
eval_dataset = cifar_testset.data / 255.0 # .reshape(-1, 32, 32, 3) / 255.0   

y = np.array(cifar_trainset.targets)
y_test = np.array(cifar_testset.targets)

X = np.array(train_dataset, dtype=np.float32)
X = np.squeeze(X)

X_test = np.array(eval_dataset, dtype=np.float32)
X_test = np.squeeze(X_test)

if X.ndim==4:
    X = np.moveaxis(X, -1, 1)
    X_test = np.moveaxis(X_test, -1, 1)
    

train_dataset.shape, eval_dataset.shape, X.shape, X_test.shape, y.shape, y_test.shape, X[0].min(),X[0].max()

# %%
ut.set_fig_prefix("train")
print("cifar10 train dataset images:")
ut.plot_train_images(X, "", n=7)
ut.set_fig_prefix("test")
print("cifar10 dataset images:")
ut.plot_train_images(X_test, "", n=7)

# %% [markdown]
# ### Fit standard sklearn PCA

# %%
32*3*3

# %%
n_components = 32 * 3 
fig, axs = plt.subplots(1,1,sharex=True, sharey=True,layout='constrained')
pca = ut.get_pca(X,n_components=n_components,title="PCA on cifar10",ax=axs)
Xpca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))
plt.show()

# %% [markdown]
# ### Fit POLCANet

# %%
N = X[0].shape[-1]
M = X[0].shape[-2]
X[0].shape, N, M

# %%
act_fn = torch.nn.SiLU
input_dim = X[0].shape
latent_dim = pca.n_components
assert N == input_dim[-1], "input_dim[-1] should match first matrix dimension N"
assert M == input_dim[-2], "input_dim[-2] should match second matrix dimension M"

from polcanet.aencoders import ConvEncoder, ConvDecoder, LinearDecoder, VGG

encoder = ConvEncoder(
    input_channels=3,
    latent_dim=latent_dim,
    conv_dim=2,
    initial_channels=8,
    growth_factor=2,
    num_layers=3,
    act_fn=act_fn,
)

# encoder = VGG('VGG11', latent_dim=latent_dim, act_fn=act_fn)


# # Load a pre-trained ResNet model and modify it for CIFAR-10
# encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load a pre-trained ResNet18 model
# num_ftrs = encoder.fc.in_features
# encoder.fc = torch.nn.Linear(num_ftrs, latent_dim)  # Modify the last layer for 10 classes of CIFAR-10



decoder = LinearDecoder(
    latent_dim=latent_dim,
    input_dim=input_dim,
    hidden_dim=8*256,
    num_layers=3,
    act_fn=act_fn,
    bias=False,
)

# decoder = ConvDecoder(
#     latent_dim=latent_dim,
#     output_channels=3,
#     conv_dim=2,
#     num_layers=3,
#     initial_channels=8,
#     growth_factor=2,
#     act_fn= torch.nn.Identity, # act_fn,
#     output_act_fn= torch.nn.Identity, # torch.nn.Sigmoid,
#     final_output_size=(32, 32),
# )


model = PolcaNet(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=1.0,  # variance reduction loss   
)
print(model)
summary = torchinfo.summary(
    model,
    (1,3, N, M),
    dtypes=[torch.float],
    verbose=1,
    col_width=16,
    col_names=["kernel_size", "output_size", "num_params"],
    row_settings=["var_names"],
)
ut.save_text(str(model), "model.txt")
ut.save_text(str(summary), "model_summary.txt")

# %%
# train_dataloader = DataLoader(X, batch_size=2*512, shuffle=True,num_workers=0)
# test_dataloader = DataLoader(X_test, batch_size=2*512, shuffle=True, num_workers=0)

# %%
model.to(device)
model.train_model(data=X,batch_size=2*256, num_epochs=1000, report_freq=10, lr=1e-3)

# %%
model.train_model(data=X,batch_size=2*256, num_epochs=1000, report_freq=10, lr=1e-4)

# %%
model.train_model(data=X, batch_size=2*256, num_epochs=100, report_freq=10, lr=1e-5)

# %% [markdown]
# ## Evaluate results

# %% [markdown]
# ## Evaluate results

# %%
ut.set_fig_prefix("train")
report.analyze_reconstruction_error(model, X[:5000])
ut.set_fig_prefix("test")
report.analyze_reconstruction_error(model, X_test[:5000])

# %%
latents, reconstructed = model.predict(X)

# %%
# Assuming images are properly defined as before
images = X[0:25]
ut.set_fig_prefix("train")
ut.plot_reconstruction_comparison(model,pca,images,n_components=pca.n_components,nrow=5)
images = X_test[0:25]
ut.set_fig_prefix("test")
ut.plot_reconstruction_comparison(model,pca,images,n_components=pca.n_components,nrow=5)

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
report.linearity_tests_analysis(model, X,     alpha_min=0,num_samples=100)
ut.set_fig_prefix("test")
report.linearity_tests_analysis(model, X_test,alpha_min=0,num_samples=100)

# %% [markdown]
# ## Test Classification with two components on PCA vs POLCA Net

# %%
_ = ut.make_classification_report(model, pca, X_test, y_test,n_components=pca.n_components)

# %%
experiment_data = {
    "cifar10" : (X_test,model,pca),   
}
df_image_metrics = ut.image_metrics_table(experiment_data, n_components=pca.n_components)

# %%
