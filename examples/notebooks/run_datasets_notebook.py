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
# ### Available Datasets:
# -     "mnist"
# -     "breastmnist"
# -     "dermamnist"
# -     "octmnist"
# -     "organamnist"
# -     "organcmnist"
# -     "organsmnist"
# -     "pathmnist"
# -     "pneumoniamnist"
# -     "retinamnist"
# -     "bloodmnist"
# -     "chestmnist"
#

# %%
# %run -tie train_datasets.py --datasets mnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets fmnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets cifar10
# %reset -f

# %%
# %run -tie train_datasets.py --datasets breastmnist
# %reset -f

# %%
# %run -tn train_datasets.py --datasets pneumoniamnist
# %reset -f

# %%
# color
# %run -tie train_datasets.py --datasets retinamnist
# %reset -f

# %%
# color
# %run -tie train_datasets.py --datasets dermamnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets octmnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets organamnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets organcmnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets organsmnist
# %reset -f

# %%
# color
# %run -tie train_datasets.py --datasets pathmnist
# %reset -f

# %%
# color
# %run -tie train_datasets.py --datasets bloodmnist
# %reset -f

# %%
# %run -tie train_datasets.py --datasets chestmnist
# %reset -f

# %%

# %%
