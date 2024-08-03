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
from polcanet.example_aencoders import (
    ConvEncoder,
    DenseDecoder,
    DenseEncoder,
    LSTMEncoder,
    MinMaxScalerTorch,
    StandardScalerTorch,
)
# -

import polcanet.polcanet_reports as report

import utils as ut 

np.random.seed(5)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# ### Load dataset

import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST
mnist_trainset = MNIST(root="data/MNIST", train=True, download=True, transform=None)

train_dataset = mnist_trainset.data[:-10000].reshape(-1, 28, 28) / 255.0
eval_dataset = mnist_trainset.data[-10000:].reshape(-1, 28, 28) / 255.0
y_train = mnist_trainset.targets[:-10000].numpy()
y_test = mnist_trainset.targets[-10000:].numpy()
X = np.array(train_dataset.numpy(), dtype=np.float32)
X = np.squeeze(X)
X_test = np.array(eval_dataset.numpy(), dtype=np.float32)
X_test = np.squeeze(X_test)
train_dataset.shape, eval_dataset.shape, X.shape,X_test.shape, y_train.shape, y_test.shape


# +
def plot_train_images(x, title, n=1):
    # Plot original and reconstructed signals for a sample
    fig, axes = plt.subplots(1, n + 1)
    fig.subplots_adjust(wspace=0.01)
    im_list = list(range(n)) + [-1]
    for i in im_list:
        axes[i].imshow(x[i], cmap="gray")
        if i==n//2:
            axes[i].set_title(f"{title}")
        axes[i].axis("off")
        axes[i].text(x[i].shape[0] // 2, x[i].shape[1] // 2, str(i),color="white")

    plt.show()

plot_train_images(X, "Sinusoidal images", n=5)
# -

# ### Fit standard sklearn PCA

n_components = int(np.prod(X.shape[1:]) // 25)
fig, axs = plt.subplots(1,1,sharex=True, sharey=True,layout='constrained')
pca = get_pca(X,n_components=n_components,title="PCA on MNIST",ax=axs,)
Xpca = pca.transform(np.squeeze(X.reshape(X.shape[0], -1)))
plt.show()

# ### Fit POLCANet

N = X[0].shape[0]
M = X[0].shape[1]

# +
act_fn = torch.nn.SiLU
input_dim = (N, M)
latent_dim = n_components
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
    act_fn=act_fn
)

model = PolcaNet(
    encoder=encoder,
    decoder=decoder,
    latent_dim=latent_dim ,
    alpha=1.0,  # ortgogonality loss
    beta=1.0,  # variance sorting loss
    gamma=1.0,  # variance reduction loss
    device=device,   
)
model

# +
# train_dataloader = DataLoader(X, batch_size=2*512, shuffle=True,num_workers=0)
# test_dataloader = DataLoader(X_test, batch_size=2*512, shuffle=True, num_workers=0)
# -

model.to(device)
model.train_model(data=X,batch_size=2*512, num_epochs=2000, report_freq=10, lr=1e-3)

# + jupyter={"outputs_hidden": false}
model.train_model(data=X,batch_size=2*512, num_epochs=2000, report_freq=10, lr=1e-4)

# + jupyter={"outputs_hidden": false}
model.train_model(data=X, batch_size=2*512, num_epochs=2000, report_freq=10, lr=1e-5)
# -

# ## Evaluate results

report.analyze_reconstruction_error(model, X[:-20000])

latents, reconstructed = model.predict(X)

# +
# Assuming images are properly defined as before
images = X_test[0:50]

# Reconstruct and visualise the images using the autoencoder
_, ae_reconstructed = model.predict(images)

# Reconstruct and visualize the imagaes by PCA
pca_latents = pca.transform(images.reshape(images.shape[0], -1))
pca_reconstructed = pca.inverse_transform(pca_latents)
pca_reconstructed = pca_reconstructed.reshape(images.shape[0], N, M)

ut.visualise_reconstructed_images(
    [images, ae_reconstructed, pca_reconstructed],
    title_list=["Original", "POLCA-Net reconstruction", "PCA reconstruction"],
    cmap="gray",nrow=10,
)
# -

report.analyze_latent_space(model, latents=latents)

report.orthogonality_test_analysis(model, X)

report.variance_test_analysis(model, X)

report.linearity_tests_analysis(model, X)


def plot2d_analysis(X, y, title, legend=True):
    fig = plt.figure(1, figsize=(5, 5))
    ax = fig.add_subplot(111)

    for label in range(10):
        ax.scatter(X[y == label, 0], X[y == label, 1], label=label)
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
    _, _ = plot2d_analysis(Xpca, y_train, title="PCA transform", legend=True)
with o2:
    _, _ = plot2d_analysis(latents, y_train, title="POLCA-Net latent")
layout = widgets.Layout(grid_template_columns="repeat(2, 600px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# +
latents, reconstructed = model.predict(X)
vectors = []
labels = [str(i) for i in range(10)]
for c, label in enumerate(labels):
    vectors.append(np.sum(latents[y_train == c, :], axis=1))


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

import pandas as pd
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import minmax_scale, scale
from sklearn.svm import SVC

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_train, test_size=0.3, random_state=1)

X_train_pca = pca.transform(X_train.reshape(X_train.shape[0], -1))
X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))
X_train_pca.shape, X_test_pca.shape

# Transform the data using POLCA-Net
# X_train_polca = model.predict(X_train,np.array([1, 1, 0, 0]))[0][:,:2]
X_train_polca = model.predict(X_train)[0][:, :pca.n_components]
# X_test_polca = model.predict(X_test, np.array([1, 1, 0, 0]))[0][:,:2]
X_test_polca = model.predict(X_test)[0][:, :pca.n_components]
X_train_polca.shape, X_test_polca.shape

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(solver='saga',n_jobs=30,max_iter=500),
    "Gaussian Naive Bayes": GaussianNB(),
    "Linear SVM": SVC(kernel="linear", probability=True),
    "Ridge Classifier": RidgeClassifier(),
    "Perceptron": Perceptron(n_jobs=30),
}

# Train and evaluate classifiers on both PCA and POLCA-Net transformed datasets
results = []
from tqdm.auto import tqdm
for name, clf in tqdm(classifiers.items()):
    # Train on PCA
    clf.fit(minmax_scale(X_train_pca), y_train)
    y_pred_pca = clf.predict(minmax_scale(X_test_pca))
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    report_pca = classification_report(y_test, y_pred_pca, output_dict=True)
    cm_pca = confusion_matrix(y_test, y_pred_pca)

    # Train on POLCA-Net
    clf.fit(minmax_scale(X_train_polca), y_train)
    y_pred_polca = clf.predict(minmax_scale(X_test_polca))
    accuracy_polca = accuracy_score(y_test, y_pred_polca)
    report_polca = classification_report(y_test, y_pred_polca, output_dict=True)
    cm_polca = confusion_matrix(y_test, y_pred_polca)

    # Append results
    results.append(
        {
            "Classifier": name,
            "Transformation": "PCA",
            "Accuracy": accuracy_pca,
            "Precision": report_pca["weighted avg"]["precision"],
            "Recall": report_pca["weighted avg"]["recall"],
            "F1-Score": report_pca["weighted avg"]["f1-score"],
            "Confusion Matrix": cm_pca,
        }
    )

    results.append(
        {
            "Classifier": name,
            "Transformation": "POLCA-Net",
            "Accuracy": accuracy_polca,
            "Precision": report_polca["weighted avg"]["precision"],
            "Recall": report_polca["weighted avg"]["recall"],
            "F1-Score": report_polca["weighted avg"]["f1-score"],
            "Confusion Matrix": cm_polca,
        }
    )

# +
# Create a DataFrame to display the results
results_df = pd.DataFrame(results)

# Display the main metrics table
main_metrics_df = results_df.drop(columns=["Confusion Matrix"])
main_metrics_df["n_components"] = pca.n_components
main_metrics_df
# -

# Statistical test: Paired t-test for accuracies
comparison_metrics = ["Accuracy","Precision","Recall","F1-Score"]
print(f"\nPaired t-test results:") 
for comparison_metric in comparison_metrics:

    print(f"{comparison_metric}:")
    pca_result = results_df[results_df["Transformation"] == "PCA"][comparison_metric]
    polca_result = results_df[results_df["Transformation"] == "POLCA-Net"][comparison_metric]    
    t_stat, p_value = ttest_rel(pca_result.values, polca_result.values)    
    print(f"\tt-statistic = {t_stat}, p-value = {p_value}, p-value threshold < {0.05}")    
    if p_value < 0.05:
        #print(f"There is a statistically significant difference between the PCA and POLCA-Net transformations")
        ans = "a"
    else:
        ans = "no"
    
    print(f"\tThere is {ans} statistically significant difference between the PCA and POLCA-Net transformations.")

# +
# Plotting the results
plt.figure()

# Plot PCA
plt.subplot(1, 2, 1)
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap="viridis", edgecolor=None, s=5)
plt.title("PCA: Test Set")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.axis("square")

# Plot POLCA-Net
plt.subplot(1, 2, 2)
plt.scatter(X_test_polca[:, 0], X_test_polca[:, 2], c=y_test, cmap="viridis", edgecolor=None, s=5)
plt.title("POLCA-Net: Test Set")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.axis("square")

plt.tight_layout()
plt.show()
# -

experiment_data = {
    "MNIST" : (X_test,model,pca),   
}
df_image_metrics = ut.image_metrics_table(experiment_data)

display(df_image_metrics.reset_index().style.hide())
print(df_image_metrics.reset_index().style.hide().to_latex())


