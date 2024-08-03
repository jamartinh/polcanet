import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import datasets, decomposition

import polcanet.polcanet_reports as report
from polcanet import LinearDecoder, PolcaNet
from polcanet.example_aencoders import (DenseEncoder, )

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

np.random.seed(1)

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

iris = datasets.load_iris()
X = iris.data
y = iris.target
print(X.shape, y.shape)

# ### Fit standard sklearn PCA

pca = decomposition.PCA(n_components=2)
pca.fit(X)
Xpca = pca.transform(X)
print(pca.explained_variance_ratio_)

# ### Fit POLCANet


act_fn = torch.nn.SiLU
input_dim = (X.shape[1],)
latent_dim = 4

encoder_iris = DenseEncoder(
    input_dim=input_dim,
    latent_dim=latent_dim,
    num_layers=5,
    act_fn=act_fn,
    # first_layer_size= 512,
    hidden_size=128,
)

decoder_iris = LinearDecoder(
    latent_dim=latent_dim,
    input_dim=input_dim,
    hidden_dim=2048,
    num_layers=2,
    act_fn=None,
)

model_iris = PolcaNet(
    encoder=encoder_iris,
    decoder=decoder_iris,
    latent_dim=latent_dim,
    alpha=1,  # orthogonality loss
    beta=1.0,  # variance sorting loss
    gamma=1.0,  # variance reduction loss
    device="cuda",  # scaler = StandardScalerTorch(),
)
model = model_iris

# In[10]:


model_iris.to("cuda")
model_iris.train_model(data=X, batch_size=512, num_epochs=10000, report_freq=100, lr=1e-3)

# In[11]:


model_iris.train_model(data=X, batch_size=512, num_epochs=10000, report_freq=100, lr=1e-4)

# In[12]:


model_iris.train_model(data=X, batch_size=512, num_epochs=10000, report_freq=100, lr=1e-5)

# ## Evaluate results

# In[34]:


report.analyze_reconstruction_error(model_iris, X)

# In[35]:


latents, reconstructed = model_iris.predict(X)

# In[36]:


report.analyze_latent_space(model_iris, latents=latents)

# In[37]:


report.orthogonality_test_analysis(model_iris, X)

# In[38]:


report.variance_test_analysis(model_iris, X)

# In[39]:


report.linearity_tests_analysis(model_iris, X)


# ## Polca Net vs. PCA

# In[40]:


def plot2d_analysis(X, y, title, legend=True):
    fig = plt.figure(1, figsize=(5, 5))
    ax = fig.add_subplot(111)

    for name, label in [("Setosa", 0), ("Versicolour", 1), ("Virginica", 2)]:
        ax.scatter(X[y == label, 0], X[y == label, 1], label=name)
        ax.set_xlabel("component 0")
        ax.set_ylabel("component 1")
    if legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.show()
    return fig, ax


# In[41]:


o1 = widgets.Output()
o2 = widgets.Output()
with o1:
    _, _ = plot2d_analysis(Xpca, y, title="PCA transform", legend=True)
with o2:
    _, _ = plot2d_analysis(latents, y, title="POLCA-Net latent")
layout = widgets.Layout(grid_template_columns="repeat(2, 600px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# In[42]:


o1 = widgets.Output()
o2 = widgets.Output()
o3 = widgets.Output()
o4 = widgets.Output()

with o1:
    fig1, ax1 = plot2d_analysis(X, y, "Original data two first componets", legend=False)

with o2:
    latents, reconstructed = model_iris.predict(X, np.ones(latent_dim))
    fig2, ax2 = plot2d_analysis(np.round(reconstructed, 1), y, title="Reconstructed with POLCA all componets",
                                legend=False, )

with o3:
    latents, reconstructed = model_iris.predict(X, np.array([1, 1, 0, 0]))
    fig3, ax3 = plot2d_analysis(np.round(reconstructed, 1), y, title="Reconstructed with POLCA two componets",
                                legend=False, )

with o4:
    fig4, ax4 = plot2d_analysis(np.round(pca.inverse_transform(Xpca), 1), y, "Reconstructed with PCA two componets",
                                legend=False, )

layout = widgets.Layout(grid_template_columns="repeat(2, 450px)")
accordion = widgets.GridBox(children=[o1, o2, o3, o4], layout=layout)
display(accordion)

# In[43]:


latents, reconstructed = model_iris.predict(X)
vectors = []
labels = ["Setosa", "Versicolour", "Virginica"]
for c, label in enumerate(labels):
    vectors.append(np.sum(latents[y == c, :], axis=1))

plt.boxplot(vectors, tick_labels=labels)
plt.violinplot(vectors, showmeans=False, showmedians=True)
plt.suptitle("Polca Analysis of the summation of latent orthogonal components")
plt.show()

# In[44]:


import seaborn as sns

o1 = widgets.Output()
o2 = widgets.Output()

with o1:
    scores = model_iris.score(X)
    sns.displot(scores, kde=True, fill=False, color="black")
    plt.title("Last component with clean data")
    plt.show()

with o2:
    scores = model_iris.score(X * (np.random.random(size=X.shape) - 0.5) * 1)
    sns.displot(scores, kde=True, fill=False, color="black")
    plt.title("Last componet with uniform noise in data")
    plt.show()

layout = widgets.Layout(grid_template_columns="repeat(2, 500px)")
accordion = widgets.GridBox(children=[o1, o2], layout=layout)
display(accordion)

# In[45]:


model_iris.std_metrics

# In[46]:


model_iris.mean_metrics

# ## Test Classification with two components on PCA vs POLCA Net

# In[47]:


import pandas as pd
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# In[48]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# In[49]:


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
X_train_pca.shape, X_test_pca.shape

# In[50]:


# Transform the data using POLCA-Net
X_train_polca = model_iris.predict(X_train, np.array([1, 1, 0, 0]))[0][:, :2]
# X_train_polca = model_iris.predict(X_train)[0][:, :pca.n_components]
X_test_polca = model_iris.predict(X_test, np.array([1, 1, 0, 0]))[0][:, :2]
# X_test_polca = model_iris.predict(X_test)[0][:, :pca.n_components]
X_train_polca.shape, X_test_polca.shape

# In[51]:


# Define classifiers
classifiers = {"Logistic Regression": LogisticRegression(), "Gaussian Naive Bayes": GaussianNB(),
               "Linear SVM": SVC(kernel="linear", probability=True), "Ridge Classifier": RidgeClassifier(),
               "Perceptron": Perceptron(), }

# In[52]:


# Train and evaluate classifiers on both PCA and POLCA-Net transformed datasets
results = []

for name, clf in classifiers.items():
    # Train on PCA
    clf.fit(X_train_pca, y_train)
    y_pred_pca = clf.predict(X_test_pca)
    accuracy_pca = accuracy_score(y_test, y_pred_pca)
    report_pca = classification_report(y_test, y_pred_pca, output_dict=True)

    # Train on POLCA-Net
    clf.fit(X_train_polca, y_train)
    y_pred_polca = clf.predict(X_test_polca)
    accuracy_polca = accuracy_score(y_test, y_pred_polca)
    report_polca = classification_report(y_test, y_pred_polca, output_dict=True)

    # Append results
    results.append({"Classifier": name, "Transformation": "PCA", "Accuracy": accuracy_pca,
                    "Precision": report_pca["weighted avg"]["precision"],
                    "Recall": report_pca["weighted avg"]["recall"],
                    "F1-Score": report_pca["weighted avg"]["f1-score"], })

    results.append({"Classifier": name, "Transformation": "POLCA-Net", "Accuracy": accuracy_polca,
                    "Precision": report_polca["weighted avg"]["precision"],
                    "Recall": report_polca["weighted avg"]["recall"],
                    "F1-Score": report_polca["weighted avg"]["f1-score"], })

# In[53]:


# Create a DataFrame to display the results
results_df = pd.DataFrame(results)
results_df

# In[54]:


# Statistical test: Paired t-test for accuracies
comparison_metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
print(f"\nPaired t-test results:")
for comparison_metric in comparison_metrics:

    print(f"{comparison_metric}:")
    pca_result = results_df[results_df["Transformation"] == "PCA"][comparison_metric]
    polca_result = results_df[results_df["Transformation"] == "POLCA-Net"][comparison_metric]
    t_stat, p_value = ttest_rel(pca_result.values, polca_result.values)
    print(f"\tt-statistic = {t_stat}, p-value = {p_value}, p-value threshold < {0.05}")
    if p_value < 0.05:
        # print(f"There is a statistically significant difference between the PCA and POLCA-Net transformations")
        ans = "a"
    else:
        ans = "no"

    print(f"\tThere is {ans} statistically significant difference between the PCA and POLCA-Net transformations.")

# In[ ]:


# In[ ]:
