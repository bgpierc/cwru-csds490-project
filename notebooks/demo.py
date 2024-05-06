#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scripts.bow as bow
import warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
#%% helper function

def plot_ellipses(gmm):
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        vx, vy = eigenvectors[:,0][0], eigenvectors[:,0][1]
        theta = np.arctan2(vy, vx)
        for n in range(1, 3):
            width, height = 2 * n * np.sqrt(eigenvalues)
            ell = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(theta), edgecolor='red', facecolor='none', label=f'{n}-std dev' if mean is gmm.means_[0] and n == 1 else "")
            plt.gca().add_patch(ell)

#%% set parameters

feature_type = 'fast'
cluster=2
haralick=True
custom=True
use_pca=True
dim=2
model_type='gmm'

train_path = "../data/1907French-etal-SDLE-EL-ImageDataSet-forML/train/"
test_path = "../data/1907French-etal-SDLE-EL-ImageDataSet-forML/test/"


#%% train the algorithm

train_bow, train_names, train_failures = bow.constructBoW(filepath=train_path,
                                                feature=feature_type,
                                                clusters=cluster,
                                                haralick=haralick,
                                                custom=custom)
test_bow, test_names, test_failures = bow.constructBoW(filepath=test_path,
                                                feature=feature_type,
                                                clusters=cluster,
                                                haralick=haralick,
                                                custom=custom)
model, svd = bow.fitBoW(train_bow,
                        names=train_names,
                        pca=(use_pca,dim),
                        clusters=cluster,
                        model_type=model_type)
df = bow.predictBoW(model=model,
                    bow=test_bow,
                    names=test_names,
                    pca=(use_pca,dim,svd),
                    model_type=model_type)

#%%  Get PCs and view them

classes = ['blue' if int(x.split('/')[-2][-1]) < 2 else 'green' for x in train_names]

train_pcs = svd.transform(train_bow)
plt.scatter(train_pcs[:,0],train_pcs[:,1],label='Training Data', c=classes)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("Training Set Transformed")


#%% fit the GMM to the data
gmm = GaussianMixture(n_components=2).fit(train_pcs)

plt.scatter(train_pcs[:, 0], train_pcs[:, 1])
plot_ellipses(gmm)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', s=100, marker='x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("Training Set Fit")

#%%
test_pcs = svd.transform(test_bow)
plt.scatter(test_pcs[:, 0], test_pcs[:, 1],c='green',label='Testing Data')
plot_ellipses(gmm)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', label='GMM Means', s=100, marker='x')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title("Test Set Transformed")


#%%
preds = gmm.predict(test_pcs)
p_classes = ['blue' if int(x.split('/')[-2][-1]) < 2 else 'green' for x in test_names]

pred_classes = ['blue' if x < 1 else 'green' for x in preds]
plt.scatter(test_pcs[:, 0], test_pcs[:, 1],c=pred_classes)
plot_ellipses(gmm)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', label='GMM Means', s=100, marker='x')
plt.title('Test Set Model Predictions')

#%%
y_true = [0 if x == 'green' else 1 for x in p_classes]
y_pred = [0 if x == 'green' else 1 for x in pred_classes]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')

#4fp and 1fn
#%%
# Helper function to plot ellipses for GMM components
def plot_ellipses(gmm, ax):
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        vx, vy = eigenvectors[:,0][0], eigenvectors[:,0][1]
        theta = np.arctan2(vy, vx)
        for n in range(1, 3):
            width, height = 2 * n * np.sqrt(eigenvalues)
            ell = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(theta), edgecolor='red', facecolor='none')
            ax.add_patch(ell)

train_pcs = svd.transform(train_bow)
test_pcs = svd.transform(test_bow)
gmm = GaussianMixture(n_components=2).fit(train_pcs)

# Setup a subplot grid
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Training set with GMM ellipses
classes = ['blue' if int(name.split('/')[-2][-1]) < 2 else 'green' for name in train_names]
good_mask = [cls == 'green' for cls in classes]
defective_mask = [cls == 'blue' for cls in classes]

axs[0].scatter(train_pcs[good_mask, 0], train_pcs[good_mask, 1], c='green', label='Good')
axs[0].scatter(train_pcs[defective_mask, 0], train_pcs[defective_mask, 1], c='blue', label='Defective')
plot_ellipses(gmm, axs[0])
axs[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', s=100, marker='x')
axs[0].set_xlabel('PC1')
axs[0].set_ylabel('PC2')
axs[0].set_title("Training Set Fit")
axs[0].set_xlim(-600, 2500)
axs[0].set_ylim(-15, 20)
axs[0].legend()

# Scatter plot of test data predictions
preds = gmm.predict(test_pcs)
pred_classes = ['green' if x < 1 else 'blue' for x in preds]
good_mask = [cls == 'green' for cls in pred_classes]
defective_mask = [cls == 'blue' for cls in pred_classes]

axs[1].scatter(test_pcs[good_mask, 0], test_pcs[good_mask, 1], c='green', label='Good')
axs[1].scatter(test_pcs[defective_mask, 0], test_pcs[defective_mask, 1], c='blue', label='Defective')
plot_ellipses(gmm, axs[1])
axs[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', s=100, marker='x')
axs[1].set_xlabel('PC1')
axs[1].set_ylabel('PC2')
axs[1].set_title("Test Set Model Predictions")
axs[1].set_xlim(-600, 2500)
axs[1].set_ylim(-15, 20)
axs[1].legend()

plt.tight_layout()
plt.show()



