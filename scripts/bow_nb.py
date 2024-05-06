#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scripts.bow as bow
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#%%
feature_types = ['daisy','fast','orb','kaze']
clusters = [2]
haralicks = [True, False]
model_types = ['gmm','agg','spectral']
use_pcas = [True,False]
dim_red = [2,5,8,12]
customs = [True, False]
train_path = "../data/1907French-etal-SDLE-EL-ImageDataSet-forML/train/"
test_path = "../data/1907French-etal-SDLE-EL-ImageDataSet-forML/test/"

models = []

#%%
for feature_type in feature_types:
    for cluster in clusters:
        for haralick in haralicks:
            for custom in customs:
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
                for model_type in model_types:
                    for use_pca in use_pcas:
                        for dim in dim_red:
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
                            acc, f1, _ = bow.evaluate(df,clusters=cluster)
                            row = {
                                'feature_type':feature_type,
                                'num_clusters':cluster,
                                'use_haralick':haralick,
                                'use_custom':custom,
                                'use_pca':use_pca,
                                'dimension_reduction':dim,
                                'cluster_method':model_type,
                                'accuracy':acc,
                                'f1_score':f1,
                                'num_train':len(train_names),
                                'num_tested':len(test_names)
                                }
                            models.append(row)
                    

all_models = pd.DataFrame(models)
all_models.to_csv('model_results.csv',index=False)
#%%

feature_type = 'fast'
cluster=2
haralick=True
custom=True
use_pca=True
dim=2
model_type='gmm'

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
acc, f1, cm = bow.evaluate(df,clusters=cluster)


#%%
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

train_pcs = svd.transform(train_bow)
gmm = GaussianMixture(n_components=2).fit(train_pcs[:,:2])
test_pcs = svd.transform(test_bow)

plt.scatter(train_pcs[:,0],train_pcs[:,1],label='Training Data')
plt.scatter(test_pcs[:, 0], test_pcs[:, 1],c='green',label='Testing Data')
#%%
from matplotlib.patches import Ellipse
import numpy as np

# Plot the PCA-reduced data
plt.scatter(test_pcs[:, 0], test_pcs[:, 1], alpha=0.5, label='PCA Data')

# Plot the means and add ellipses
for mean, cov in zip(gmm.means_, gmm.covariances_):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    vx, vy = eigenvectors[:,0][0], eigenvectors[:,0][1]
    theta = np.arctan2(vy, vx)
    for n in range(1, 4):  # Plot 1-std, 2-std,
        width, height = 2 * n * np.sqrt(eigenvalues)
        ell = Ellipse(xy=mean, width=width, height=height, angle=np.degrees(theta), edgecolor='red', facecolor='none', label=f'{n}-std dev' if mean is gmm.means_[0] and n == 1 else "")
        plt.gca().add_patch(ell)

# Plot the means of the Gaussian components
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', label='GMM Means', s=100, marker='x')

# Adding labels, title, and legend
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Reduction with GMM Means and Standard Deviations')
plt.legend()

# Show the plot
plt.show()



