# cwru-csds490-project
Course project for CSDS 490 class, Spring 2024

Abstract:
Photovoltaics (PV) have become an increasingly inexpensive source of renewable energy in recent years, largely due to maturing technology and economies of scale. 
As the total amount of solar PV installed exceeds 1 terawatt (as of 2022), quantification of PV performance becomes crucial to manufacturers, insurers, and system operators. 
A commonly used form of nondestructive testing of PVs is electroluminescence (EL) imaging, where a bias is applied to a PV cell or module and the usable area emits light in the near-infrared spectrum, which is captured by a sensitive camera. 
These EL images are commonly used to manually and qualitatively evaluate modules by the failures or flaws they exhibit, commonly cracking, corrosion, or various manufacturing defects. 
However, it is difficult to do this at scale, especially in an assembly line fashion. 
Previous works have explored the possibilities of using convolutional neural networks (CNNs) to automatically classify EL images, but these CNNs require labeled training data in large quantities, and are vulnerable to the imbalanced class problem. 
In this work, we introduce a unsupervised convolutional auto-encoder (CAE) model, and compare it to a classical unsupervised image processing approach. 
We will show that unsupervised approaches can attain very good accuracy scores on multiple EL image datasets without the need for costly training data with labels.

# Usage
## Autoencoder
To run the autoencoder, see `notebooks/bgp-vae-example.ipynb`
## Classical image processing
See `notebooks/demo.py` to run the classical pipeline

