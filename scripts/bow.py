#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from skimage.feature import daisy
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import glob
import pandas as pd
import mahotas.features as mf
cv2.ocl.setUseOpenCL(False)


#%% Functions


def initialize_feature_detector(feature_type):
    if feature_type == 'kaze':
        return cv2.KAZE_create(), cv2.KAZE_create()
    elif feature_type == 'orb':
        return cv2.ORB_create(), cv2.ORB_create()
    elif feature_type == 'fast':
        return cv2.FastFeatureDetector_create(), cv2.ORB_create()
    elif feature_type == 'daisy':
        return None, None
    else:
        raise ValueError("Unsupported feature type provided.")
        
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    blurred_image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    return blurred_image


def extract_features(image, detector, computer):
    if detector == computer == None:
        descriptors = daisy(image, normalization='l2')
        descriptors = descriptors.reshape(-1, descriptors.shape[-1])
    else:
        keypoints = detector.detect(image, None)
        _, descriptors = computer.compute(image, keypoints)
    return descriptors

def compute_global_features(image, length_mult=2):
    vector = []
    for i in range(0,len(image),length_mult):
        tmp = 0
        for j in range(len(image)):
            tmp += int(image[i][j])+ int(image[i+1][j])
        vector.append(tmp)
    return np.array(vector)/np.linalg.norm(np.array(vector))

def compute_haralick_features(image):
    features = mf.haralick(image, return_mean=True, compute_14th_feature=True)
    feature_0 = features[0]  # 0 degrees
    feature_45 = features[1]  # 45 degrees
    feature_90 = features[2]  # 90 degrees
    feature_135 = features[3]  # 135 degrees
    return feature_0, feature_45, feature_90, feature_135

def aggregate_descriptors(descriptors):
    return np.vstack(descriptors)

def cluster_descriptors(descriptors, num_clusters):
    kmeans = MiniBatchKMeans(n_clusters=num_clusters)
    kmeans.fit(descriptors)
    return kmeans

def construct_histograms(descriptors, kmeans_model, hist_len):
    histograms = []
    for descriptor in descriptors:
        hist = np.zeros(hist_len)
        if descriptor is not None:
            indices = kmeans_model.predict(descriptor)
            for i in indices:
                hist[i] += 1
        histograms.append(hist)
    return histograms

def assemble_feature_vectors(histograms, global_features, haralick_features, use_custom, use_haralick):
    feature_vectors = []
    for i, hist in enumerate(histograms):
        l2_norm = hist / np.linalg.norm(hist)
        feature_vector = l2_norm
        if use_custom:
            feature_vector = np.append(feature_vector, global_features[i])
        if use_haralick:
            feature_vector = np.append(feature_vector, haralick_features[i])
        feature_vectors.append(feature_vector)
    return feature_vectors

def constructBoW(filepath, feature='kaze', clusters=3, hist_len=100, global_vec_len=125, haralick=False, custom=True):

    detector, descriptor = initialize_feature_detector(feature)
    
    image_paths = glob.glob(f'{filepath}/**/*.jpg', recursive=True)
    
    # Data structures for storing descriptors and additional features
    all_descriptors = []
    all_global_features = []
    all_haralick_features = []
    names = []
    failures = []
    
    # Process each image
    for img_path in image_paths:
        if "Blank" in img_path:
            continue
        
        image = preprocess_image(img_path)
        descriptors = extract_features(image, detector, descriptor)
        
        if descriptors is None:
            print(f"Failed to compute descriptors for image: {img_path}")
            failures.append(img_path)
            continue
        
        all_descriptors.append(descriptors)
        names.append(img_path)
        
        if custom:
            global_features = compute_global_features(image, global_vec_len)
            all_global_features.append(global_features)
        
        if haralick:
            haralick_features = compute_haralick_features(image)
            all_haralick_features.append(haralick_features)
    

    aggregated_descriptors = aggregate_descriptors(all_descriptors)
    kmeans_model = cluster_descriptors(aggregated_descriptors, hist_len)
    
    histograms = construct_histograms(all_descriptors, kmeans_model, hist_len)
    
    feature_vectors = assemble_feature_vectors(histograms, all_global_features, all_haralick_features, custom, haralick)
    
    bow = np.vstack(feature_vectors)
    
    print(f"Processed {len(names)} images with {len(failures)} failures.")
    
    return bow, names, failures

def fitBoW(bow, names=None, pca=(True,2), clusters=3, model_type='agg'):
    svd = None
    if pca[0]:
        svd = PCA(n_components=pca[1], svd_solver='full')
        svd.fit(bow)
        bow = svd.transform(bow)
        
    if model_type == 'agg':
        model = AgglomerativeClustering(n_clusters=clusters)
    elif model_type == 'gmm':
        model = GaussianMixture(n_components=clusters).fit(bow)
    elif model_type == 'spectral':
        model = SpectralClustering(n_clusters=clusters, eigen_solver='arpack', affinity="nearest_neighbors")
    else:
        raise ValueError("Unsupported model type. Choose 'agg', 'gmm', or 'spectral'.")
    
    return model, svd

def predictBoW(model, bow, names, pca=(True,2,None), csv_path="test.csv", model_type='agg'):
    if pca[2] is not None:
        bow = pca[2].transform(bow)

    predictions = None
    if model_type == 'agg':
        predictions = model.fit_predict(bow)
    elif model_type == 'gmm':
        predictions = model.predict(bow)
    elif model_type == 'spectral':
        predictions = model.fit_predict(bow)
    else:
        raise ValueError("Unsupported model type. Choose 'agg', 'gmm', or 'spectral'.")

    df = pd.DataFrame(list(zip(names, list(predictions))), columns=['filename', 'class'])
    df.to_csv(csv_path, index=False) 

    return df


def evaluate(df,clusters=3):
    actual = df['filename'].str.split('/').str[-2].str[-1].astype(int)
    preds = df['class']

    if clusters == 2:
        for idx, label in enumerate(actual):
            actual[idx] = 0 if label < 2 else 1
    
    f1 = f1_score(actual, preds, average='macro')
    accuracy = accuracy_score(actual, preds)
    cm = confusion_matrix(actual, preds)
    return accuracy, f1, cm
    
























