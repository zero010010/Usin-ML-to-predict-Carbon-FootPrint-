import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns   
import os
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN


from sklearn.cluster import KMeans 
import pickle
print(os.getcwd()) 
# Initialize new KMeans model
kmeans = KMeans(n_clusters=5)

# Load model parameters 
with open('kmeans_model.pkl', 'rb') as f:
    notebooks = pickle.load(f) 

# Save model to file
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
    
# Use model to make predictions
y_pred = kmeans.predict(new_data)
