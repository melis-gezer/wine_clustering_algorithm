# Wine Clustering Algorithm

This project applies unsupervised learning to a small wine dataset in order to discover natural groups of wines based on their chemical properties. The main goals are to practice end-to-end clustering workflow and to build a clean, reproducible example for a data science portfolio.

## Dataset

The data comes from the **"Wine Dataset for Clustering"** by Harry Wang on Kaggle:  
https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering

The dataset contains 178 wines with 13 numerical features (e.g. Alcohol, Malic Acid, Color Intensity, Proline), commonly used as a benchmark for clustering and dimensionality reduction.

## Methods (Overview)

- Standardize all numeric features using `StandardScaler`.
- Apply **K-Means** clustering for different values of *k* and inspect the **elbow curve**.
- Train a final K-Means model (k = 3) on the scaled features.
- Use **PCA (2 components)** to project the data to 2D and visualize the clusters.
- Analyze cluster sizes and the average feature values per cluster to interpret the segments.
