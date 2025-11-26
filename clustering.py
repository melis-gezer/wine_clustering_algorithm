import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path


def load_data():
    """Load the wine CSV using a relative path."""
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "wine-clustering.csv"

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print("Data loaded. Shape:", df.shape)
    return df


def basic_eda(df: pd.DataFrame):
    """Print basic info and numeric summary statistics."""
    print("\n=== DataFrame info ===")
    print(df.info())
    print("\n=== Numeric summary (describe) ===")
    print(df.describe().T)
    print("\n=== Missing values per column ===")
    print(df.isna().sum().sort_values(ascending=False))


def scale_features(df: pd.DataFrame):
    """
    Select numeric features for clustering and scale them.
    şu an tüm kolonları alıyor, istersen filtreleyebilirsin.
    """
    feature_cols = df.columns.tolist()
    X = df[feature_cols].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    print("\nUsing the following features for clustering:")
    print(feature_cols)

    return scaled_df, feature_cols


def elbow_method(X_scaled: np.ndarray, max_k: int = 20):
    k_values = list(range(1, max_k + 1))
    inertias = []

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    print("\n=== Inertia values for each k ===")
    for k, inertia in zip(k_values, inertias):
        print(f"k = {k}: inertia = {inertia:.2f}")

    plt.figure()
    plt.plot(k_values, inertias, marker="o")
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow curve for KMeans on wine data")
    plt.tight_layout()
    plt.show()


def train_final_kmeans(X_scaled: np.ndarray, n_clusters: int = 3):
    """
    Final KMeans training.
    df'e burada dokunmuyoruz, sadece labels ve modeli döndürüyoruz.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def inspect_cluster_sizes(labels: np.ndarray):
    """Print how many samples fall into each cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    print("\n=== Cluster size distribution ===")
    for lab, cnt in zip(unique, counts):
        print(f"Cluster {lab}: {cnt} samples")


def plot_clusters(scaled_df: pd.DataFrame, labels: np.ndarray):
    """
    PCA ile 2D'ye indirip cluster'ları çiziyoruz.
    scaled_df ve labels'tan kendi plot_df'imizi oluşturuyoruz.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(scaled_df.values)

    plot_df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "cluster": labels
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="viridis"
    )
    plt.title("Wine Clusters (PCA 2D Projection)")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()


def main():
    # 1. Load data
    df = load_data()

    # 2. Basic EDA
    basic_eda(df)

    # 3. Scale features for clustering
    scaled_df, feature_cols = scale_features(df)

    # 4. Elbow method (optional – visual inspection)
    elbow_method(scaled_df.values, max_k=15)

    # 5. Train final KMeans model (set k based on elbow, here k=3)
    labels, kmeans_model = train_final_kmeans(scaled_df.values, n_clusters=3)

    # 6. Attach labels back to original dataframe
    df["cluster_kmeans"] = labels

    # 7. Inspect cluster sizes
    inspect_cluster_sizes(labels)

    # 8. Visualization (using scaled features)
    plot_clusters(scaled_df, labels)

    # 9. Optionally, save the clustered data to a new CSV
    output_path = Path(__file__).resolve().parent / "data" / "wine-clustering.csv"
    df.to_csv(output_path, index=False)
    print(f"\nClustered data saved to: {output_path}")


if __name__ == "__main__":
    main()
