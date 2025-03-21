import pandas as pd
from numpy import dtype, ndarray
from pandas import DataFrame, Series
from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.cm as cm
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans


def compute_pca_and_kmeans(
        df: pd.DataFrame,
        n_clusters_biplot: int = 2,
        n_clusters_kmeans: int = 2,
        n_components_kmeans: Optional[int] = None,
        figsize_pca: Optional[Tuple[int, int]] = (8, 8),
        figsize_biplot: Optional[Tuple[int, int]] = (8, 8),
        figsize_kmeans: Optional[Tuple[int, int]] = (8, 8),
) -> tuple[
    pd.DataFrame, pd.DataFrame, Union[np.ndarray, Any], Any, pd.Series]:
    """
    Perform Principal Component Analysis (PCA) on a standardized dataset,
    returning PCA results, component loadings, cumulative variance,
    and explained variance. This function includes scree plot, biplot,
    and K-Means clustering visualizations, using a notebook-friendly theme.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing the data to be analyzed, where rows represent observations
        and columns represent features/variables.
    n_clusters_biplot : int, optional (default=2)
        Number of clusters for the biplot to identify clusters among the first two principal components.
    n_clusters_kmeans : int, optional (default=2)
        Number of clusters for K-Means clustering on the PCA-transformed data.
    n_components_kmeans : int, optional
        Number of principal components to use for clustering. If None, all components are used.
    figsize_pca : tuple of int, optional (default=(8, 8))
        Figure size for the scree plot.
    figsize_biplot : tuple of int, optional (default=(8, 8))
        Figure size for the biplot.
    figsize_kmeans : tuple of int, optional (default=(8, 8))
        Figure size for the K-Means cluster plot.

    Returns
    -------
    tuple
        A tuple containing:
        - pca_components: DataFrame with principal component scores for each observation.
        - loadings: DataFrame of PCA loadings for each original variable.
        - cumulative_variance: Array with the cumulative explained variance.
        - explained_variance: Array with the explained variance ratio per component.
        - series_clusters: Series of cluster labels assigned to each observation.
    """

    # Set a notebook-friendly theme and style using Seaborn
    sns.set_theme(context='notebook', style='whitegrid', palette='deep')

    # Step 1: Copy and Standardize Data
    df_pca = df.copy()
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_pca)

    # Step 2: PCA Fitting and Transformation
    pca = PCA()
    pca_result = pca.fit_transform(df_standardized)

    # Step 3: Variance Analysis
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Step 4: Component Loadings Interpretation
    loadings = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f'PC{i + 1}' for i in range(len(df.columns))]
    )

    # Step 5: K-Means Clustering on PCA Components
    kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
    if n_components_kmeans is None:
        cluster_labels = kmeans.fit_predict(pca_result)
    else:
        cluster_labels = kmeans.fit_predict(pca_result[:, :n_components_kmeans])

    # Step 6: Scree Plot using notebook-friendly aesthetics
    plt.figure(figsize=figsize_pca)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'o-', markersize=6,
             label='Individual Explained Variance')
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 's-', markersize=6,
             label='Cumulative Explained Variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot of Principal Components')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.tight_layout()
    plt.show()

    # Step 7: Biplot of the first two principal components
    # Assumes the existence of the 'plot_clustered_biplot' function.
    plot_clustered_biplot(pca=pca,
                          pca_result=pca_result,
                          vars_of_interest=[*df.columns],
                          n_clusters=n_clusters_biplot,
                          figsize=figsize_biplot,
                          arrow_fontsize=16,
                          legend_fontsize=18)
    # Step 8: K-Means Cluster Plot on First Two Principal Components
    plt.figure(figsize=figsize_kmeans)
    # set1_colors = plt.cm.tab10.colors[0]
    tab10_colors = ['#1f77b4', '#ff7f0e']  # Define colors for the clusters
    # Create a mapping from each unique label to its color
    unique_labels = np.unique(cluster_labels)
    color_map = {label: color for label, color in zip(unique_labels, tab10_colors)}
    # Assign a color to each point based on its cluster label
    colors_assigned = [color_map[label] for label in cluster_labels]
    plt.scatter(pca_result[:, 0], pca_result[:, 1],
                          # c=cluster_labels,
                          c=colors_assigned,
                          # cmap= colors_assigned,  #'tab10',
                          s=50,
                          alpha=0.7)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X', label='Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'K-Means Clustering on First Two Principal Components with {n_clusters_kmeans} Clusters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    pca_components = pd.DataFrame(pca_result, columns=[f'PC{i + 1}' for i in range(pca_result.shape[1])])
    series_clusters = pd.Series(cluster_labels, index=df.index, name='Cluster')
    return (
        pca_components,
        loadings,
        cumulative_variance,
        explained_variance,
        series_clusters
    )


def plot_clustered_biplot(pca,
                          pca_result,
                          vars_of_interest,
                          n_clusters=3,
                          figsize: Tuple[int, int] = (10, 7),
                          legend_fontsize: Optional[int] = 12,
                          arrow_fontsize: Optional[int] = 10):
    """
    Creates a biplot of the first two principal components with variable loadings grouped by cluster colors.

    Parameters:
    pca (PCA): Fitted PCA model.
    pca_result (array): PCA transformed data.
    vars_of_interest (list): List of variable names corresponding to loadings.
    n_clusters (int): Number of clusters for grouping similar loadings.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, label="Data Points")

    # Extract the first two loading vectors for clustering
    loadings = pca.components_[:2].T

    # Perform hierarchical clustering on loadings
    linkage_matrix = linkage(loadings, method='ward')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    # Define a colormap for the clusters
    colors = cm.get_cmap('tab10', n_clusters)

    # Dictionary for variable mapping with clustering
    variable_mapping = {i + 1: var for i, var in enumerate(vars_of_interest)}

    # Plot arrows with colors based on clusters and annotate with numbers
    arrow_handles = []
    for i, (num, var) in enumerate(variable_mapping.items()):
        cluster_idx = clusters[i] - 1
        color = colors(cluster_idx)

        # Draw arrow with color based on cluster
        arrow = ax.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3,
                         color=color, head_width=0.15, head_length=0.15, alpha=0.8)
        ax.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2, str(num),
                color='black', ha='center', va='center', fontsize=arrow_fontsize, fontweight='bold')

        # Store arrow for legend with variable name and number
        arrow_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=f'{num}: {var}'))

    # Create legend with variable names and numbers
    legend = ax.legend(handles=arrow_handles,
                       loc='upper left', bbox_to_anchor=(1.05, 1),
                       fontsize=legend_fontsize, title='Variables',
                       borderaxespad=0.1, framealpha=0.8)

    # Labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('Clustered Biplot of Symptoms on First Two Principal Components')
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(right=0.65)  # Increased space for legend on the right
    plt.show()


# def plot_clustered_biplot(pca,
#                           pca_result,
#                           vars_of_interest,
#                           n_clusters=3,
#                           figsize: Tuple[int, int] = (10, 7),
#                           legend_fontsize:Optional[int] = 18,
#                           arrow_fontsize:Optional[int] = 10, ):
#     """
#     Creates a biplot of the first two principal components with variable loadings grouped by cluster colors.
#
#     Parameters:
#     pca (PCA): Fitted PCA model.
#     pca_result (array): PCA transformed data.
#     vars_of_interest (list): List of variable names corresponding to loadings.
#     n_clusters (int): Number of clusters for grouping similar loadings.
#
#     """
#     # sns.set_theme(context='notebook', style='whitegrid', palette='deep')
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, label="Data Points")
#
#     # Extract the first two loading vectors for clustering
#     loadings = pca.components_[:2].T  # Only take the loadings on the first two PCs
#
#     # Perform hierarchical clustering on loadings
#     linkage_matrix = linkage(loadings, method='ward')
#     clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
#
#     # Define a colormap for the clusters
#     colors = cm.get_cmap('tab10', n_clusters)
#
#     # Dictionary for variable mapping with clustering
#     variable_mapping = {i + 1: var for i, var in enumerate(vars_of_interest)}
#
#     # Plot arrows with colors based on clusters and annotate with numbers
#     for i, (num, var) in enumerate(variable_mapping.items()):
#         cluster_idx = clusters[i] - 1  # Adjust for 0-based index in color map
#         color = colors(cluster_idx)
#
#         # Draw arrow with color based on cluster
#         ax.arrow(0, 0, pca.components_[0, i] * 3, pca.components_[1, i] * 3,
#                  color=color, head_width=0.15, head_length=0.15, alpha=0.8)
#         ax.text(pca.components_[0, i] * 3.2, pca.components_[1, i] * 3.2, str(num),
#                 color='black', ha='center', va='center', fontsize=arrow_fontsize, fontweight='bold')
#
#     # Add a legend for variable mapping with a transparent background
#     text_box = "\n".join([f"{num}: {var}" for num, var in variable_mapping.items()])
#     plt.gcf().text(0.85, 0.5, text_box, fontsize=legend_fontsize, ha='left', va='center',
#                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="none"))
#
#     # Labels and title
#     ax.set_xlabel('Principal Component 1')
#     ax.set_ylabel('Principal Component 2')
#     ax.set_title('Clustered Biplot of Symptoms on First Two Principal Components')
#     plt.grid(True)
#     plt.tight_layout(rect=[0, 0, 0.95, 1])
#     plt.show()
#
