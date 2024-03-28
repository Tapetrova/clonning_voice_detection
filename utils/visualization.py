import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np

def plot_embeddings_with_pca(embeddings):
    """
    Apply Principal Component Analysis (PCA) to reduce the dimensionality of the embeddings
    and plot the reduced embeddings using a scatter plot.

    Args:
        embeddings (dict): A dictionary with file paths as keys and embeddings as values.

    Plots:
        A scatter plot of the PCA-reduced embeddings.
    """
    vectors = np.array([embedding.squeeze() for embedding in embeddings.values()])
    labels = [os.path.basename(os.path.dirname(path)) for path in embeddings.keys()]

    # Perform PCA to reduce the embeddings to 2 dimensions
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)

    # Plot the reduced vectors
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_vectors[indices, 0], reduced_vectors[indices, 1], label=label, s=100)
    plt.title('PCA of Audio Embeddings')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_embeddings_with_tsne(embeddings):
    """
    Apply t-Distributed Stochastic Neighbor Embedding (t-SNE) to reduce the dimensionality of
    the embeddings and plot the reduced embeddings using a scatter plot.

    Args:
        embeddings (dict): A dictionary with file paths as keys and embeddings as values.

    Plots:
        A scatter plot of the t-SNE reduced embeddings.
    """
    vectors = np.array([embedding.squeeze() for embedding in embeddings.values()])
    labels = [os.path.basename(os.path.dirname(path)) for path in embeddings.keys()]

    # Perform t-SNE to reduce the embeddings to 2 dimensions
    tsne = TSNE(n_components=2, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)

    # Plot the reduced vectors
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_vectors[indices, 0], reduced_vectors[indices, 1], label=label, s=100)
    plt.title('t-SNE of Audio Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_embeddings_with_umap(embeddings):
    """
    Apply Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality of
    the embeddings and plot the reduced embeddings using a scatter plot.

    Args:
        embeddings (dict): A dictionary with file paths as keys and embeddings as values.

    Plots:
        A scatter plot of the UMAP reduced embeddings.
    """
    vectors = np.array([embedding.squeeze() for embedding in embeddings.values()])
    labels = [os.path.basename(os.path.dirname(path)) for path in embeddings.keys()]

    # Perform UMAP to reduce the embeddings to 2 dimensions
    reducer = umap.UMAP()
    reduced_vectors = reducer.fit_transform(vectors)

    # Plot the reduced vectors
    plt.figure(figsize=(10, 7))
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_vectors[indices, 0], reduced_vectors[indices, 1], label=label, s=100)
    plt.title('UMAP of Audio Embeddings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.show()
