import os
import torch
import nemo.collections.asr as nemo_asr
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained speaker model once upon import
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

def extract_embeddings_for_directory(directory_path):
    """
    Extract embeddings for all WAV files in the specified directory and its subdirectories.

    Args:
        directory_path (str): Path to the directory containing WAV files and subdirectories.

    Returns:
        dict: A dictionary with file paths as keys and their embeddings as values.
    """
    embeddings = {}

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                # Construct the full path to the file
                file_path = os.path.join(root, file)

                # Read the audio file and extract the embedding
                embedding = speaker_model.get_embedding(file_path)

                # Store the embedding with the file path as key
                embeddings[file_path] = embedding.cpu().numpy()

    return embeddings

def calculate_cosine_similarities(embeddings):
    """
    Calculate and print average, variance, and median of intra-folder and inter-folder cosine similarities
    for the given embeddings.

    Args:
        embeddings (dict): Dictionary with file paths as keys and embeddings as values.

    Returns:
        None
    """
    labels = []
    vectors = []

    # Prepare data
    for file_path, embedding in embeddings.items():
        vectors.append(embedding.squeeze())  # Remove unnecessary dimensions
        label = os.path.basename(os.path.dirname(file_path))  # Use folder name as label
        labels.append(label)

    vectors = np.array(vectors)
    cos_sim_matrix = cosine_similarity(vectors)

    unique_labels = set(labels)
    intra_folder_sims = []
    inter_folder_sims = []

    # Calculate intra-folder and inter-folder cosine similarities
    for label in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == label]
        other_indices = [i for i, l in enumerate(labels) if l != label]

        # Intra-folder similarities
        if indices:
            intra_folder_sim = cos_sim_matrix[np.ix_(indices, indices)].mean()
            intra_folder_sims.append(intra_folder_sim)

        # Inter-folder similarities
        if other_indices:
            inter_folder_sim = cos_sim_matrix[np.ix_(indices, other_indices)].mean()
            inter_folder_sims.append(inter_folder_sim)

    # Print statistics for intra-folder similarities
    if intra_folder_sims:
        print(f"Average Intra-Folder Cosine Similarity: {np.mean(intra_folder_sims):.2f}")
        print(f"Variance of Intra-Folder Cosine Similarity: {np.var(intra_folder_sims):.2f}")
        print(f"Median of Intra-Folder Cosine Similarity: {np.median(intra_folder_sims):.2f}")
    else:
        print("No intra-folder similarities computed.")

    # Print statistics for inter-folder similarities
    if inter_folder_sims:
        print(f"Average Inter-Folder Cosine Similarity: {np.mean(inter_folder_sims):.2f}")
        print(f"Variance of Inter-Folder Cosine Similarity: {np.var(inter_folder_sims):.2f}")
        print(f"Median of Inter-Folder Cosine Similarity: {np.median(inter_folder_sims):.2f}")
    else:
        print("No inter-folder similarities computed due to lack of multiple folders.")

