import math
import numpy as np
import networkx as nx
from networkx.algorithms import community
from skimage import io, transform
from skimage.color import rgb2gray
from pytorch_msssim import ssim as ssim_torch
import torch


def process_image_gpu(image_path, im_size=(352, 352), device="cuda"):
    """Loads an image, converts to grayscale, resizes, and sends tensor to GPU."""
    img = io.imread(image_path)
    img_resized = transform.resize(rgb2gray(img), im_size, anti_aliasing=True)
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return img_tensor


def calculate_ssim_similarity_matrix_gpu(image_paths, im_size=(352, 352), device="cuda"):
    """Computes pairwise SSIM similarity matrix using PyTorch SSIM on GPU."""
    num_images = len(image_paths)
    images = [process_image_gpu(p, im_size=im_size, device=device) for p in image_paths]

    similarity_matrix = np.zeros((num_images, num_images), dtype=np.float32)
    for i in range(num_images):
        for j in range(i):
            sim = ssim_torch(images[i], images[j], data_range=1.0).item()
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    np.fill_diagonal(similarity_matrix, 1.0)
    return similarity_matrix


def construct_graph_and_get_communities(
    similarity_matrix,
    cc_th=0.8,
    resolutions=(0.85, 0.9, 0.95, 0.98, 0.99, 1.0, 1.01, 1.05, 1.1, 1.15, 1.2)
):
    """
    Thresholds similarity matrix, constructs the adjacency graph, and finds
    Louvain communities maximizing modularity across resolution levels.
    """
    adj_matrix = similarity_matrix.copy()
    adj_matrix[adj_matrix > cc_th] = 1.0
    adj_matrix[adj_matrix <= cc_th] = 0.0
    np.fill_diagonal(adj_matrix, 0.0)

    node_degrees = np.sum(adj_matrix, axis=1)
    graph = nx.from_numpy_array(adj_matrix)

    best_modularity = -float("inf")
    best_communities = None

    for res in resolutions:
        comms = community.louvain_communities(graph, resolution=res, threshold=1e-7, seed=123)
        mod_score = community.modularity(graph, comms)
        if mod_score > best_modularity:
            best_modularity = mod_score
            best_communities = comms

    return best_communities, node_degrees


def select_prime_indices(communities, node_degrees, per=10):
    """Selects top-degree exemplar indices from each community."""
    selected_indices = []

    for comm in communities:
        current_community = list(comm)
        comm_degrees = node_degrees[current_community]

        num_to_select = math.ceil((len(comm_degrees) * per) / 100.0)
        num_to_select = max(1, num_to_select)

        degrees_tensor = torch.tensor(comm_degrees)
        topk_indices = torch.topk(degrees_tensor, num_to_select).indices.numpy()

        for idx in topk_indices:
            selected_indices.append(current_community[idx])

    return selected_indices


def run_prime_selection(image_paths, cc_th=0.8, per=10, im_size=(352, 352), similarity_matrix=None):
    """
    End-to-end PRIME pruning pipeline.
    Returns the list of selected index positions.
    """
    if similarity_matrix is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        similarity_matrix = calculate_ssim_similarity_matrix_gpu(image_paths, im_size=im_size, device=device)

    communities, node_degrees = construct_graph_and_get_communities(similarity_matrix, cc_th=cc_th)
    selected_ids = select_prime_indices(communities, node_degrees, per=per)
    return selected_ids, communities
