import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.special import kl_div
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances, silhouette_score, calinski_harabasz_score
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataloader import *
import torch.nn as nn
import scipy.sparse as sp

L2norm = nn.functional.normalize


def get_knn_graph(data, k):
    num_samples = data.size(0)
    graph = torch.zeros(num_samples, num_samples, dtype=torch.int32, device=data.device)

    for i in range(num_samples):
        distance = torch.sum((data - data[i]) ** 2, dim=1)
        _, small_indices = torch.topk(distance, k + 1, largest=False)
        graph[i, small_indices[1:]] = 1  # 1

    result_graph = torch.max(graph, graph.t())

    return result_graph


def fetch_normalization(type):
    switcher = {
        'AugNormAdj': aug_normalized_adjacency,  # A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
        'NormAdj': normalized_adjacency,  # A' = (D)^-1/2 * ( A) * (D)^-1/2
    }
    func = switcher.get(type, lambda: "Invalid normalization technique.")
    return func


def normalized_adjacency(adj):
    adj = adj
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def save_results(args, best_acc, best_nmi, best_ari, best_pur):
    subdirectory = 'results'
    filename = '%s.txt' % args.dataset
    file_path = os.path.join(os.getcwd(), subdirectory, filename)
    with open(file_path, 'a+') as f:
        f.write('{} \t {} \t {}\t {} \t {}\t {} \t {} \t {}  \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \n'.format(
            args.high_feature_dim, args.feature_dim, args.batch_size, args.learning_rate,
            args.lamda1, args.lamda2, args.lamda3, args.lamda4, args.lamda5, best_acc, best_nmi, best_ari, best_pur))
        f.flush()


def get_cluster_centers_(model, device, loader, view, class_num):
    model.eval()
    for batch_idx, (xs, _, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            hs, qs, xrs, zs, H, H_pre, pred = model(xs)
        kmeans = KMeans(n_clusters=class_num, n_init=100).fit(H.cpu().detach().numpy())

    return kmeans.cluster_centers_


def visual(model, device, dataset, view, data_size, class_num, args):
    t = time.time()
    data_loader = DataLoader(
        dataset,
        batch_size=data_size,
        shuffle=False,
    )
    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)

    labels = y.cpu().detach().data.numpy().squeeze()

    with torch.no_grad():
        hs, qs, xrs, zs, H, H_pre, pred = model(xs)

    H = H.cpu()
    data_np = H.numpy()

    tsne = TSNE(n_components=2, random_state=10, perplexity=20, learning_rate=1500, n_iter=6000)
    X_tsne = tsne.fit_transform(data_np)

    plt.figure(figsize=(16, 10))
    for i in range(class_num):
        plt.scatter(X_tsne[labels == i, 0], X_tsne[labels == i, 1], marker='o', alpha=0.8)

    plt.legend(fontsize=14)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    ax = plt.gca()

    # # 移除边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # # 移除轴线和刻度
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # # 完全移除轴
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # # 移除网格
    ax.grid(False)

    folder_path = 'Visualization'
    file_name = '%s.png' % args.dataset
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    # # 保存图片到文件
    plt.savefig(os.path.join(folder_path, file_name))
    plt.close()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    last_point = points[0]
    smoothed_points.append(last_point)
    for point in points[1:]:
        smoothed_point = last_point * factor + point * (1 - factor)
        smoothed_points.append(smoothed_point)
        last_point = smoothed_point
    return smoothed_points

