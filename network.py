import numpy as np
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
from typing import Optional
from torch.nn import Parameter


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Clustering(nn.Module):
    def __init__(self, num_cluster, cluster_hidden_dim):
        super(Clustering, self).__init__()
        self.kmeans = KMeans(n_clusters=num_cluster, n_init=40)
        self.initialized = True
        self.clustering_layer = DECModule(cluster_number=num_cluster,
                                          embedding_dimension=cluster_hidden_dim
                                          )

    def forward(self, h):
        if not self.initialized:
            self.kmeans.fit(h.cpu().detach().numpy())
            cluster_centers = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float)
            with torch.no_grad():
                self.clustering_layer.cluster_centers.copy_(cluster_centers)
            self.initialized = True

        clustering_layer = self.clustering_layer
        q = clustering_layer(h)
        return q


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, num_samples, device):
        super(Network, self).__init__()
        self.view = view
        self.num_samples = num_samples
        self.class_num = class_num
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )

        self.label_contrastive_module1 = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.label_contrastive_module2 = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.feature_fusion_module = nn.Sequential(
            nn.Linear(self.view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        self.clustering1 = Clustering(class_num, feature_dim)
        self.clustering2 = Clustering(class_num, high_feature_dim)

    def feature_fusion(self, zs, zs_gradient):
        input = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        return normalize(self.feature_fusion_module(input), dim=1)

    def forward(self, xs, zs_gradient=True):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module1(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)

        H = self.feature_fusion(zs, zs_gradient)
        H_pre = self.label_contrastive_module2(H)
        pred = torch.argmax(H_pre, dim=1)

        return hs, qs, xrs, zs, H, H_pre, pred

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module1(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds


class DECModule(nn.Module):
    def __init__(
            self,
            cluster_number: int,
            embedding_dimension: int,
            alpha: float = 1.0,
            cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:

        super(DECModule, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha

        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:

        norm_squared = torch.sum((batch.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Returns the current cluster centers.
        """
        return self.cluster_centers.detach().cpu()  # Detach and move to CPU for standalone use
