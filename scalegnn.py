import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class ScaleGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, K=3, top_k=10, mask_ratio=0.5):
        super(ScaleGNN, self).__init__()
        self.K = K  # highest neighbor order
        self.top_k = top_k
        self.mask_ratio = mask_ratio

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.linears.append(nn.Linear(hidden_channels, hidden_channels))
        self.linears.append(nn.Linear(hidden_channels, out_channels))

        self.lcs_score_fn = nn.CosineSimilarity(dim=-1)
        self.alpha = nn.Parameter(torch.Tensor(K))  # learnable weights for each order
        nn.init.constant_(self.alpha, 1.0 / K)

        self.fusion = nn.Linear(hidden_channels * 2, hidden_channels)

    def compute_multi_hop_adj(self, edge_index, num_nodes):
        A_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # shape: [N, N]
        A_k_list = [torch.eye(num_nodes, device=A_dense.device)]
        A_power = A_dense.clone()
        for _ in range(1, self.K):
            A_power = torch.matmul(A_power, A_dense)
            A_k_list.append(A_power.clone())
        return A_k_list

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        A_k_list = self.compute_multi_hop_adj(edge_index, num_nodes)  # [K, N, N]

        high_order_feats = []
        for k in range(self.K):
            A_k = A_k_list[k]  # [N, N]
            neighbors = torch.matmul(A_k, x)  # [N, F]
            high_order_feats.append(neighbors.unsqueeze(0))

        high_order_feats = torch.cat(high_order_feats, dim=0)  # [K, N, F]

        fused_feats = torch.zeros_like(x)
        for i in range(num_nodes):
            v_feat = x[i].unsqueeze(0).repeat(num_nodes, 1)  # [N, F]
            sim_scores = self.lcs_score_fn(v_feat, x)  # [N]

            _, topk_idx = torch.topk(sim_scores, self.top_k)
            mask = torch.rand(num_nodes, device=x.device) < self.mask_ratio
            mask[topk_idx] = True

            selected_neighbors = high_order_feats[:, i] * mask.unsqueeze(0).unsqueeze(-1)  # [K, N, F]
            weighted_neighbors = (self.alpha.view(-1, 1, 1) * selected_neighbors).sum(dim=0)  # [N, F]
            fused_feats[i] = weighted_neighbors.sum(dim=0) / (mask.sum() + 1e-6)

        out = torch.cat([x, fused_feats], dim=-1)
        out = self.fusion(out)

        for i, lin in enumerate(self.linears[:-1]):
            out = F.relu(lin(out))
            out = F.dropout(out, p=0.5, training=self.training)
        out = self.linears[-1](out)
        return F.log_softmax(out, dim=1)

    def lcs_loss(self, x):
        N = x.size(0)
        sim_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)  # [N, N]
        topk_vals, _ = sim_matrix.topk(self.top_k, dim=1)
        loss = (1 - topk_vals).pow(2).mean()
        return loss
