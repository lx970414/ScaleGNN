import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class ScaleGNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2, dropout=0.5, num_hops=3, top_k=8, mask_ratio=0.5):
        super(ScaleGNN, self).__init__()
        self.num_hops = num_hops  
        self.top_k = top_k  
        self.mask_ratio = mask_ratio  
        self.dropout = dropout  

        self.input_proj = nn.Linear(in_channels, out_channels)
        self.output_proj = nn.Linear(out_channels, out_channels)

        self.alpha = nn.Parameter(torch.ones(num_hops))
        
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, adj_list):
        """
        x: 节点特征 (N, F)，F 是特征维度
        adj_list: 每阶邻接矩阵的列表 [A1, A2, ..., Ak]，每阶是稀疏tensor或dense matrix
        """

        H_low = self.input_proj(x)

        high_features = []

        for k, Ak in enumerate(adj_list):
            Ak = Ak.to_dense() if Ak.is_sparse else Ak
            alpha_k = F.softmax(self.alpha, dim=0)[k]

            # LCS
            sim = F.normalize(x, dim=-1) @ F.normalize(x.T, dim=0)
            Ak_weighted = alpha_k * Ak * sim

            # Top-K + masking 
            topk_values, topk_indices = torch.topk(Ak_weighted, self.top_k, dim=-1)
            mask = torch.rand_like(Ak_weighted)
            retain_mask = (mask < self.mask_ratio).float()

            # 
            final_mask = torch.zeros_like(Ak_weighted)
            for i in range(Ak_weighted.size(0)):
                final_mask[i, topk_indices[i]] = 1.0
            final_mask += retain_mask
            final_mask = (final_mask > 0).float()  


            Ak_filtered = Ak_weighted * final_mask
            Hk = Ak_filtered @ x
            high_features.append(alpha_k * Hk)

        H_high = torch.stack(high_features, dim=0).sum(dim=0)

        H = self.beta * H_low + (1 - self.beta) * H_high
        H = F.relu(H)
        H = F.dropout(H, p=self.dropout, training=self.training)  # Dropout层

        # output
        out = self.output_proj(H)
        return F.log_softmax(out, dim=-1)

def train_loss(out, y, train_mask, A_filtered, features, topk_indices, device):
    task_loss = F.nll_loss(out[train_mask], y[train_mask])
    loss_lcs = lcs_loss(A_filtered, features, topk_indices, device)
    loss_sparse = sparse_loss(A_filtered)

    lambda1 = 0.5
    lambda2 = 1e-4

    total_loss = task_loss + lambda1 * loss_lcs + lambda2 * loss_sparse
    return total_loss

# LCS_loss
def lcs_loss(A_filtered, features, topk_indices, device):
    N = A_filtered.size(0)

    sim = torch.mm(features, features.T)
    sim = sim / (features.norm(dim=1, keepdim=True) * features.norm(dim=1, keepdim=True).T + 1e-8)

    loss = 0.0
    for i in range(N):
        neighbors = topk_indices[i]
        lcs_vals = sim[i, neighbors]
        loss += torch.sum((1 - lcs_vals) ** 2)
    return loss / N

# Sparse_loss
def sparse_loss(A_filtered):
    return torch.sum(torch.abs(A_filtered)) / A_filtered.numel()
