"""
ScaleGNN/ScaleGNN_b 
===============================

Support all ablation studies （main、LCS、fusion、scalegnn_b）+  PyG mini-batch（dense）和 OGB large-scale（sparse batch processing）。

use ablation to switch different functions

参数说明:
- in_dim: Input feature dimension
- hidden_dim: hidden layer dimension
- out_dim: number of output categories/regression dimension
- num_layers: multi-order maximum order K
- max_k: LCS screening for maximum number of neighbours per order (only useful if full ScaleGNN version)
- dropout: dropout ratio
- ablation: functional dictionary for controlling various types of ablation/control experiments

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleGNN(nn.Module):
    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers, max_k=20, dropout=0.5, ablation=None
    ):
        """
        ablation dictionary optional：
        - "only_high_order": True -> ScaleGNN_b (multi-order neighbourhood decomposition only)
        - "no_lcs": True -> Turn off LCS screening
        - "no_fusion": True -> Close feature fusion
        - "fix_alpha": True -> Fixed averaging of multi-order weights
        - "fix_mk": True -> Fixed (max_k)/2 per order topk
        - "fix_beta": True -> Fusion weight beta fixed 0.5
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.max_k = max_k
        self.dropout = dropout

        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.lin_hid = nn.Linear(hidden_dim, hidden_dim)
        self.low_linear = nn.Linear(hidden_dim, hidden_dim)
        self.high_linear = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.ablation = ablation or {}

        if self.ablation.get("fix_alpha", False):
            self.alpha_raw = nn.Parameter(torch.ones(num_layers), requires_grad=False)
        else:
            self.alpha_raw = nn.Parameter(torch.randn(num_layers))

        if self.ablation.get("fix_mk", False):
            self.topk_logit = nn.Parameter(torch.zeros(num_layers), requires_grad=False)
        else:
            self.topk_logit = nn.Parameter(torch.ones(num_layers) * 1.5)

        self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if self.ablation.get("fix_beta", False):
            self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        else:
            self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index=None, batch_params=None):
        """
        x: node characteristics [N, in_dim]
        edge_index: adjacency sparse format [2, E]
        batch_params: Support for sparse batch/OGB scenarios (e.g. PyG NeighborLoader batch input)
        return: (logits, lcs_scores_list, Ak_filter_list, mk_int, alpha)
        """

        # MLP
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_hid(x))

        # ========= ScaleGNN_b =========
        if self.ablation.get("only_high_order", False):
            if batch_params is not None:
                adjs = batch_params["adjs"]
                batch_size = adjs[0].size[1]
                K = self.num_layers
                agg = 0
                out_x = x
                for k in range(K):
                    edge_index_k, _, size_k = adjs[k]
                    Ak_dense = self.sparse_adj_to_dense(adjs[k], size_k).to(x.device)
                    agg = agg + torch.matmul(Ak_dense, out_x)
                agg = agg / K
                out = F.dropout(agg, p=self.dropout, training=self.training)
                logits = self.classifier(out)
                return logits, None, None, None, None
            else:
                N = x.size(0)
                As = self.get_multi_hop_adj(edge_index, N, self.num_layers)
                A_diff = []
                for k in range(self.num_layers):
                    if k == 0:
                        A_diff.append(As[0])
                    else:
                        A_diff.append(As[k] - As[k - 1])
                agg = 0
                for Ak in A_diff:
                    agg = agg + torch.matmul(Ak, x)
                agg = agg / self.num_layers
                out = F.dropout(agg, p=self.dropout, training=self.training)
                logits = self.classifier(out)
                return logits, None, None, None, None

        # ====== Full version/ablation (with batch branch) ======
        if self.ablation.get("fix_alpha", False):
            alpha = torch.ones(self.num_layers, device=x.device) / self.num_layers
        else:
            alpha = torch.softmax(self.alpha_raw, dim=0)
        if self.ablation.get("fix_mk", False):
            mk_float = torch.ones(self.num_layers, device=x.device) * self.max_k // 2
        else:
            mk_float = self.max_k * torch.sigmoid(self.topk_logit)
        mk_int = torch.clamp(mk_float.round().long(), min=1, max=self.max_k)

        # Sparse batch branching (OGB/NeighborLoader etc.)
        if batch_params is not None:
            adjs = batch_params['adjs']
            batch_size = adjs[0].size[1]
            K = self.num_layers
            lcs_scores_list = []
            Ak_filter_list = []
            high_order_features = []
            out_x = x
            for k in range(K):
                edge_index_k, _, size_k = adjs[k]
                mk = mk_int[k].item()
                # ablation: Close LCS/TopK
                if self.ablation.get("no_lcs", False) or k == 0:
                    Ak_filter = self.sparse_adj_to_dense(adjs[k], size_k).to(x.device)
                    lcs_score = None
                else:
                    Ak_filter, lcs_score = self.lcs_topk_filter_sparse(out_x, edge_index_k, size_k, mk)
                Ak_filter_list.append(Ak_filter)
                lcs_scores_list.append(lcs_score)
                high_order_features.append(torch.matmul(Ak_filter, out_x))
            alpha_ = alpha.view(1, K, 1)
            high_feats = torch.stack(high_order_features, dim=1)
            high_out = (alpha_ * high_feats).sum(dim=1)
            high_out = F.relu(self.high_linear(high_out))
            low_A = self.sparse_adj_to_dense(adjs[1], size_k).to(x.device)
            low_out = torch.matmul(low_A, out_x)
            low_out = F.relu(self.low_linear(low_out))
            if self.ablation.get("no_fusion", False):
                out = high_out
            else:
                out = self.beta * low_out + (1 - self.beta) * high_out
            out = F.dropout(out, p=self.dropout, training=self.training)
            logits = self.classifier(out)
            return logits, lcs_scores_list, Ak_filter_list, mk_int.detach().cpu().tolist(), alpha.detach().cpu().tolist()
        else:
            # Dense branching (common in Cora/Pubmed)
            N = x.size(0)
            As = self.get_multi_hop_adj(edge_index, N, self.num_layers)
            A_diff = []
            for k in range(self.num_layers):
                if k == 0:
                    A_diff.append(As[0])
                else:
                    A_diff.append(As[k] - As[k-1])
            high_order_features, lcs_scores_list, Ak_filter_list = [], [], []
            for k, Ak in enumerate(A_diff):
                if self.ablation.get("no_lcs", False) or k == 0:
                    Ak_filter = Ak
                    lcs_score = None
                else:
                    Ak_filter, lcs_score = self.lcs_topk_filter_dense(x, Ak, mk_int[k])
                Ak_filter_list.append(Ak_filter)
                lcs_scores_list.append(lcs_score)
                high_order_features.append(torch.matmul(Ak_filter, x))
            alpha_ = alpha.view(1, self.num_layers, 1)
            high_feats = torch.stack(high_order_features, dim=1)
            high_out = (alpha_ * high_feats).sum(dim=1)
            high_out = F.relu(self.high_linear(high_out))
            low_out = torch.matmul(As[1], x)
            low_out = F.relu(self.low_linear(low_out))
            if self.ablation.get("no_fusion", False):
                out = high_out
            else:
                out = self.beta * low_out + (1 - self.beta) * high_out
            out = F.dropout(out, p=self.dropout, training=self.training)
            logits = self.classifier(out)
            return logits, lcs_scores_list, Ak_filter_list, mk_int.detach().cpu().tolist(), alpha.detach().cpu().tolist()

    @staticmethod
    def get_multi_hop_adj(edge_index, N, num_layers):
        from torch_sparse import SparseTensor
        A = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(N, N))
        As = [A]
        for _ in range(1, num_layers):
            As.append(As[-1] @ A)
        return [a.to_dense() for a in As]

    def lcs_topk_filter_sparse(self, x, edge_index, size, mk):
        batch_size = size[1]
        device = x.device
        src, dst = edge_index
        x_proj1 = self.W1(x)
        x_proj2 = self.W2(x)
        Ak_filter = torch.zeros(batch_size, x.size(0), device=device)
        lcs_softmax = torch.zeros_like(Ak_filter)
        for i in range(batch_size):
            idx = (dst == i).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            src_idx = src[idx]
            scores = (x_proj1[i] * x_proj2[src_idx]).sum(-1) / (x_proj1.size(1) ** 0.5)
            k = min(mk, src_idx.numel())
            topk_idx = scores.topk(k, sorted=False).indices
            selected = src_idx[topk_idx]
            Ak_filter[i, selected] = 1
            lcs_softmax[i, src_idx] = F.softmax(scores, dim=0)
        return Ak_filter, lcs_softmax

    def lcs_topk_filter_dense(self, x, Ak, mk):
        N = x.size(0)
        device = x.device
        neighbor_mask = Ak > 0
        x_proj1 = self.W1(x)
        x_proj2 = self.W2(x)
        lcs_all = torch.matmul(x_proj1, x_proj2.t()) / (x_proj1.size(1) ** 0.5)
        lcs_all[~neighbor_mask] = float('-inf')
        Ak_filter = torch.zeros_like(Ak)
        lcs_softmax = torch.zeros_like(Ak)
        for i in range(N):
            valid_idx = neighbor_mask[i].nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                continue
            scores = lcs_all[i, valid_idx]
            k = min(mk.item(), valid_idx.numel())
            topk_idx = scores.topk(k, sorted=False).indices
            selected = valid_idx[topk_idx]
            Ak_filter[i, selected] = 1
            lcs_softmax[i, valid_idx] = F.softmax(scores, dim=0)
        return Ak_filter, lcs_softmax

    def sparse_adj_to_dense(self, adj, size):
        from torch_sparse import SparseTensor
        edge_index, _, _ = adj
        src_nodes = size[0]
        batch_size = size[1]
        A = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(batch_size, src_nodes))
        return A.to_dense()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class ScaleGNN(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, num_layers, max_k=20, dropout=0.5, ablation=None):
#         super().__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.out_dim = out_dim
#         self.num_layers = num_layers
#         self.max_k = max_k
#         self.dropout = dropout

#         self.lin_in = nn.Linear(in_dim, hidden_dim)
#         self.lin_hid = nn.Linear(hidden_dim, hidden_dim)
#         self.low_linear = nn.Linear(hidden_dim, hidden_dim)
#         self.high_linear = nn.Linear(hidden_dim, hidden_dim)
#         self.classifier = nn.Linear(hidden_dim, out_dim)

#         self.ablation = ablation or {}

#         # Multi-order structural weights alpha (can be learnt, or ablated to fixed uniform)
#         if self.ablation.get('fix_alpha', False):
#             self.alpha_raw = nn.Parameter(torch.ones(num_layers), requires_grad=False)
#         else:
#             self.alpha_raw = nn.Parameter(torch.randn(num_layers))

#         # Multi-order screening of the number of neighbours (learnable, or ablated to fixed max_k/2)
#         if self.ablation.get('fix_mk', False):
#             self.topk_logit = nn.Parameter(torch.zeros(num_layers), requires_grad=False)
#         else:
#             self.topk_logit = nn.Parameter(torch.ones(num_layers) * 1.5)

#         self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
#         self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

#     def forward(self, x, edge_index=None, batch_params=None):
#         x = F.relu(self.lin_in(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.relu(self.lin_hid(x))

#         if self.ablation.get('fix_alpha', False):
#             alpha = torch.ones(self.num_layers, device=x.device) / self.num_layers
#         else:
#             alpha = torch.softmax(self.alpha_raw, dim=0)
#         if self.ablation.get('fix_mk', False):
#             mk_float = torch.ones(self.num_layers, device=x.device) * self.max_k // 2
#         else:
#             mk_float = self.max_k * torch.sigmoid(self.topk_logit)
#         mk_int = torch.clamp(mk_float.round().long(), min=1, max=self.max_k)

#         if batch_params is not None:
#             adjs = batch_params['adjs']
#             batch_size = adjs[0].size[1]
#             K = self.num_layers
#             lcs_scores_list = []
#             Ak_filter_list = []
#             high_order_features = []
#             out_x = x
#             for k in range(K):
#                 edge_index_k, _, size_k = adjs[k]
#                 mk = mk_int[k].item()
#                 Ak_filter, lcs_score = self.lcs_topk_filter_sparse(out_x, edge_index_k, size_k, mk)
#                 Ak_filter_list.append(Ak_filter)
#                 lcs_scores_list.append(lcs_score)
#                 high_order_features.append(torch.matmul(Ak_filter, out_x))
#             alpha_ = alpha.view(1, K, 1)
#             high_feats = torch.stack(high_order_features, dim=1)
#             high_out = (alpha_ * high_feats).sum(dim=1)
#             high_out = F.relu(self.high_linear(high_out))
#             low_A = self.sparse_adj_to_dense(adjs[1], size_k).to(x.device)
#             low_out = torch.matmul(low_A, out_x)
#             low_out = F.relu(self.low_linear(low_out))
#             beta = 0.5
#             out = beta * low_out + (1 - beta) * high_out
#             out = F.dropout(out, p=self.dropout, training=self.training)
#             logits = self.classifier(out)
#             return logits, lcs_scores_list, Ak_filter_list, mk_int.detach().cpu().tolist(), alpha.detach().cpu().tolist()
#         else:
#             N = x.size(0)
#             As = self.get_multi_hop_adj(edge_index, N, self.num_layers)
#             A_diff = []
#             for k in range(self.num_layers):
#                 if k == 0:
#                     A_diff.append(As[0])
#                 else:
#                     A_diff.append(As[k] - As[k-1])
#             high_order_features, lcs_scores_list, Ak_filter_list = [], [], []
#             for k, Ak in enumerate(A_diff):
#                 if k == 0:
#                     Ak_filter = Ak
#                     lcs_score = None
#                 else:
#                     Ak_filter, lcs_score = self.lcs_topk_filter_dense(x, Ak, mk_int[k])
#                 Ak_filter_list.append(Ak_filter)
#                 lcs_scores_list.append(lcs_score)
#                 high_order_features.append(torch.matmul(Ak_filter, x))
#             alpha_ = alpha.view(1, self.num_layers, 1)
#             high_feats = torch.stack(high_order_features, dim=1)
#             high_out = (alpha_ * high_feats).sum(dim=1)
#             high_out = F.relu(self.high_linear(high_out))
#             low_out = torch.matmul(As[1], x)
#             low_out = F.relu(self.low_linear(low_out))
#             beta = 0.5
#             out = beta * low_out + (1 - beta) * high_out
#             out = F.dropout(out, p=self.dropout, training=self.training)
#             logits = self.classifier(out)
#             return logits, lcs_scores_list, Ak_filter_list, mk_int.detach().cpu().tolist(), alpha.detach().cpu().tolist()

#     def get_multi_hop_adj(self, edge_index, N, num_layers):
#         from torch_sparse import SparseTensor
#         A = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(N, N))
#         As = [A]
#         for _ in range(1, num_layers):
#             As.append(As[-1] @ A)
#         return [a.to_dense() for a in As]

#     def lcs_topk_filter_sparse(self, x, edge_index, size, mk):
#         batch_size = size[1]
#         device = x.device
#         src, dst = edge_index
#         x_proj1 = self.W1(x)
#         x_proj2 = self.W2(x)
#         Ak_filter = torch.zeros(batch_size, x.size(0), device=device)
#         lcs_softmax = torch.zeros_like(Ak_filter)
#         for i in range(batch_size):
#             idx = (dst == i).nonzero(as_tuple=False).view(-1)
#             if idx.numel() == 0:
#                 continue
#             src_idx = src[idx]
#             scores = (x_proj1[i] * x_proj2[src_idx]).sum(-1) / (x_proj1.size(1) ** 0.5)
#             k = min(mk, src_idx.numel())
#             topk_idx = scores.topk(k, sorted=False).indices
#             selected = src_idx[topk_idx]
#             Ak_filter[i, selected] = 1
#             lcs_softmax[i, src_idx] = F.softmax(scores, dim=0)
#         return Ak_filter, lcs_softmax

#     def lcs_topk_filter_dense(self, x, Ak, mk):
#         N = x.size(0)
#         device = x.device
#         neighbor_mask = Ak > 0
#         x_proj1 = self.W1(x)
#         x_proj2 = self.W2(x)
#         lcs_all = torch.matmul(x_proj1, x_proj2.t()) / (x_proj1.size(1) ** 0.5)
#         lcs_all[~neighbor_mask] = float('-inf')
#         Ak_filter = torch.zeros_like(Ak)
#         lcs_softmax = torch.zeros_like(Ak)
#         for i in range(N):
#             valid_idx = neighbor_mask[i].nonzero(as_tuple=False).view(-1)
#             if valid_idx.numel() == 0:
#                 continue
#             scores = lcs_all[i, valid_idx]
#             k = min(mk.item(), valid_idx.numel())
#             topk_idx = scores.topk(k, sorted=False).indices
#             selected = valid_idx[topk_idx]
#             Ak_filter[i, selected] = 1
#             lcs_softmax[i, valid_idx] = F.softmax(scores, dim=0)
#         return Ak_filter, lcs_softmax

#     def sparse_adj_to_dense(self, adj, size):
#         from torch_sparse import SparseTensor
#         edge_index, _, _ = adj
#         src_nodes = size[0]
#         batch_size = size[1]
#         A = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(batch_size, src_nodes))
#         return A.to_dense()
