import torch
import torch.nn.functional as F

def compute_loss(logits, labels, lcs_scores, Ak_filters, mk, alpha, cfg):
    """
    计算总损失（支持主任务、LCS监督、稀疏正则，兼容消融实验）
    """
    loss_cfg = cfg.get('loss', {})
    lambda_lcs = loss_cfg.get('lambda_lcs', 0.0)
    lambda_sc = loss_cfg.get('lambda_sc', 0.0)
    loss_cls = F.cross_entropy(logits, labels)
    loss_lcs = 0
    loss_sc = 0

    # LCS和SC仅在非消融时有效
    if lcs_scores is not None and Ak_filters is not None and lambda_lcs > 0:
        for lcs, ak in zip(lcs_scores, Ak_filters):
            if lcs is None or ak is None:
                continue
            mask = ak > 0
            score = lcs[mask]
            loss_lcs += ((1 - score) ** 2).mean() if score.numel() > 0 else 0
            loss_sc += ak.sum() if lambda_sc > 0 else 0

    loss = loss_cls + lambda_lcs * loss_lcs + lambda_sc * loss_sc
    return loss

def train_epoch(model, data, loader, cfg, optimizer, device, logger=None):
    """
    单轮训练。兼容全图训练和子图采样Loader。
    返回 (loss, mk均值, alpha均值)
    """
    model.train()
    mk_history = []
    alpha_history = []

    if loader is not None:
        total_loss = 0
        n_total = 0
        for batch in loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            x = batch.x
            batch_params = {'adjs': batch.adjs_t, 'n_id': batch.n_id}
            out, lcs_scores, Ak_filters, mk, alpha = model(x, batch.edge_index, batch_params)
            loss = compute_loss(
                out, batch.y[:batch.batch_size],
                lcs_scores, Ak_filters, mk, alpha, cfg
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.batch_size
            n_total += batch.batch_size
            mk_history.append(mk)
            alpha_history.append(alpha)
        mk_mean = [float(torch.tensor([x[k] for x in mk_history]).float().mean()) for k in range(len(mk_history[0]))] if mk_history else []
        alpha_mean = [float(torch.tensor([x[k] for x in alpha_history]).float().mean()) for k in range(len(alpha_history[0]))] if alpha_history else []
        if logger is not None:
            logger.log(None, alpha_mean, mk_mean)
        return total_loss / max(n_total, 1), mk_mean, alpha_mean
    else:
        optimizer.zero_grad()
        logits, lcs_scores, Ak_filters, mk, alpha = model(data.x, data.edge_index)
        loss = compute_loss(
            logits[data.train_mask], data.y[data.train_mask],
            lcs_scores, Ak_filters, mk, alpha, cfg
        )
        loss.backward()
        optimizer.step()
        if logger is not None:
            logger.log(None, alpha, mk)
        return loss.item(), mk, alpha

@torch.no_grad()
def eval_model(model, data, loader, split, device):
    """
    验证/测试接口。兼容Loader和mask分支。
    返回准确率 (acc)。
    """
    model.eval()
    if loader is not None:
        ys, preds = [], []
        for batch in loader:
            batch = batch.to(device)
            x = batch.x
            batch_params = {'adjs': batch.adjs_t, 'n_id': batch.n_id}
            out, _, _, _, _ = model(x, batch.edge_index, batch_params)
            pred = out.argmax(dim=1)
            ys.append(batch.y[:batch.batch_size].cpu())
            preds.append(pred.cpu())
        y_cat = torch.cat(ys, dim=0)
        pred_cat = torch.cat(preds, dim=0)
        acc = (pred_cat == y_cat).sum().item() / y_cat.size(0)
        return acc
    else:
        logits, _, _, _, _ = model(data.x, data.edge_index)
        pred = logits.argmax(dim=1)
        if split == 'val':
            mask = data.val_mask
        elif split == 'test':
            mask = data.test_mask
        else:
            mask = data.train_mask
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        return acc
