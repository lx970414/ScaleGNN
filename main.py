import yaml
import torch
import os
import shutil
import argparse
from dataloader import load_small_graph, load_ogb_graph, get_dataloader
from model import ScaleGNN
from train import (
    train_epoch_small, eval_small,
    train_epoch_large, eval_large
)
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    return parser.parse_args()

def main():
    # 1. Read Configuration
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg.get('device', 'cuda:0'))

    # 2. Load data
    if cfg['dataset'].startswith('ogbn'):
        data, split_idx, in_dim, num_classes = load_ogb_graph(cfg)
        loader_train, loader_val, loader_test = get_dataloader(cfg, data, split_idx)
        data = data.to(device)
    else:
        data, in_dim, num_classes = load_small_graph(cfg)
        loader_train = loader_val = loader_test = None
        data = data.to(device)

    # Automatic correction of in_dim/out_dim
    cfg['model']['in_dim'] = in_dim
    cfg['model']['out_dim'] = num_classes

    # 3. Support for ablation parameters (optional, recommended to write to config.yaml or specify manually)
    ablation = cfg['model'].get('ablation', {})

    # 4. Initialisation model (automatic unwrapping of parameters, support for extensions such as ablation)
    model = ScaleGNN(**cfg['model'], ablation=ablation).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['train'].get('lr', 1e-2),
        weight_decay=cfg['train'].get('weight_decay', 5e-4)
    )

    # 4. Training and validation
    best_val, best_test, best_epoch = 0, 0, 0
    save_dir = cfg['train'].get('save_path', './save')
    os.makedirs(save_dir, exist_ok=True)

    # Automatically archive the current experiment configuration
    shutil.copy('config.yaml', os.path.join(save_dir, 'config.yaml'))

    for epoch in range(1, cfg['train']['epochs'] + 1):
        try:
            if cfg['dataset'].startswith('ogbn'):
                loss = train_epoch_large(model, loader_train, data, cfg, optimizer, device)
                val_acc = eval_large(model, loader_val, data, device)
                test_acc = eval_large(model, loader_test, data, device)
            else:
                loss = train_epoch_small(model, data, cfg, optimizer)
                val_acc = eval_small(model, data, split='val')
                test_acc = eval_small(model, data, split='test')
        except Exception as e:
            print(f"[Error][Epoch {epoch}] {str(e)}")
            break

        if val_acc > best_val:
            best_val, best_test, best_epoch = val_acc, test_acc, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))

        if epoch % cfg['train'].get('log_interval', 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

    # 5. Optimal Model Replication Verification
    model.load_state_dict(torch.load(os.path.join(save_dir, "best.pt"), map_location=device))
    if cfg['dataset'].startswith('ogbn'):
        test_acc = eval_large(model, loader_test, data, device)
    else:
        test_acc = eval_small(model, data, split='test')
    print(f"Best Val Acc: {best_val:.4f} at epoch {best_epoch} | Final Test Acc (best.pt): {test_acc:.4f}")

if __name__ == "__main__":
    main()



# import yaml
# import torch
# import os
# from dataloader import load_small_graph, load_ogb_graph, get_dataloader
# from model import ScaleGNN
# from train import (
#     train_epoch_small, eval_small,
#     train_epoch_large, eval_large
# )
# from utils import set_seed

# def main():
#     # 1. 读取配置
#     with open("config.yaml", "r") as f:
#         cfg = yaml.safe_load(f)
#     set_seed(cfg['seed'])
#     device = torch.device(cfg['device'])

#     # 2. 加载数据
#     if cfg['dataset'].startswith('ogbn'):
#         data, split_idx, in_dim, num_classes = load_ogb_graph(cfg)
#         loader_train, loader_val, loader_test = get_dataloader(cfg, data, split_idx)
#         data = data.to(device)
#     else:
#         data, in_dim, num_classes = load_small_graph(cfg)
#         loader_train, loader_val, loader_test = None, None, None
#         data = data.to(device)

#     # 自动修正in_dim/out_dim
#     cfg['model']['in_dim'] = in_dim
#     cfg['model']['out_dim'] = num_classes

#     # 3. 初始化模型
#     model = ScaleGNN(
#         in_dim=cfg['model']['in_dim'],
#         hidden_dim=cfg['model']['hidden_dim'],
#         out_dim=cfg['model']['out_dim'],
#         num_layers=cfg['model']['num_layers'],
#         topk=cfg['model']['topk'],
#         alpha=cfg['model']['alpha'],
#         beta=cfg['model']['beta'],
#         dropout=cfg['model']['dropout']
#     ).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

#     # 4. 训练与验证
#     best_val, best_test, best_epoch = 0, 0, 0
#     for epoch in range(1, cfg['train']['epochs'] + 1):
#         if cfg['dataset'].startswith('ogbn'):
#             loss = train_epoch_large(model, loader_train, data, cfg, optimizer, device)
#             val_acc = eval_large(model, loader_val, data, device)
#             test_acc = eval_large(model, loader_test, data, device)
#         else:
#             loss = train_epoch_small(model, data, cfg, optimizer)
#             val_acc = eval_small(model, data, split='val')
#             test_acc = eval_small(model, data, split='test')

#         if val_acc > best_val:
#             best_val, best_test, best_epoch = val_acc, test_acc, epoch
#             os.makedirs(cfg['train']['save_path'], exist_ok=True)
#             torch.save(model.state_dict(), os.path.join(cfg['train']['save_path'], "best.pt"))

#         if epoch % cfg['train']['log_interval'] == 0 or epoch == 1:
#             print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

#     print(f"Best Val Acc: {best_val:.4f} at epoch {best_epoch} | Final Test Acc: {best_test:.4f}")

# if __name__ == "__main__":
#     main()
