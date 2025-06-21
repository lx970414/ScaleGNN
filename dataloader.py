from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader

def load_small_graph(cfg):
    """
    Only for small Planetoid datasets such as Cora/CiteSeer/Pubmed, full graph training
    """
    name = cfg['dataset']
    root = cfg.get('data_root', './data')
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    return data, dataset.num_features, dataset.num_classes

def load_ogb_graph(cfg):
    """
    For OGB NodeProp datasets, returns data objects and split indexes
    """
    name = cfg['dataset']
    root = cfg.get('data_root', './data')
    dataset = PygNodePropPredDataset(name=name, root=root)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.y = data.y.squeeze()
    return data, split_idx, data.x.size(1), dataset.num_classes

def get_dataloader(cfg, data, split_idx):
    """
    Constructing NeighborLoader for OGB Big Picture with support for training, validation, testing
    """
    num_layers = cfg['model'].get('num_layers', 3)
    max_k = cfg.get('max_k', 20)  # Can be set directly in the main config
    batch_size = cfg['train'].get('batch_size', 1024)
    num_workers = cfg['train'].get('num_workers', 8)

    neighbor_sizes = [max_k] * num_layers

    loader_train = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
    )
    loader_val = NeighborLoader(
        data,
        input_nodes=split_idx['valid'],
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        persistent_workers=True,
        pin_memory=True,
    )
    loader_test = NeighborLoader(
        data,
        input_nodes=split_idx['test'],
        num_neighbors=neighbor_sizes,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers // 2,
        persistent_workers=True,
        pin_memory=True,
    )
    return loader_train, loader_val, loader_test
