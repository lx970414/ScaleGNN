# ScaleGNN

This is the code for paper:
> ScaleGNN: Towards Scalable Graph Neural Networks via Adaptive High-order Neighboring Feature Fusion

## Dependencies
Recent versions of the following packages for Python 3(>=3.8) are required:
* torch==2.1.0
* torch-geometric==2.4.0
* torch-sparse==0.6.17
* torch-scatter==2.1.2
* torch-cluster==1.6.3
* torch-spline-conv==1.2.2
* ogb==1.3.6
* pyyaml==6.0.1
* wandb==0.16.6
* pandas==2.2.2
* matplotlib==3.8.4
* scipy==1.13.0
* numpy==1.26.4
* tqdm==4.66.4

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Citeseer/Cora/Pubmed: https://github.com/tkipf/gcn/tree/master/gcn/data
* ogbn-arxiv/ogbn-products/ogbn-papers100M: https://ogb.stanford.edu/docs/leader_nodeprop

## Usage

* `python main.py`
