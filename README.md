# ScaleGNN

This is the code for paper:
> ScaleGNN: Towards Scalable Graph Neural Networks via Adaptive High-order Neighboring Feature Fusion

## Dependencies
Recent versions of the following packages for Python 3 are required:
* networkx==2.8.4
* numpy==1.22.3
* matplotlib==3.5.1
* PyYAML==6.0.1
* Requests==2.31.0
* scikit_learn==1.2.2
* scipy==1.10.1
* setuptools==60.2.0
* sphinx_gallery==0.15.0
* tensorboardX==2.6.2
* torch==1.10.1
* torch_cluster==1.6.0
* torch_geometric==2.2.0
* torch_sparse==0.6.13
* tqdm==4.65.0
* dgl==0.4.1

## Datasets
All of the datasets we use are publicly available datasets.
### Link
The used datasets are available at:
* Citeseer/Cora/Pubmed: https://github.com/tkipf/gcn/tree/master/gcn/data
* ogbn-arxiv/ogbn-products/ogbn-papers100M: https://ogb.stanford.edu/docs/leader_nodeprop

## Usage

For Cora/Citeseer/Pubmed:

* `python cit.py  --model ScaleGNN --degree 20 --dataset cora/citeseer/pubmed --weight-decay 1e-5 --lr 0.001 --dropout 0.2 --hidden 128 --epoch 100`

For ogbn-arxiv/products/papers100M:

* `python cit_ogbn.py  --model ScaleGNN --degree 20 --dataset ogbn-arxiv/products/papers100M --weight-decay 0 --lr 0.001 --epoch 300 --seed 1 --gpu 4 --patience 200 --hidden 512 --input-drop 0.2`



