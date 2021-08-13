# Graph-Free Knowledge Distillation for Graph Neural Networks

This repository is a PyTorch implementation of [Graph-Free Knowledge Distillation for Graph Neural Networks].

## Requirements

```
The code was tested on
Python 3.6
torch 1.6.0
torchvision 0.7.0
DGL 0.4.2
```
```
torch 1.2.0 with torchvision 0.4.0 and torch 1.4.0 with torchvision 0.5.0 should also work.
```


## Running the code
To distill knowledge from GCN5-64 to student GCN-3-32 or GIN-3-32 with the generated fake data (from teacher GCN5-64), run the command: \
`sh train_kd_gcn.sh`


To re-generate fake graphs from teacher GCN-5-64, run the command: \
`sh generate_fake_graphs.sh`


Note

When you use a different teacher GNN or dataset, you need to re-tune the hyper-parameters (i.e., bn_reg_scale or/and onehot_cof).

## Citation

```bibtex
@inproceedings{deng2021graph,
  title={Graph-Free Knowledge Distillation for Graph Neural Networks},
  author={Deng, Xiang and Zhang, Zhongfei},
  booktitle = {The 30th International Joint Conference on Artificial Intelligence},
  year={2021}
}
```
