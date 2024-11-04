from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch

def get_cora(nodes_per_class):
    dataset = Planetoid(
        root='../dataset',
        name='Cora',
        split='random',
        num_train_per_class=nodes_per_class
    )
    return dataset, dataset[0]