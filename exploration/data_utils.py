from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch

def get_cora(nodes_per_class):
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                        split_labels=True, add_negative_train_samples=False),
    ])
    dataset = Planetoid(
        root='../dataset',
        name='Cora',
        split='random',
        num_train_per_class=nodes_per_class,
        transform=transform
    )
    return dataset, dataset[0]