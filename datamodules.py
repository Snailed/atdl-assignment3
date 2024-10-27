from torch_geometric.datasets import Planetoid
from torch_geometric.data.lightning.datamodule import LightningDataset
import torch_geometric.transforms as T

def get_datamodule():
    transform = T.Compose([
        T.NormalizeFeatures(),
        T.RandomLinkSplit(num_val=0.05, num_test=0.2, is_undirected=True,
                        split_labels=True, add_negative_train_samples=False),
    ])
    dataset = dataset = Planetoid(
        root='./dataset',
        name='Cora',
        split='random',
        transform=transform
    )
    train_dataset, val_dataset, test_dataset = dataset[0]
    return LightningDataset(train_dataset, val_dataset, test_dataset), dataset.num_features