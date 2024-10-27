import torch
import torch_geometric as pyg
from torch.nn import Linear
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, num_outputs):
        super().__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, num_outputs)
        #self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.
        
        # Apply a final (linear) classifier.
        #out = self.classifier(h)

        return h

class InnerProductDecoder(torch.nn.Module): # Samples Adjacency matrix
    def forward(self, inputs, *args):
        x = inputs.T
        x = inputs @ x
        x = x.reshape(-1)
        ps = torch.nn.functional.sigmoid(x)
        return ps

class GVAE(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, embedding_dim=2):
        super().__init__()
        torch.manual_seed(1234)
        self.encoder = encoder
        self.decoder = decoder
        self.embedding_dim = embedding_dim
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        mu, log_std = z[:,:self.embedding_dim], z[:,self.embedding_dim:]
        z_sampled = torch.randn_like(log_std) * torch.exp(log_std) + mu # Reparameterization trick
        x_recon = self.decoder(z_sampled, edge_index)
        return x_recon, mu, log_std, z_sampled
