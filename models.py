import torch
import torch_geometric
import pytorch_lightning as pl

class EncoderBlock(torch.nn.Module):
    def __init__(self, input_features, output_features, hidden_layers, convolution=torch_geometric.nn.GCNConv, activation=torch.nn.functional.relu):
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.gcns = torch.nn.ModuleList()
        for dim1, dim2 in zip([input_features] + hidden_layers, hidden_layers + [output_features]):
            self.gcns.append(convolution(dim1, dim2))
        self.activation = activation
    
    def forward(self, x, edge_index):
        for gcn in self.gcns:
            x = self.activation(gcn(x, edge_index))
        return x

class Encoder(torch.nn.Module):
    def __init__(self, pre_encoder, mu_encoder, logstd_encoder):
        super().__init__()
        self.pre_encoder = pre_encoder
        self.mu_encoder = mu_encoder
        self.logstd_encoder = logstd_encoder
    
    def forward(self, x):
        x = self.pre_encoder(x)
        return self.mu_encoder(x), self.logstd_encoder(x)

class VariationalWrapper(torch.nn.Module):
    def __init__(self, input_features, latent_dim, convolution, pre_hidden_layers, post_hidden_layers, activation):
        super().__init__()
        self.convolution = convolution
        self.activation = activation

        pre_encoder = EncoderBlock(input_features, pre_hidden_layers[-1], pre_hidden_layers[:-1])
        mu_encoder = EncoderBlock(input_features, latent_dim, convolution=convolution, hidden_layers=post_hidden_layers, activation=activation)
        logstd_encoder = EncoderBlock(input_features, latent_dim, convolution=convolution, hidden_layers=post_hidden_layers, activation=activation)
        self.vae = torch_geometric.nn.models.VGAE(Encoder(pre_encoder, mu_encoder, logstd_encoder))
    
    def forward(self, x, edge_index):
        z = self.vae.encode(x, edge_index)
        recon_loss = self.vae.recon_loss(z, edge_index)
        kld = self.vae.kld_loss()
        return self.vae.decode(z, edge_index), z, recon_loss, kld


class LightningVAE(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = VariationalWrapper(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        A, z, recon_loss, kld = self.model(batch.x, batch.edge_index)
        loss = recon_loss + 1/(batch.num_nodes) * kld
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer