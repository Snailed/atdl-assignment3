import torch
import models
import datamodules
import argparse
import pytorch_lightning as pl
import torch_geometric


def cli_main(args):
    if args.train:
        pl.seed_everything(42, workers=True)
        trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=10, deterministic=True, default_root_dir='checkpoints', enable_checkpointing=True)
        datamodule, num_features = datamodules.get_datamodule()
        model = models.LightningVAE(num_features, 2, torch_geometric.nn.GCNConv, [16], [32], activation=torch.nn.functional.relu)
        trainer.fit(model, datamodule)


    
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='GVAE encoder characterizer'
    )
    parser.add_argument('-t', '--train', type=bool)
    parser.add_argument('--epochs', default=1)
    parser.add_argument('--test')

    cli_main(parser.parse_args())