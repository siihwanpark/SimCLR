import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from data_loader import DataSetWrapper


def main(args):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    ### Hyperparameters setting ###
    epochs = args.epochs
    batch_size = args.batch_size
    T = args.temperature
    proj_dim = args.out_dim

    ### DataLoader ###
    dataset = DataSetWrapper(
        args.batch_size, args.num_worker, args.valid_size, input_shape=(96, 96, 3))
    train_loader, valid_loader = dataset.get_data_loaders()

    ### You may use below optimizer & scheduler ###
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    '''
    Model-- ResNet18(encoder network) + MLP with one hidden layer(projection head)
    Loss -- NT-Xent Loss
    '''

    for epoch in range(epochs):

        #TODO: Traninig
        for (xi, xj), _ in train_loader:
            pass

        #TODO: Validation
        # You have to save the model using early stopping
        with torch.no_grad():
            for (val_xi, val_xj), _ in valid_loader:
                pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimCLR implementation")

    parser.add_argument(
        '--epochs',
        type=int,
        default=40)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5)
    parser.add_argument(
        '--out_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--num_worker',
        type=int,
        default=8)

    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)

    args = parser.parse_args()
    main(args)
