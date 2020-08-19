import os, time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from data_loader import DataSetWrapper
from model import resnet18, ProjectionHead, NT_XentLoss
from utils import make_permutation, save_checkpoint

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

    f, g = resnet18().to(device), ProjectionHead(512, 2048).to(device)
    criterion = NT_XentLoss(T)

    ### You may use below optimizer & scheduler ###
    optimizer = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    '''
    Model-- ResNet18(encoder network) + MLP with one hidden layer(projection head)
    Loss -- NT-Xent Loss
    '''
    
    best_val_loss = 1e9
    epochs_since_improvement = 0
    patience = 5
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        #TODO: Tranining
        t = time.time()
        for i, ((x_odd, x_even), _) in enumerate(train_loader):
            loss = SimCLR(x_odd, x_even, f, g, criterion, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (i + 1) % 100 == 0:
                print("[%d/%d][%d/%d] loss : %.4f | time : %.2fs"
                    %(epoch + 1, epochs, i + 1, len(train_loader), loss.item(), time.time() - t))
                t = time.time()
        
        save_checkpoint(f, g, 'checkpoints/epoch%d.pt'%(epoch + 1))

        #TODO: Validation
        # You have to save the model using early stopping
        val_loss = 0.
        with torch.no_grad():
            t = time.time()
            for (val_xi, val_xj), _ in valid_loader:
                loss = SimCLR(val_xi, val_xj, f, g, criterion, device)
                val_loss += loss

                val_losses.append(loss.item())
            
            val_loss /= len(val_loss)

            print("[%d/%d] validation loss : %.4f | time : %.2fs"
                  % (epoch + 1, epochs, loss.item(), time.time() - t))

            if val_loss < best_val_loss :
                epochs_since_improvement = 0
                best_val_loss = val_loss
                save_checkpoint(f, g, 'checkpoints/best.pt')
            else:
                epochs_since_improvement += 1
                print("There's no improvement for %d epochs"%(epochs_since_improvement))
                if epochs_since_improvement >= patience :
                    print("Early Stopping!")
                    break
                
        scheduler.step()

    # Loss Curve Plotting ###################################
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train loss')
    ax.plot(val_losses, label='validation loss')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.set(title="Loss Curve")
    ax.grid()

    fig.savefig("loss_curve.png")
    plt.close()
    #########################################################

def SimCLR(x_odd, x_even, f, g, criterion, device):
    N = x_odd.size(0)  # batch size

    # x_odd = [x1, x3, ..., x2N-1] and x_even = [x2, x4, ..., x2N] where N is the batch size
    x_odd, x_even = x_odd.to(device), x_even.to(device)

    # z_odd = [z_1^T, z_3^T, ..., z_2N-1^T], z_even = [z_2^T, z_4^T, ..., z_2N^T] ; of size [N, 512], [N,512]
    z_odd, z_even = g(f(x_odd)), g(f(x_even))

    # [z_1^T, z_3^T, ..., z_2N-1^T, z_2^T, z_4^T, ..., z_2N^T] ; of size [2N, 512]
    z = torch.cat([z_odd, z_even], dim=0)

    perm = make_permutation(N)  # [0, N, 1, N+1, ..., N-1, 2N-1]
    z = z[perm, :]  # [z_1^T, z_2^T, ..., z_2N-1^T, z_2N^T]
    
    # [||z_1||, ||z_2||, ..., ||z_2N||] ; of size [2N, 1]
    z_norm = torch.norm(z, dim=-1).unsqueeze(-1)

    zT = z.transpose(0, 1)  # [z_1, z_2, ..., z_2N] ; of size [512, 2N]
    z_normT = z_norm.transpose(0, 1)

    # zTz = [[z_1^T * z_1, z_1^T * z_2, ..., z_1^T * z_2N],
    #        [z_2^T * z_1, z_2^T * z_2, ..., z_2^T * z_2N],
    #                               ...
    #        [z_2N^T * z_1, z_2N^T * z_2, ..., z_2N^T * z_2N]]
    #
    # zTz_norm = [[||z_1|| * ||z_1||, ||z_1|| * ||z_2||, ..., ||z_1|| * ||z_2N||],
    #             [||z_2|| * ||z_1||, ||z_2|| * ||z_2||, ..., ||z_2|| * ||z_2N||],
    #                                           ...
    #             [||z_2N|| * ||z_1||, ||z_2N|| * ||z_2||, ..., ||z_2N|| * ||z_2N||]

    zTz = torch.mm(z, zT)  # [2N, 2N]
    zTz_norm = torch.mm(z_norm, z_normT)  # [2N, 2N]

    s = zTz / zTz_norm  # cosine similarities : s[i,j] = s_(i,j)

    loss = 0.
    for k in range(N):
        loss += criterion(2*k, 2*k + 1, s) + criterion(2*k + 1, 2*k, s)

    return loss/(2*N)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimCLR implementation")

    parser.add_argument(
        '--epochs',
        type=int,
        default=40)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)

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
        default=4)

    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)

    args = parser.parse_args()
    main(args)
