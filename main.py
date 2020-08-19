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
        for i, ((xi, xj), _) in enumerate(train_loader):
            loss = SimCLR(xi, xj, f, g, criterion, device)

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
            
            val_loss /= len(valid_loader)

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
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.set(title="Train Loss Curve")
    ax.grid()

    fig.savefig("train_loss_curve.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(val_losses, label='validation loss')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.set(title="Validation Loss Curve")
    ax.grid()

    fig.savefig("val_loss_curve.png")
    plt.close()
    #########################################################

def SimCLR(xT_odd, xT_even, f, g, criterion, device):
    N = xT_odd.size(0)  # batch size

    '''
    ########################## Linear Algebraic View ###################################
    x_T mean x.transpose()

    xT_odd = [x1_T,             xT_even = [x2_T,
              x3_T,                        x4_T,
               ...                          ...
              x2N-1_T]                     x2N_T]
    
    zT_odd = [z1_T,             zT_even = [z2_T,
              z3_T,                        z4_T,
               ...                          ...
              z2N-1_T]                     z2N_T]

    zT = [z1_T,        --->     zT = [z1_T,
          z3_T,        perm           z2_T,
           ...                         ...
          z2N-1_T,                    zN_T,
          z2_T,                       zN+1_T,
          z4_T,                       zN+2_T,
           ...                         ...
          z2N_T]                      z2N_T]
    
    z_norm = [||z1_T||,  =  [||z1||,
              ||z2_T||,      ||z2||,
                ...             ...
              ||z2N_T||]     ||z2N||]  (Note that ||z_T|| = ||z||)

    z = [z1, z2, ..., z2N]

    z_normT = [||z1||, ||z2||, ... ||z2N||]

    zTz = zT * z
        = [[z1_T * z1, z1_T * z2, ..., z1_T * z2N],
           [z2_T * z1, z2_T * z2, ..., z2_T * z2N],
                          ...
           [z2N_T * z1, z2N_T * z2, ..., z2N_T * z2N]]

    zzT_norm = z_norm * z_normT
             = [[||z1|| * ||z1||, ||z1|| * ||z2||, ..., ||z1|| * ||z2N||],
                [||z2|| * ||z1||, ||z2|| * ||z2||, ..., ||z2|| * ||z2N||],
                                              ...
                [||z2N|| * ||z1||, ||z2N|| * ||z2||, ..., ||z2N|| * ||z2N||]

    s = zTz / zzT_norm
      = [[s11, s12, ..., s12N],
         [s21, s22, ..., s22N],
                  ...
         [s2N1, s2N2, ..., s2N2N]]      where sij = cosine_similarity(zi, zj)

    ####################################################################################
    '''

    zT_odd, zT_even = g(f(xT_odd.to(device))), g(f(xT_even.to(device))) # [N, 512], [N,512]
    zT = torch.cat([zT_odd, zT_even], dim=0) # [2N, 512]

    perm = make_permutation(N) # [0, N, 1, N+1, ..., N-1, 2N-1]
    zT = zT[perm, :]
    z_norm = torch.norm(zT, dim=-1).unsqueeze(-1) # [2N, 1]

    z = zT.transpose(0, 1) # [512, 2N]
    z_normT = z_norm.transpose(0, 1) # [1, 2N]
    
    zTz = torch.mm(zT, z)  # [2N, 2N]
    zzT_norm = torch.mm(z_norm, z_normT)  # [2N, 2N]

    s = zTz / zzT_norm # [2N, 2N]

    loss = 0.
    for k in range(N):
        loss += criterion(2 * k, 2 * k + 1, s) + criterion(2 * k + 1, 2 * k, s)

    return loss/(2 * N)




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
        default=1)

    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)

    args = parser.parse_args()
    main(args)
