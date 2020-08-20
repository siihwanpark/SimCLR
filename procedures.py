import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils import make_permutation, save_checkpoint, load_checkpoint, save_checkpoint_classifier, load_checkpoint_classifier

def SimCLR(xT_odd, xT_even, encoder, projection, criterion):
    N = xT_odd.size(0)  # batch size

    '''
    ########################## Linear Algebraic View ###################################
    x_T means x.transpose()

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
                [||z2N|| * ||z1||, ||z2N|| * ||z2||, ..., ||z2N|| * ||z2N||]]

    s = zTz / zzT_norm
      = [[s11, s12, ..., s12N],
         [s21, s22, ..., s22N],
                  ...
         [s2N1, s2N2, ..., s2N2N]]      where sij = cosine_similarity(zi, zj)
                                                  = zi_T * zj / ||zi|| * ||zj||

    ####################################################################################
    '''

    zT_odd, zT_even = projection(encoder(xT_odd)), projection(encoder(xT_even))  # [N, 512], [N,512]
    zT = torch.cat([zT_odd, zT_even], dim=0)  # [2N, 512]

    perm = make_permutation(N)  # [0, N, 1, N+1, ..., N-1, 2N-1]
    zT = zT[perm, :]
    z_norm = torch.norm(zT, dim=-1).unsqueeze(-1)  # [2N, 1]

    z = zT.transpose(0, 1)  # [512, 2N]
    z_normT = z_norm.transpose(0, 1)  # [1, 2N]

    zTz = torch.mm(zT, z)  # [2N, 2N]
    zzT_norm = torch.mm(z_norm, z_normT)  # [2N, 2N]

    s = zTz / zzT_norm  # [2N, 2N]

    loss = 0.
    for k in range(N):
        loss += criterion(2 * k, 2 * k + 1, s) + criterion(2 * k + 1, 2 * k, s)

    return loss/(2 * N)


def train(epochs, patience, optimizer, scheduler, train_loader, valid_loader, encoder, projection, criterion):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    best_val_loss = 1e9
    epochs_since_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        ################# TODO: Tranining #################
        t = time.time()
        for i, ((xi, xj), _) in enumerate(train_loader):
            xi, xj = xi.to(device), xj.to(device)
            loss = SimCLR(xi, xj, encoder, projection, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            if (i + 1) % 10 == 0:
                print("[%d/%d][%d/%d] loss : %.4f | time : %.2fs"
                      % (epoch + 1, epochs, i + 1, len(train_loader), loss.item(), time.time() - t))
                t = time.time()

        save_checkpoint(encoder, projection, 'checkpoints/epoch%d.pt' % (epoch + 1))

        ################# TODO: Validation #################
        val_loss = 0.
        with torch.no_grad():
            t = time.time()
            for (val_xi, val_xj), _ in valid_loader:
                val_xi, val_xj = val_xi.to(device), val_xj.to(device)
                loss = SimCLR(val_xi, val_xj, encoder, projection, criterion)

                val_loss += loss
                val_losses.append(loss.item())
            val_loss /= len(valid_loader)

            print("[%d/%d] validation loss : %.4f | time : %.2fs"%(epoch + 1, epochs, loss.item(), time.time() - t))

            if val_loss < best_val_loss:
                epochs_since_improvement = 0
                best_val_loss = val_loss
                save_checkpoint(encoder, projection, 'checkpoints/best.pt')

            else:
                epochs_since_improvement += 1
                print("There's no improvement for %d epochs."%(epochs_since_improvement))
                
                if epochs_since_improvement >= patience:
                    print("The training halted by early stopping criterion.")
                    break

        scheduler.step()
    
    return train_losses, val_losses

def train_classifier(epochs, train_loader, encoder, classifier, criterion, optimizer):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    for epoch in range(epochs):
        # Classifier Training
        t = time.time()
        for x, label in train_loader:
            x, label = x.to(device), label.to(device)

            h = encoder(x)  # [batch, 512]
            preds = classifier(h)  # [batch, num_classes]

            loss = criterion(preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("[%d/%d] classifier loss : %.4f | time : %.2fs" %
                (epoch + 1, epochs, loss.item(), time.time() - t))

def test(test_loader, encoder, classifier):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    correct, total = 0, 0
    for x, label in test_loader:
        x, label = x.to(device), label.to(device)
        h = encoder(x)  # [batch, 512]

        output = classifier(h)  # [batch, num_classes]
        preds = torch.argmax(output, dim=1)

        correct += int(torch.sum(preds == label))
        total += int(label.size(0))
    
    return correct / total