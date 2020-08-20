import os, time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from data_loader import DataSetWrapper
from model import resnet18, ProjectionHead, NT_XentLoss, Classifier
from utils import make_permutation, save_checkpoint, load_checkpoint, save_checkpoint_classifier, load_checkpoint_classifier, plot_loss_curve
from procedures import train, train_classifier, test

def main(args):

    ### Hyperparameters setting ###
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    main_epochs = args.epochs
    classifier_epochs = args.c_epochs
    T = args.temperature
    patience = args.patience
    num_classes = args.num_classes
    classifier_hidden_dim = args.c_dim
    projection_hidden_dim = args.p_dim
    in_dim = 512 # Constant as long as we use ResNet18

    # model definition
    f, g = resnet18().to(device), ProjectionHead(in_dim, projection_hidden_dim).to(device)

    if not args.test:
        ### Train SimCLR ###
        dataset = DataSetWrapper(args.batch_size, args.num_worker, args.valid_size, input_shape=(96, 96, 3))
        train_loader, valid_loader = dataset.get_data_loaders()

        criterion = NT_XentLoss(T)
        optimizer = torch.optim.Adam(list(f.parameters()) + list(g.parameters()), 3e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        train_losses, val_losses = train(main_epochs, patience, optimizer, scheduler, train_loader, valid_loader, f, g, criterion)

        plot_loss_curve(train_losses, val_losses, 'train_loss.png', 'val_loss.png')

    else:
        ### Test ###
        load_checkpoint(f, g, args.checkpoint)
        classifier = Classifier(in_dim, num_classes, classifier_hidden_dim).to(device)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])

        if not os.path.exists('checkpoints/classifier.pt'):
            ### Train Classifier ###
            train_dataset = datasets.STL10('./data', split='train', download=True, transform=data_transform)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_worker)

            criterion = nn.CrossEntropyLoss()
            
            if args.fine_tuning:
                params = list(f.parameters) + list(classifier.parameters())
            else:
                params = classifier.parameters()
            
            optimizer = torch.optim.Adam(params, lr = 1e-4)

            train_classifier(classifier_epochs, train_loader, f, classifier, criterion, optimizer)
            save_checkpoint_classifier(classifier, 'checkpoints/classifier.pt')

        else:
            load_checkpoint_classifier(classifier, 'checkpoints/classifier.pt')

        ### Test ###
        test_dataset = datasets.STL10('./data', split='test', download=True, transform=data_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_worker)
        
        accuracy = test(test_loader, f, classifier)
        print("Test Accuracy : %.4f"%(accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimCLR implementation")

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--c_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--p_dim', type=int, default=2048)
    parser.add_argument('--c_dim', type=int, default=1024)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--valid_size', type=float, default=0.05)
    parser.add_argument('--test', action = 'store_true', default = False)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt')
    parser.add_argument('--fine_tuning', action='store_true', default=False)    

    args = parser.parse_args()
    main(args)
