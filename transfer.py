import os, argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from src.model import Classifier, resnet18_encoder, ProjectionHead
from src.procedures import train_classifier, test
from src.utils import load_checkpoint, save_checkpoint_classifier, load_checkpoint_classifier


def main(args):
    in_dim = 512
    projection_hidden_dim = 2048
    classifier_hidden_dim = 1024
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])

    if args.dataset == 'CIFAR-10':
        classifier_path = 'checkpoints/classifier_cifar10.pt'
        num_classes = 10
        train_dataset = datasets.CIFAR10(
            './data', train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR10(
            './data', train=False, download=True, transform=data_transform)
    elif args.dataset == 'CIFAR-100':
        classifier_path = 'checkpoints/classifier_cifar100.pt'
        num_classes = 100
        train_dataset = datasets.CIFAR100(
            './data', train=True, download=True, transform=data_transform)
        test_dataset = datasets.CIFAR100(
            './data', train=False, download=True, transform=data_transform)
        
    f, g = resnet18_encoder().to(device), ProjectionHead(in_dim, projection_hidden_dim).to(device)
    load_checkpoint(f, g, args.checkpoint)
    
    classifier = Classifier(in_dim, num_classes, classifier_hidden_dim).to(device)

    
    if not os.path.exists(classifier_path):
        ### Train Classifier ###
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_worker)

        criterion = nn.CrossEntropyLoss()

        if args.fine_tuning:
            params = list(f.parameters()) + list(classifier.parameters())
        else:
            params = classifier.parameters()

        optimizer = torch.optim.Adam(params, lr=1e-4)

        train_classifier(args.epochs, train_loader,
                            f, classifier, criterion, optimizer)
        save_checkpoint_classifier(classifier, classifier_path)

    else:
        load_checkpoint_classifier(classifier, classifier_path)

    ### Test ###
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_worker)

    accuracy = test(test_loader, f, classifier)
    print("Test Accuracy : %.4f" % (accuracy))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SimCLR implementation")

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/best.pt')
    parser.add_argument('--fine_tuning', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='CIFAR-10',
                        choices=['CIFAR-10', 'CIFAR-100'])

    args = parser.parse_args()
    main(args)
