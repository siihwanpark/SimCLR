import torch
import os
import matplotlib.pyplot as plt

def make_permutation(N):
    return [i//2 if (i % 2 == 0) else N+i//2 for i in range(2*N)]

def save_checkpoint(encoder, projection_head, path):
    model_state = {
        'encoder_state_dict': encoder.state_dict(),
        'projection_head_state_dict': projection_head.state_dict()
    }

    torch.save(model_state, path)
    print('A check point has been generated : ' + path)

def load_checkpoint(encoder, projection_head, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        print("trained encoder and projection head " + path + " has been loaded successfully.")
    else:
        raise NameError("There's no such directory or file : " + path)

def save_checkpoint_classifier(classifier, path):
    model_state = {
        'state_dict': classifier.state_dict()
    }

    torch.save(model_state, path)
    print('A check point for classifier has been generated : ' + path)

def load_checkpoint_classifier(classifier, path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        classifier.load_state_dict(checkpoint['state_dict'])
        print('trained classifier ' + path + ' has been loaded successfully.')
    else:
        raise NameError("There's no such directory or file : " + path)

def plot_loss_curve(train_losses, val_losses, path1, path2):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='train loss')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.set(title="Train Loss Curve")
    ax.grid()

    fig.savefig(path1)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(val_losses, label='validation loss')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.set(title="Validation Loss Curve")
    ax.grid()

    fig.savefig(path2)
    plt.close()