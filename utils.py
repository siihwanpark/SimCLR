import torch
import os

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

#x = torch.tensor([[1,1], [3,3], [5,5], [7,7], [9,9], [2,2], [4,4], [6,6], [8,8], [10,10]])
#perm = make_permutation(5)
#x = x[perm,:]
#print(x)