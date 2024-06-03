'''
This script is designed to load a pretrained SASRec model, extract user embeddings from the data, 
and save these embeddings to an output file.
'''
import torch
import os
import numpy as np
from model import SASRec
from utils import WarpSampler, data_partition
import argparse

'''
This first part defines arguments for various settings like dataset, training directory, 
hyperparameters, and model paths.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=139, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=200, type=int)
parser.add_argument('--hidden_units', default=768, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--state_dict_path', default=None, type=str)
args = parser.parse_args()

'''
This part includes: 
    - Data Partitioning: Loads the user and item sequences from the dataset specified by the user.
    - Sampler and Model Initialization: Initializes the WarpSampler for generating training batches 
      and the SASRec model with the specified parameters.
    - Load Pretrained Model: Loads the pretrained model weights and initializes the model with these weights, 
      moving it to the specified device and setting it to evaluation mode since it will not be trained further.
'''
# Load user and item sequences using data_partition
dataset = data_partition(args.dataset)
[user_train, _, _, usernum, itemnum] = dataset
# Initialize the WarpSampler
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
model = SASRec(usernum, itemnum, args).to(args.device)
# Path to the pretrained model
path_to_trained_model = '/home/hajar.laktaoui/ImplementationFolder/SASRec.pytorch/moviesAndTV_default/SASRec.epoch=201.lr=0.001.layer=2.head=1.hidden=768.maxlen=200.pth'
# Load the pretrained model
sd = torch.load(path_to_trained_model, map_location=torch.device('cpu'))
# Load state dict and then move the model to the specified device
model.load_state_dict(sd)
model.to(args.device)
model.eval()  # We are not going to train the model
'''
This part iterates over batches of user sequences, extracts user embeddings using the model, 
and concatenates them into a single tensor.
'''
# Determine the number of batches
num_batch = len(user_train) // args.batch_size
# Process each batch and save embeddings incrementally
user_embeddings = []
for step in range(num_batch):
    u, seq, pos, neg = sampler.next_batch()  # Get the next batch
    # Extract user embeddings
    seq = np.array(seq)
    seq_tensor = torch.tensor(seq, dtype=torch.long).to(args.device)
    embed_feat = model.extract_userEmd(seq_tensor)
    user_embeddings.append(embed_feat)

# Concatenate all user embeddings
user_emb = torch.cat(user_embeddings, dim=0)
'''
This section saves the user embeddings to our specified file path.
'''
# Save user embeddings
user_embeddings_path = '/home/hajar.laktaoui/ImplementationFolder/SASRec.pytorch/moviesAndTV_default/UserEmbeddings.pt'
torch.save(user_emb.cpu(), user_embeddings_path)  # Ensure tensor is on CPU before saving
print(user_emb.shape) # To verify the shape of user_emb

# Close the sampler
sampler.close()

