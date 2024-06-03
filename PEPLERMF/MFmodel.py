import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pickle


''' Data loading and processing :
This part loads data from a pickle file, processes it to map user and item IDs to numerical indices, 
and stores the processed ratings in a list of tuples. 
Each tuple contains a mapped user index, a mapped item index, and the corresponding rating
Args:
    - ratings: list that will store the processed rating data.
    - user_map: dictionary that will map  user IDs to numerical indices.
    - item_map: dictionary that will map item IDs to numerical indices.
    - user_counter: counter to track the number of unique users.
    - item_counter: counter to track the number of unique items.
'''

with open('/home/hajar.laktaoui/ImplementationFolder/MoviesAndTV/reviews.pickle', 'rb') as f:
    data = pickle.load(f)     

ratings = []
user_map = {}
item_map = {}
user_counter = 0
item_counter = 0

for entry in data:
    user = entry['user']
    item = entry['item']
    rating = entry['rating']

    if user not in user_map:
        user_map[user] = user_counter
        user_counter += 1
    if item not in item_map:
        item_map[item] = item_counter
        item_counter += 1

    ratings.append((user_map[user], item_map[item], rating))


''' Matrix Factorization Model class:
This class defines a matrix factorization model that learns low-dimensional embeddings for users and items, 
which are then used to predict ratings by computing the dot product of the user and item embeddings.

Args:
    num_users: The number of unique users.
    num_items: The number of unique items.
    embedding_dim: The dimensionality of the embeddings for both users and items.
    user_ids: A tensor containing user IDs.
    item_ids: A tensor containing item IDs.

Returns:
    The predicted ratings.
'''

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, user_ids, item_ids):
        user_factors = self.user_embeddings(user_ids)  # (batch_size, embedding_dim)
        item_factors = self.item_embeddings(item_ids)  # (batch_size, embedding_dim)
        # Dot product for predicted ratings (unnormalized)
        predicted_ratings = torch.mul(user_factors, item_factors).sum(dim=1)
        return predicted_ratings


'''
This class defines a custom Dataset to handle rating data, facilitating the use of DataLoader for batching and shuffling.
Args:
    ratings: A list of tuples, where each tuple contains a user ID, an item ID, and a rating.
Methods:
    __init__(self, ratings): Initializes the dataset with the provided ratings data.
    __len__(self): Returns the number of ratings in the dataset.
    __getitem__(self, idx): Retrieves the user ID, item ID, and rating at the specified index idx and returns them as tensors.
Returns:
    Tensors for user ID, item ID, and rating, which can be used for training a model.
'''

class RatingDataset(Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user_id, item_id, rating = self.ratings[idx]
        return (torch.tensor(user_id, dtype=torch.long),
                torch.tensor(item_id, dtype=torch.long),
                torch.tensor(rating, dtype=torch.float))

''' Build and training the model:
This part sets up and trains the matrix factorization model by:
Dataset and Data Loader:
   Initializes the custom `RatingDataset` with the `ratings` data.
   Creates a `DataLoader` for batching and shuffling the dataset.
Model and Optimizer Initialization:
   Determines the number of unique users and items.
   Sets the embedding dimension for the model.
   Moves the model to GPU if available.
   Initializes the Adam optimizer.
Training Loop:
   Runs for a specified number of epochs.
   For each batch:
     - Moves the data to the appropriate device.
     - Performs a forward pass to get predicted ratings.
     - Computes the MSE loss.
     - Performs backpropagation and updates the model parameters.
   Prints the loss at the end of each epoch.
Returns:
    Trains a `MatrixFactorization` model to predict ratings by learning user and item embeddings.
'''
# Create the dataset and data loader
dataset = RatingDataset(ratings)
data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

# Initialize the model and optimizer
num_users = len(user_map)
num_items = len(item_map)
embedding_dim = 768  # Adjust embedding_dim to emsize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MatrixFactorization(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for user_ids, item_ids, target_ratings in data_loader:
        user_ids, item_ids, target_ratings = user_ids.to(device), item_ids.to(device), target_ratings.to(device)
        optimizer.zero_grad()

        # Forward pass
        predicted_ratings = model(user_ids, item_ids)

        # Loss function (e.g., Mean Squared Error)
        loss = nn.functional.mse_loss(predicted_ratings, target_ratings)

        # Backward pass and parameter update
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch: {epoch+1:03d}, Loss: {loss.item():.4f}')


''' Extracting and saving user and item embeddings
This part saves the trained user and item embeddings from the matrix factorization model by: 
Extract Embeddings:
   - Retrieves user and item embeddings from the trained model.
   - Moves the embeddings to CPU 
Save Embeddings:
   - Saves the user embeddings to `user_embeddings.pt`.
   - Saves the item embeddings to `item_embeddings.pt`.
'''
user_embeddings = model.user_embeddings.weight.detach().cpu()
item_embeddings = model.item_embeddings.weight.detach().cpu()


# Save the user and item embeddings to .pt files
torch.save(user_embeddings, 'user_embeddings.pt')
torch.save(item_embeddings, 'item_embeddings.pt')

# verify shape of item and user embeddings
print(f'User Embeddings shape: {user_embeddings.shape}')
print(f'Item Embeddings shape: {item_embeddings.shape}')
