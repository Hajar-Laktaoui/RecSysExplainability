import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
########## For the pickle file ##########
''' Data loading and processing for the pickle file :
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
# def load_data(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
    
#     ratings = []
#     user_map = {}
#     item_map = {}
#     user_counter = 0
#     item_counter = 0
#     for entry in data:
#         user = entry['user']
#         item = entry['item']
#         rating = entry['rating']
#         if user not in user_map:
#             user_map[user] = user_counter
#             user_counter += 1
#         if item not in item_map:
#             item_map[item] = item_counter
#             item_counter += 1
#         rating_class = int(rating) - 1  # Assuming ratings are from 1 to 5
#         ratings.append((user_map[user], item_map[item], rating_class))
    
#     ratings = np.array(ratings)
#     print("Total num_users: ", len(user_map), "Total num_items: ", len(item_map))
#     return ratings, user_map, item_map

def load_data(file_path):
    """
    The load_data function reads the user-item interaction data from a file and removes the timestamp column.
    
    Args:
        file_path (str): Path to the dataset file.
        
    Returns:
        df (DataFrame): DataFrame containing only the user_id, item_id, and rating.
    """
    columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(file_path, sep='\t', names=columns)
    df = df.drop('timestamp', axis=1)
    return df

def preprocess_data(df):
    """
    The preprocess_data function maps the user and item IDs to unique indices, adjusts ratings, and prepares the data for training.
    
    Args:
        df (DataFrame): DataFrame containing user_id, item_id, and rating.
        
    Returns:
        ratings (ndarray): Numpy array with columns for user, item, and adjusted rating.
        user_map (dict): Dictionary mapping original user_ids to unique indices.
        item_map (dict): Dictionary mapping original item_ids to unique indices.
    """
    user_map = {user_id: idx for idx, user_id in enumerate(df['user_id'].unique())}
    item_map = {item_id: idx for idx, item_id in enumerate(df['item_id'].unique())}
    df['user'] = df['user_id'].map(user_map)
    df['item'] = df['item_id'].map(item_map)
    df['rating'] = df['rating'] - 1
    ratings = df[['user', 'item', 'rating']].values
    print("Total num_users: ", len(user_map), "Total num_items: ", len(item_map))
    return ratings, user_map, item_map


def split_data(ratings):
    """
    Splitting data into training, validation, and test sets:

    Args:
        - ratings (ndarray): Numpy array containing tuples of (mapped user index, mapped item index, adjusted rating).

    Returns:
        - train_data (ndarray): Numpy array containing the training data.
        - val_data (ndarray): Numpy array containing the validation data.
        - test_data (ndarray): Numpy array containing the test data.
    """
    train_data, val_data = train_test_split(ratings, test_size=0.2, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data


class MF(nn.Module):
    """
    Matrix Factorization Model:
    This class defines a matrix factorization model, which resembles a simplified two-tower architecture. 
    The model learns embeddings for users and items and predicts ratings based on the dot product of these embeddings.

    Args:
        - num_users (int): The number of unique users in the dataset.
        - num_items (int): The number of unique items in the dataset.
        - mf_dim (int): The dimensionality of the embeddings for users and items.

    Methods:
        - __init__(self, num_users, num_items, mf_dim): Initializes the user and item embeddings and their weights.
        - forward(self, user, item=None): Computes the predicted rating for a given user-item pair or for all items for a given user.

    Returns:
        - Predicted ratings as sigmoid-activated values.
    """
    def __init__(self, num_users, num_items, mf_dim):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, mf_dim)
        self.item_embedding = nn.Embedding(num_items, mf_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user, item=None):
        user_embedding = self.user_embedding(user)
        if item is not None:
            item_embedding = self.item_embedding(item)
            pred = (user_embedding * item_embedding).sum(1)
            return torch.sigmoid(pred)
        else:
            all_items = torch.arange(self.item_embedding.num_embeddings, device=user.device)
            item_embedding = self.item_embedding(all_items)
            pred = torch.matmul(user_embedding, item_embedding.T)
            return torch.sigmoid(pred)


def get_train_instances(train, num_negatives, num_items):
    """
    Generating Training Instances with Negative Sampling:
    This function creates training instances for the model, including negative samples for implicit feedback learning. 
    For each observed user-item interaction, it generates multiple negative samples (items that the user has not interacted with) to help the model distinguish between relevant and irrelevant items.

    Args:
        - train (list of tuples): The training data containing (user, item, rating) tuples.
        - num_negatives (int): The number of negative samples to generate for each positive instance.
        - num_items (int): The total number of unique items in the dataset.

    Returns:
        - user_input (list): A list of user indices for training.
        - item_input (list): A list of item indices for training.
        - labels (list): A list of labels (1 for positive instances, 0 for negative instances).
    """
    user_input, item_input, labels = [], [], []
    train_set = set((u, i) for (u, i, _) in train)
    for (u, i, _) in train:
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_set:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

def get_negative_samples(dataRatings, train_data, num_items, num_negatives):
    """
    Generating Negative Samples for Each User:
    This function creates a list of negative samples (items that a user has not interacted with) for each user in the dataset. These samples are used to train the model by distinguishing between interacted and non-interacted items.

    Args:
        - dataRatings (list of tuples): The data containing user-item pairs for which negative samples are to be generated.
        - train_data (list of tuples): The training data containing (user, item, rating) tuples.
        - num_items (int): The total number of unique items in the dataset.
        - num_negatives (int): The number of negative samples to generate for each user.

    Returns:
        - NegativesSamples (list of lists): A list where each element is a list of negative item indices for a corresponding user in `dataRatings`.
    """
    train_set = set((u, i) for u, i, *_ in train_data)    
    NegativesSamples = []
    for u, _ in dataRatings:
        negatives = []
        for _ in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train_set:
                j = np.random.randint(num_items)
            negatives.append(j)
        NegativesSamples.append(negatives)
    return NegativesSamples

def evaluate_model(model, testRatings, testNegatives, K, device):
    """
    Evaluating Model Performance:
    This function evaluates the performance of the model on a set of test ratings using two metrics: Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG). For each user in the test set, it computes scores for both the true item and a set of negative items, ranks them, and calculates the HR and NDCG values.

    Args:
        - model (nn.Module): The trained recommendation model used for generating predictions.
        - testRatings (list of tuples): The test data containing user-item pairs and the true item to be evaluated.
        - testNegatives (list of lists): A list where each element contains negative item indices for a corresponding user in `testRatings`.
        - K (int): The number of top items to consider for ranking.
        - device (torch.device): The device (CPU or GPU) to which tensors should be moved for evaluation.

    Returns:
        - (float, float): The mean Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG) over all users in the test set.
    """
    model.eval()
    hits = []
    ndcgs = []
    for idx, (user, true_item) in enumerate(testRatings):
        neg_items = testNegatives[idx]
        items_to_evaluate = neg_items + [true_item]
        item_tensor = torch.LongTensor(items_to_evaluate).to(device)
        user_tensor = torch.LongTensor([user] * len(items_to_evaluate)).to(device)
        
        with torch.no_grad():
            scores = model(user_tensor, item_tensor).cpu().numpy()
        
        item_score_dict = {item: score for item, score in zip(items_to_evaluate, scores)}
        ranklist = heapq.nlargest(K, item_score_dict, key=item_score_dict.get)
        
        hr = getHitRatio(ranklist, true_item)
        ndcg = getNDCG(ranklist, true_item)
        
        hits.append(hr)
        ndcgs.append(ndcg)
    return np.mean(hits), np.mean(ndcgs)

def getHitRatio(ranklist, gtItem):
    """
    Calculating Hit Ratio:
    This function computes whether the true item (ground truth item) is present in the top-K ranked items. The Hit Ratio (HR) is `1` if the true item is found in the ranked list, otherwise `0`.

    Args:
        - ranklist (list of int): The list of item indices ranked by the model's predicted scores.
        - gtItem (int): The true item index that we are checking for presence in the ranked list.

    Returns:
        - int: `1` if the true item is in the ranklist, otherwise `0`.
    """
    return 1 if gtItem in ranklist else 0

def getNDCG(ranklist, gtItem):
    """
    Calculating Normalized Discounted Cumulative Gain (NDCG):
    This function calculates the NDCG score for the true item within the ranked list. NDCG measures the relevance of the true item based on its position in the ranking, with higher relevance for items appearing earlier in the list.

    Args:
        - ranklist (list of int): The list of item indices ranked by the model's predicted scores.
        - gtItem (int): The true item index whose NDCG score is being calculated.

    Returns:
        - float: The NDCG score for the true item. Returns `0` if the true item is not in the ranklist.
    """
    for i, item in enumerate(ranklist):
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0

def train_model(model, train_data, valRatings, valNegatives, num_epochs, batch_size, num_negatives, num_items, optimizer, scheduler, loss_function, topK, device, model_out_file):
    best_hr, best_ndcg, best_epoch = 0, 0, 0 # To save the model with the best performance.
    patience, wait = 5, 0  # Early stopping parameters to prevent overfitting.
    '''
    Loop over epochs to train the model:
        - For each epoch, the model processes training data in batches. 
        - It calculates the loss for each batch, performs backpropagation, 
          and updates the model parameters. 
    '''
    for epoch in range(num_epochs):
        model.train()
        user_input, item_input, labels = get_train_instances(train_data, num_negatives, num_items)
        epoch_loss = 0
        for step in range(0, len(user_input), batch_size):
            batch_user = torch.LongTensor(user_input[step:step + batch_size]).to(device)
            batch_item = torch.LongTensor(item_input[step:step + batch_size]).to(device)
            batch_labels = torch.FloatTensor(labels[step:step + batch_size]).to(device)
            optimizer.zero_grad()
            outputs = model(batch_user, batch_item)
            loss = loss_function(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Print the loss for each step
            print(f'Epoch {epoch + 1}, Step {step // batch_size + 1}, Loss: {loss.item():.4f}')
        scheduler.step()
        #Print the average loss for each epoch.
        print(f'Epoch {epoch + 1}, Avrg Loss: {np.mean(epoch_loss):.4f}')
        '''
        Evaluate the model's performance on validation data at the end of each epoch. 
        - Print the HR and NDCG scores.
        - Then, if the current HR is the best seen so far, save the model's state and reset the patience counter.
        - If no improvement is observed for a number of epochs equal to the patience parameter, perform early stopping.
        - Print the best evaluation metrics and epoch at the end of training.

        '''
        hr, ndcg = evaluate_model(model, valRatings, valNegatives, topK, device)
        print(f'Validation Epoch {epoch + 1}: HR = {hr:.4f}, NDCG = {ndcg:.4f}')
        if hr > best_hr:
            best_hr, best_ndcg, best_epoch = hr, ndcg, epoch
            torch.save(model.state_dict(), model_out_file)
            print(f'Model saved to {model_out_file}')
            wait = 0  # Reset the wait counter if we have a new best
        else:
            wait += 1  # Increment the wait counter if no improvement
            if wait >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break
    # Print the best evaluation metrics and the corresponding epoch
    print(f'End. Best Epoch {best_epoch + 1}: HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}')

def main():
    '''
    Main function to run MF model:
    - Load and preprocess the dataset.
    - Split data into training, validation, and test sets.
    - Initialize model, optimizer, and scheduler.
    - Compute initial evaluation metrics on the test set.
    - Train the model on the training data.
    - Evaluate the trained model on the test set and print results.
    '''
    file_path = '/home/hajar.laktaoui/ImplementationFolder/ml-1k.data'
    df = load_data(file_path)
    ratings, user_map, item_map = preprocess_data(df)
    train_data, val_data, test_data = split_data(ratings)
    # file_path = '/home/hajar.laktaoui/ImplementationFolder/MoviesAndTV/reviews.pickle' #'/home/hajar.laktaoui/ImplementationFolder/TripAdvisor.pickle'
    # ratings, user_map, item_map = load_data(file_path)
    # train_data, val_data, test_data = split_data(ratings)
    #print(f'len of training set : {len(train_data):.4f}')

    num_users = len(user_map)
    num_items = len(item_map)
    num_epochs = 10
    batch_size = 128
    mf_dim = 75
    num_negatives = 100
    learning_rate = 0.0001 #0.0001713536406050813
    lambda_reg = 0.01 #0.0007522644876939285
    topK = 10
    model_out_file = 'MF_model.pth12'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MF(num_users, num_items, mf_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lambda_reg)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    loss_function = nn.BCELoss()
    
    testRatings = [(u, i) for u, i, r in test_data]
    testNegatives = get_negative_samples(testRatings, train_data, num_items, num_negatives)
    valRatings = [(u, i) for u, i, r in val_data]
    valNegatives = get_negative_samples(valRatings, train_data, num_items, num_negatives)
    
    # print(f'Debug: Number of valRatings: {len(valRatings)}')
    # print(f'Debug: Number of valNegatives: {len(valNegatives)}')
    
    hr, ndcg = evaluate_model(model, testRatings, testNegatives, topK, device)
    print(f'Initial: HR = {hr:.4f}, NDCG = {ndcg:.4f}')
    debug_negative_sampling(train_data, num_negatives, num_items)

    train_model(model, train_data, valRatings, valNegatives, num_epochs, batch_size, num_negatives, num_items, optimizer, scheduler, loss_function, topK, device, model_out_file)

    model.load_state_dict(torch.load(model_out_file))
    hr, ndcg = evaluate_model(model, testRatings, testNegatives, topK, device)
    print(f'Test: HR = {hr:.4f}, NDCG = {ndcg:.4f}')

if __name__ == '__main__':
    main()

