import os
import re
import math
import torch
import random
import pickle
import datetime
from rouge import rouge
from bleu import compute_bleu
import pandas as pd
import numpy as np
import json
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



def rouge_score(references, generated):
    """both are a list of strings"""
    score = rouge(generated, references)
    rouge_s = {k: (v * 100) for (k, v) in score.items()}
    '''
    "rouge_1/f_score": rouge_1_f,
    "rouge_1/r_score": rouge_1_r,
    "rouge_1/p_score": rouge_1_p,
    "rouge_2/f_score": rouge_2_f,
    "rouge_2/r_score": rouge_2_r,
    "rouge_2/p_score": rouge_2_p,
    "rouge_l/f_score": rouge_l_f,
    "rouge_l/r_score": rouge_l_r,
    "rouge_l/p_score": rouge_l_p,
    '''
    return rouge_s
def bleu_score(references, generated, n_gram=4, smooth=False):
    """a list of lists of tokens"""
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for (wa, wb) in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_list = []
        for i in ids:
            if i in feature_set:
                feature_list.append(i)
        feature_batch.append(set(feature_list))

    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for (fea_set, fea) in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1

    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb

    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)

    total_count = 0
    for i, x in enumerate(feature_batch):
        for j in range(i + 1, list_len):
            y = feature_batch[j]
            total_count += len(x & y)

    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def mean_absolute_error(predicted, max_r, min_r, mae=True):
    total = 0
    for (r, p) in predicted:
        if p > max_r:
            p = max_r
        if p < min_r:
            p = min_r

        sub = p - r
        if mae:
            total += abs(sub)
        else:
            total += sub ** 2

    return total / len(predicted)


def root_mean_square_error(predicted, max_r, min_r):
    mse = mean_absolute_error(predicted, max_r, min_r, False)
    return math.sqrt(mse)


class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)


class DataLoader:
    def __init__(self, data_path, index_dir, tokenizer, seq_len):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            (fea, adj, tem, sco) = review['template']
            tokens = self.tokenizer(tem)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])  # keep seq_len tokens at most
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text,
                         'feature': fea})
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index


class Batchify:
    def __init__(self, data, tokenizer, bos, eos, batch_size=128, shuffle=False):
        u, i, r, t, self.feature = [], [], [], [], []
        for x in data:
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append('{} {} {}'.format(bos, x['text'], eos))
            self.feature.append(x['feature'])

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        return user, item, rating, seq, mask


class Batchify2:
    def __init__(self, data, user2feature, item2feature, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):
        t, self.feature, features = [], [], []
        for x in data:
            ufea = set(user2feature[x['user']])
            ifea = set(item2feature[x['item']])
            intersection = ufea & ifea
            difference = ufea | ifea - intersection
            features.append(' '.join(list(intersection) + list(difference)))
            t.append('{} {} {}'.format(bos, x['text'], eos))
            self.feature.append(x['feature'])

        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        self.mask = encoded_inputs['attention_mask'].contiguous()
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        seq = self.seq[index]  # (batch_size, seq_len)
        mask = self.mask[index]
        prompt = self.prompt[index]
        return seq, mask, prompt

class DataLoader2:
    def __init__(self, data_path, index_dir, tokenizer, tokenizerFast, seq_len, n_clusters=5):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()
        self.max_rating = float('-inf')
        self.min_rating = float('inf')
        self.initialize(data_path)
        self.feature_set = set()
        self.tokenizer = tokenizer
        self.tokenizerFast = tokenizerFast
        self.seq_len = seq_len
        self.n_clusters = n_clusters  # Number of clusters for feature topics
        self.user2feature = {}  # Initialize empty dictionary
        self.item2feature = {}  # Initialize empty dictionary
        # Cluster features into topics
        self.feature_topics = self.cluster_features()
        self.metadata_path = '/content/drive/MyDrive/SFTmodel/filtered_combined_metadata.json'
        #'/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/filtered_combined_metadataMovieAndTv.json'
        #  '/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/filtered_combined_metadata.json'
        self.train, self.valid, self.test, self.user2feature, self.item2feature = self.load_data(data_path, index_dir)

    def initialize(self, data_path):
        assert os.path.exists(data_path)
        reviews = pickle.load(open(data_path, 'rb'))
        for review in reviews:
            self.user_dict.add_entity(review['user'])
            self.item_dict.add_entity(review['item'])
            rating = review['rating']
            if self.max_rating < rating:
                self.max_rating = rating
            if self.min_rating > rating:
                self.min_rating = rating

    # def load_metadata(self, metadata_path):
    #     """
    #     Load metadata from a separate JSON file and create a lookup dictionary.
    #     """
    #     with open(metadata_path, 'r') as f:
    #         metadata = json.load(f)
        
    #     # Organize metadata into dictionaries for users and items
    #     user_metadata = {entry['userID']: entry for entry in metadata if 'userID' in entry}
    #     item_metadata = {entry['hotelID']: entry for entry in metadata if 'hotelID' in entry}
    
    #     return user_metadata, item_metadata
    def cluster_features(self):
        """Cluster features into topics using TF-IDF and K-Means."""
        all_features = list(self.feature_set)
        if not all_features:
            print("Feature set is empty. Cannot perform clustering.")
            return {}

        vectorizer = TfidfVectorizer()
        try:
            feature_vectors = vectorizer.fit_transform(all_features)
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(feature_vectors)
            feature_to_topic = {feature: cluster_labels[idx] for idx, feature in enumerate(all_features)}
            print("Clustered features:", feature_to_topic)
            return feature_to_topic
        except ValueError as e:
            print(f"Error during clustering: {e}")
            return {}

    def select_features(self, user_features, item_features, feature_to_topic, relevant_topics):
        """Select relevant features based on topics and metadata relevance."""
        combined_features = set(user_features) | set(item_features)
        selected = [feature for feature in combined_features if feature_to_topic.get(feature) in relevant_topics]
        print("Selected features:", selected)
        return selected

    def load_data(self, data_path, index_dir):
        data = []
        reviews = pickle.load(open(data_path, 'rb'))
        # user_metadata, item_metadata = self.load_metadata(self.metadata_path)
        
        for review in reviews:
            # user_id = review.get('user')
            # item_id = review.get('item')
            # user_meta = user_metadata.get(user_id, {})
            # item_meta = item_metadata.get(item_id, {})
            # # user_features = [f.lower() for f in user_features if isinstance(f, str)]
            # # item_features = [f.lower() for f in item_features if isinstance(f, str)]
            # # self.feature_set.update(user_features + item_features)

            # # Process metadata and features
            # relevant_topics = set()  # Define logic to infer topics
            # selected_features = self.select_features(user_features, item_features, self.feature_topics, relevant_topics)

            # Combine metadata into text
            # metadata_text = f"""
            # User ID: {user_meta.get('userID', 'Unknown')}
            # User Name: {user_meta.get('userName', 'Unknown')}
            # User Location: {user_meta.get('userLocation', 'Unknown')}
            # Hotel ID: {item_meta.get('hotelID', 'Unknown')}
            # Hotel Title: {item_meta.get('hotelTitle', 'Unknown')}
            # Hotel City: {item_meta.get('hotelCity', 'Unknown')}
            # Travel Type: {item_meta.get('travelType', 'Unknown')}
            # Travel Date: {item_meta.get('travelDate', 'Unknown')}
            # """
            
            # Tokenize and embed metadata with FastGPT-2
            # tokens = self.tokenizerFast(metadata_text, return_tensors="pt", truncation=True, max_length=200)
            # metadata_embedding = tokens.get('input_ids', [])
            # metadata_embedding = torch.mean(tokens.float(), dim=1).tolist()  # Average embedding for simplicity

          
            (fea, adj, tem, sco) = review['template']
            tokens = self.tokenizer(tem)['input_ids']
            text = self.tokenizer.decode(tokens[:self.seq_len])  # keep seq_len tokens at most
            # tokensfea = self.tokenizer(fea)['input_ids']
            # fea = self.tokenizer.decode(tokensfea[:self.seq_len])  # keep seq_len tokens at most
            data.append({'user': self.user_dict.entity2idx[review['user']],
                         'item': self.item_dict.entity2idx[review['item']],
                         'rating': review['rating'],
                         'text': text,
                         'feature': fea }) #,'concepts': concepts ,
                        # 'metadata': metadata_embedding
            # self.feature_set.add(fea)
            self.feature_set.add(fea)

        train_index, valid_index, test_index = self.load_index(index_dir)
        train, valid, test = [], [], []
        user2feature, item2feature = {}, {}
        for idx in train_index:
            review = data[idx]
            train.append(review)
            u = review['user']
            i = review['item']
            f = review['feature']
            if u in user2feature:
                user2feature[u].append(f)
            else:
                user2feature[u] = [f]
            if i in item2feature:
                item2feature[i].append(f)
            else:
                item2feature[i] = [f]
        for idx in valid_index:
            valid.append(data[idx])
        for idx in test_index:
            test.append(data[idx])
        # return train, valid, test
        return train, valid, test, user2feature, item2feature

    def load_index(self, index_dir):
        assert os.path.exists(index_dir)
        with open(os.path.join(index_dir, 'train.index'), 'r') as f:
            train_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'validation.index'), 'r') as f:
            valid_index = [int(x) for x in f.readline().split(' ')]
        with open(os.path.join(index_dir, 'test.index'), 'r') as f:
            test_index = [int(x) for x in f.readline().split(' ')]
        return train_index, valid_index, test_index

class Batchify3:
    def __init__(self, data, user2feature, item2feature, tokenizer, bos, eos, seq_len, batch_size=128, shuffle=False):#tokenizerFast, 
        u, i, r, t, self.feature, features, self.context = [], [], [], [], [], [], []
        
        for x in data:
            ufea = set(user2feature[x['user']])
            ifea = set(item2feature[x['item']])
            intersection = ufea & ifea
            difference = ufea | ifea - intersection
            features.append(' '.join(list(intersection)))
            self.feature.append(' '.join(x['feature']))
            u.append(x['user'])
            i.append(x['item'])
            r.append(x['rating'])
            t.append('{} {} {}'.format(bos, x['text'], eos))
            # Ensure metadata is converted to 1D tensors
            # if not intersection and not difference:
            #     features.append("Unknown")
            # else:
            #     features.append(' '.join(list(intersection) + list(difference)))

            # self.feature.append(x['feature'])
            # if not intersection and not difference:
            #     features.append("Unknown")
            # else:
            #     features.append(' '.join(list(intersection)))
            # if isinstance(x['metadata'], torch.Tensor):
            #     self.context.append(x['metadata'].view(-1))
            # else:
            #     self.context.append(torch.tensor(x['metadata'], dtype=torch.long))
        # print(t[:2])
        # # # print(self.feature [:5])
        # print(features [:2])
        # print(self.context[:3])
        encoded_inputs = tokenizer(t, padding=True, return_tensors='pt')
        self.seq = encoded_inputs['input_ids'].contiguous()
        # print(self.seq.shape)
        self.mask = encoded_inputs['attention_mask'].contiguous()
        self.user = torch.tensor(u, dtype=torch.int64).contiguous()
        self.item = torch.tensor(i, dtype=torch.int64).contiguous()
        self.rating = torch.tensor(r, dtype=torch.float).contiguous()
        # print(self.feature [:5])
        encoded_features = tokenizer(features, padding=True, return_tensors='pt')
        self.prompt = encoded_features['input_ids'][:, :seq_len].contiguous()
        # print(self.prompt[:3])
        # self.context = torch.tensor(self.context, dtype=torch.float32).contiguous()
        # Ensure all metadata tensors have the same shape
        # max_metadata_length = 200  # Define a maximum length for metadata embeddings
        # self.context = [
        #     torch.cat([c, torch.zeros(max_metadata_length - len(c), dtype=torch.long)])
        #     if len(c) < max_metadata_length else c[:max_metadata_length]
        #     for c in self.context
        #     ]
        # self.context = torch.stack(self.context)  # Stack into a single tensor
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_step = int(math.ceil(self.sample_num / self.batch_size))
        self.step = 0

    def next_batch(self):
        if self.step == self.total_step:
            self.step = 0
            if self.shuffle:
                random.shuffle(self.index_list)

        start = self.step * self.batch_size
        offset = min(start + self.batch_size, self.sample_num)
        self.step += 1
        index = self.index_list[start:offset]
        user = self.user[index]  # (batch_size,)
        item = self.item[index]
        rating = self.rating[index]
        seq = self.seq[index]  # (batch_size, seq_len)
        feat = self.prompt[index]
        # context = self.context[index]
        mask = self.mask[index]
        return user, item, rating, seq, feat, mask


def now_time():
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def postprocessing(string):
    '''
    adopted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub('\'s', ' \'s', string)
    string = re.sub('\'m', ' \'m', string)
    string = re.sub('\'ve', ' \'ve', string)
    string = re.sub('n\'t', ' n\'t', string)
    string = re.sub('\'re', ' \'re', string)
    string = re.sub('\'d', ' \'d', string)
    string = re.sub('\'ll', ' \'ll', string)
    string = re.sub('\(', ' ( ', string)
    string = re.sub('\)', ' ) ', string)
    string = re.sub(',+', ' , ', string)
    string = re.sub(':+', ' , ', string)
    string = re.sub(';+', ' . ', string)
    string = re.sub('\.+', ' . ', string)
    string = re.sub('!+', ' ! ', string)
    string = re.sub('\?+', ' ? ', string)
    string = re.sub(' +', ' ', string).strip()
    return string

def ids2tokensReal(ids, tokenizer, eos_token="<eos>", pad_token="<pad>"):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    # for token in text.split():
    for token in re.findall(f"{eos_token}|{pad_token}|[^\s<]+", text):
        if token == eos_token:
            break
        tokens.append(token)
    return tokens

def ids2tokens(ids, tokenizer, eos):
    text = tokenizer.decode(ids)
    text = postprocessing(text)  # process punctuations: "good!" -> "good !"
    tokens = []
    for token in text.split():
        if token == eos:
            break
        tokens.append(token)
    return tokens

    
