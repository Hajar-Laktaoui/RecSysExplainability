from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch
import copy
from sasrec_model import SASRec
import json
import numpy as np
from torch.nn.functional import log_softmax

'''
This class initializes item embeddings using pretrained weights from SASRec model 
and freezes the embedding weights. Then, it then provides a forward method to extract item 
embeddings from input item indices.
'''
class SasrecItemEmbeddings(nn.Module):
    def __init__(self, sasrec_hidden_untis, emsize):
        super().__init__()

        path_to_trained_sasrec = '/home/hajar.laktaoui/Implementation/Implementation4TkL/SASRec.pytorchTest/TripAdvisor_default/SASRec.epoch=40.lr=0.001.layer=2.head=1.hidden=100.maxlen=200.pth'
        # /content/drive/MyDrive/SASRec.pytorch-master/TripAdvisorGP_default/SASRec.epoch=80.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth'
        #'/home/hajar.laktaoui/Implementation/Implementation4TkL/SASRec.pytorchTest/moviesAndTV_default/SASRec.epoch=80.lr=0.001.layer=2.head=1.hidden=100.maxlen=200.pth'
        # '
        # '/home/hajar.laktaoui/Implementation/Implementation4TkL/SASRec.pytorchTest/Yelp_default/SASRec.epoch=500.lr=0.001.layer=2.head=1.hidden=100.maxlen=200.pth'
        sasrec_weights = torch.load(path_to_trained_sasrec, map_location='cpu')
        self.item_embeddingsAll = nn.Embedding.from_pretrained(sasrec_weights['item_emb.weight'])
        # Exclude the padding index (index 0) by taking the weights from index 1 onwards
        self.item_embeddings = nn.Embedding.from_pretrained(self.item_embeddingsAll.weight[1:])
        
        # Freeze item embeddings weights
        for p in self.item_embeddings.parameters():
            p.requires_grad = False
         # Add linear layer to adapt sasrec_hidden_units to pepler_hidden_units
        self.proj_item = nn.Linear(sasrec_hidden_untis, emsize)
    
    def forward(self, item_embeds):
        item_embeds = self.item_embeddings(item_embeds)  # Subtract 1 to match LLM indexing
        item_embeds = self.proj_item(item_embeds)
        return item_embeds

'''
This class initializes user embeddings using pretrained weights extracted from the SASRec model 
and freezes them. Then, it provides a forward method to extract user embeddings from input user indices.
'''

class SasrecUserEmbeddings(nn.Module):
    def __init__(self, sasrec_hidden_untis, emsize):
        super().__init__()

        path_to_trained_sasrec = '/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/UserEmbRatingsTripAdvisor.pt'
        # '/content/drive/MyDrive/SFTmodel/UserEmbTripAdvisor.pt'
        #'/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/UserEmbRatingsMoviesAndTV.pt'
        # '/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/UserEmbRatingsTripAdvisor.pt'
        # '/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/UserEmbeddingsAndRatingsYelp3.pt'
        loaded_data = torch.load(path_to_trained_sasrec, map_location='cpu')
        user_weights = loaded_data['user_embeddings']
        self.user_embeddings = nn.Embedding.from_pretrained(user_weights)
        
        # Freeze user embeddings weights
        for p in self.user_embeddings.parameters():
            p.requires_grad = False
        self.proj_user = nn.Linear(sasrec_hidden_untis, emsize)
    
    def forward(self, user_embeds):
        user_embeds = self.user_embeddings(user_embeds)  # Subtract 1 to match LLM indexing
        user_embeds = self.proj_user(user_embeds)
        return user_embeds
        # user_embeds = self.user_embeddings(user_embeds)
        # user_embeds = self.proj_user(user_embeds)
        # return user_embeds
class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem ):
        self.src_len = 4 #7, 
        emsize = self.transformer.wte.weight.size(1)  # 768
        # self.Context_embeddings = ContextEmbeddings(384,emsize)
        self.user_embs1 = SasrecUserEmbeddings(50, emsize)
        self.item_embs1 = SasrecItemEmbeddings(50, emsize)
        # Second embedding set (trainable embeddings)
        self.user_embs2 = nn.Embedding(nuser, emsize)
        self.item_embs2 = nn.Embedding(nitem, emsize)
        # self.rating_embeds = SasrecRatings(emsize)
        initrange = 0.1
        self.user_embs2.weight.data.uniform_(-initrange, initrange)
        self.item_embs2.weight.data.uniform_(-initrange, initrange)
        # Adding a projection layer to reduce concatenated embeddings to match emsize
        self.proj_combined = nn.Linear(emsize * 2, emsize)
        # self.metadata_proj = nn.Linear(384, emsize)
    def forward(self, user, item, text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)
        # self.rating_embeds.ratings = self.rating_embeds.ratings.to(device)
        # Get embeddings from both sets
        u_src1 = self.user_embs1(user)  # (batch_size, emsize) from pretrained user embeddings
        i_src1 = self.item_embs1(item)  # (batch_size, emsize) from pretrained item embeddings
        u_src2 = self.user_embs2(user)  # (batch_size, emsize) from trainable user embeddings
        i_src2 = self.item_embs2(item)  # (batch_size, emsize) from trainable item embeddings
        # Concatenate embeddings
        u_src = torch.cat([u_src1, u_src2], dim=-1)  # (batch_size, 2 * emsize)
        i_src = torch.cat([i_src1, i_src2], dim=-1)  # (batch_size, 2 * emsize)
        u_src = self.proj_combined(u_src)  # (batch_size, emsize)
        i_src = self.proj_combined(i_src)  # (batch_size, emsize)
        # rating_ids = self.rating_embeds.ratings[user.to(device), item.to(device)] 
        # r_src = self.rating_embeds(rating_ids)
        w_src = self.transformer.wte(text)
        # print(feat[:5])
        # feat_embeds = self.transformer.wte(feat.to(device))
        # context_src = self.transformer.wte(context.to(device).long())
        src = torch.cat([u_src1.unsqueeze(1), u_src2.unsqueeze(1), i_src1.unsqueeze(1), i_src2.unsqueeze(1),  w_src], 1) #unsqueeze(1) adding new dim, context_src, feat_embeds,
        # Compute logits and log probabilities
        outputs = super().forward(inputs_embeds=src)
        logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        return {
            "logits": logits,
            "log_probs": log_probs
        }
        # if mask is None:
        #     return super().forward(inputs_embeds=src)
        # else:
        #     # training
        #     # input padding
        #     pad_left = torch.ones((batch_size, self.src_len + feat_embeds.size(1)), dtype=torch.int64).to(device) # concept_embeds.size(1) + feat_embeds.size(1)+ context_src.size(1)
        #     pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)
        #     # prediction for training
        #     pred_left = torch.full((batch_size, self.src_len + feat_embeds.size(1)), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
        #     pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
        #     prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)
        #     return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
