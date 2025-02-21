from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch
import copy
from sasrec_model import SASRec
import numpy as np
from torch.nn.functional import log_softmax

"""
Descriptions for Each Class:
1. SasrecItemEmbeddings
   - Initializes item embeddings from a pretrained SASRec model, freezes them,
     and maps from SASRec's hidden dimension to the GPT-2 embedding dimension.
2. SasrecUserEmbeddings
   - Initializes user embeddings from a pretrained SASRec model, freezes them,
     and maps from SASRec's hidden dimension to the GPT-2 embedding dimension.
3. UIPrompt
   - A parent class that loads a pretrained GPT-2 model and adds custom prompts
     for user and item embeddings. Also includes a method to freeze/unfreeze
     parts of the language model.
4. ContinuousPromptLearning
   - Extends GPT2LMHeadModel and UIPrompt to create a trainable system that
     combines SASRec embeddings with GPT-2 embeddings for user and item inputs.
"""

class SasrecItemEmbeddings(nn.Module):
    """
    Initializes item embeddings from a pretrained SASRec model and projects them
    to match the GPT-2 embedding size. Freezes the pretrained weights.
    """
    def __init__(self, sasrec_hidden_untis, emsize):
        super().__init__()
        path_to_trained_sasrec = '/home/hajar.laktaoui/Implementation/Implementation4TkL/SASRec.pytorchTest/TripAdvisor_default/SASRec.epoch=40.lr=0.001.layer=2.head=1.hidden=100.maxlen=200.pth'
        sasrec_weights = torch.load(path_to_trained_sasrec, map_location='cpu')

        # Load pretrained item embeddings
        self.item_embeddingsAll = nn.Embedding.from_pretrained(sasrec_weights['item_emb.weight'])
        self.item_embeddings = nn.Embedding.from_pretrained(self.item_embeddingsAll.weight[1:])

        # Freeze item embedding weights
        for p in self.item_embeddings.parameters():
            p.requires_grad = False

        # Linear projection to adapt from SASRec dimension to GPT-2 dimension
        self.proj_item = nn.Linear(sasrec_hidden_untis, emsize)

    def forward(self, item_embeds):
        # Retrieve frozen item embeddings and project them
        item_embeds = self.item_embeddings(item_embeds)
        item_embeds = self.proj_item(item_embeds)
        return item_embeds

class SasrecUserEmbeddings(nn.Module):
    """
    Initializes user embeddings from a pretrained SASRec model and projects them
    to match the GPT-2 embedding size. Freezes the pretrained weights.
    """
    def __init__(self, sasrec_hidden_untis, emsize):
        super().__init__()
        path_to_trained_sasrec = '/home/hajar.laktaoui/Implementation/Implementation4TkL/PEPLERSASRec/UserEmbRatingsTripAdvisor.pt'
        loaded_data = torch.load(path_to_trained_sasrec, map_location='cpu')
        user_weights = loaded_data['user_embeddings']

        # Load pretrained user embeddings
        self.user_embeddings = nn.Embedding.from_pretrained(user_weights)
        for p in self.user_embeddings.parameters():
            p.requires_grad = False

        # Linear projection to adapt from SASRec dimension to GPT-2 dimension
        self.proj_user = nn.Linear(sasrec_hidden_untis, emsize)

    def forward(self, user_embeds):
        # Ensure embeddings layer is on the correct device
        device = user_embeds.device
        self.user_embeddings = self.user_embeddings.to(device)

        # Retrieve frozen user embeddings and project them
        user_embeds = self.user_embeddings(user_embeds)
        user_embeds = self.proj_user(user_embeds)
        return user_embeds

class UIPrompt:
    """
    A parent class that provides a mechanism to freeze the GPT-2 model parameters
    and inject user/item embeddings as continuous prompts at the start of the input sequence.
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem):
        # Number of special prompt tokens
        self.src_len = 4
        emsize = self.transformer.wte.weight.size(1)

        # Pretrained SASRec user/item embeddings
        self.user_embs1 = SasrecUserEmbeddings(100, emsize)
        self.item_embs1 = SasrecItemEmbeddings(100, emsize)

        # Trainable user/item embeddings
        self.user_embs2 = nn.Embedding(nuser, emsize)
        self.item_embs2 = nn.Embedding(nitem, emsize)
        initrange = 0.1
        self.user_embs2.weight.data.uniform_(-initrange, initrange)
        self.item_embs2.weight.data.uniform_(-initrange, initrange)

        # Projection to combine pretrained + trainable embeddings
        self.proj_combined = nn.Linear(emsize * 2, emsize)

    def forward(self, user, item, text, feat, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # Extract user and item embeddings from both pretrained and trainable layers
        u_src1 = self.user_embs1(user)
        i_src1 = self.item_embs1(item)
        u_src2 = self.user_embs2(user)
        i_src2 = self.item_embs2(item)

        # Combine embeddings and project them
        u_src = self.proj_combined(torch.cat([u_src1, u_src2], dim=-1))
        i_src = self.proj_combined(torch.cat([i_src1, i_src2], dim=-1))

        # GPT-2 embeddings for text
        w_src = self.transformer.wte(text)

        # Concatenate prompt embeddings with text embeddings
        src = torch.cat([u_src1.unsqueeze(1), 
                         u_src2.unsqueeze(1), 
                         i_src1.unsqueeze(1), 
                         i_src2.unsqueeze(1), 
                         w_src], dim=1)

        # If mask is provided, handle training scenario
        if mask is None:
            return super().forward(inputs_embeds=src)
        else:
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], dim=1)
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index, device=device))
            prediction = torch.cat([pred_left, pred_right], dim=1)
            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)

class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
