from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch
import copy


class MFItemEmbeddings(nn.Module):
    def __init__(self, MF_hidden_untis):
        super().__init__()

        path_to_trained_MFItemEmbeddings = '/home/hajar.laktaoui/ImplementationFolder/PEPLERMF/item_embeddings.pt'
        item_weights = torch.load(path_to_trained_MFItemEmbeddings, map_location='cpu')
        self.item_embeddings = nn.Embedding.from_pretrained(item_weights)
        
        
        # Freeze item embeddings weights
        for p in self.item_embeddings.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(self.item_embeddings.embedding_dim, MF_hidden_untis)

    def forward(self, item_embeds):
        item_embeds = self.item_embeddings(item_embeds)
        item_embeds = self.proj(item_embeds)
        return item_embeds


        # initrange = 0.1
        # self.item_embeddings.weight.data.uniform_(-initrange, initrange)
    #     self.proj = nn.Linear(self.item_embeddings.shape(0), nitem)
    #     # self.user_embeddings = torch.load(path)

    # def forward(self, x):
    #     x = self.item_embeddings(x)
    #     x = self.proj(x)
    #     return x


class MFUserEmbeddings(nn.Module):
    def __init__(self, MF_hidden_untis):
        super().__init__()

        path_to_trained_MFUserEmbeddings = '/home/hajar.laktaoui/ImplementationFolder/PEPLERMF/user_embeddings.pt'
        user_weights = torch.load(path_to_trained_MFUserEmbeddings, map_location='cpu')
        self.user_embeddings = nn.Embedding.from_pretrained(user_weights)
        
        
        # Freeze item embeddings weights
        for p in self.user_embeddings.parameters():
            p.requires_grad = False
        
        self.proj = nn.Linear(self.user_embeddings.embedding_dim, MF_hidden_untis)

    def forward(self, user_embeds):
        user_embeds = self.user_embeddings(user_embeds)
        user_embeds = self.proj(user_embeds)
        return user_embeds

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

    def init_prompt(self, nuser, nitem):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1)  # 768

        self.user_embeddings = MFUserEmbeddings(emsize)

        # self.user_embeddings = nn.Embedding(nuser, emsize)
        # self.user_embeddings = nn.Embedding(nuser, emsize)
        # user_weights = torch.load(path)
        # self.user_embeddings.weight = nn.Parameter(user_weights, requires_grad=False)
        # self.item_embeddings = nn.Embedding(nitem, emsize)
        self.item_embeddings = MFItemEmbeddings(emsize)

        # initrange = 0.1
        # self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        # # self.item_embeddings.weight.data.uniform_(-initrange, initrange)
        # item_embeddings_data = torch.tensor(self.load_embeddings(file_path, emsize))
        # self.item_embeddings.weight.data.copy_(item_embeddings_data)
    
    def forward(self, user, item, text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)
        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


