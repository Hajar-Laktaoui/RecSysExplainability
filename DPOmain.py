import os
import math
import torch
import numpy as np
import argparse
from transformers import GPT2Tokenizer, AdamW
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import argparse
import random
import numpy as np
from functools import partial
# from VersionModule import ContinuousPromptLearning
# from module import ContinuousPromptLearning
from DPOLs import ContinuousPromptLearning
# from ModuleHistory import ContinuousPromptLearning
from DPOutils import rouge_score, bleu_score, DataLoader, DataLoader, Batchify, now_time, ids2tokens, ids2tokensReal, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity, seed_everything
# from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
#     feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity

os.environ["WANDB_API_KEY"] = "7c0382c84f747e007f12b15329a10ddad3de09d5"
parser = argparse.ArgumentParser(description='DPO for SFT model')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                     help='load indexes')
parser.add_argument('--lr', type=float, default=0.0001, #1e-6
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=64,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, #128
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times1', type=int, default=3,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--endure_times', type=int, default=3,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
parser.add_argument("--seed", type=int, default=2003)

parser.add_argument("--wandb_project", type=str, default="test-dpo")

args = parser.parse_args()
seed_everything(args.seed)

wandb.login()
wandb.init(project=args.wandb_project, config=args)

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)
###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
# tokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
# corpus = DataLoader2(args.data_path, tokenizer, tokenizerFast, args.words)
# corpusFeat = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
feature_set = corpus.feature_set
train_data = Batchify(corpus.train, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,args.words, args.batch_size, shuffle=True) # tokenizerFast,
val_data = Batchify(corpus.valid, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,args.words, args.batch_size)
test_data = Batchify(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,args.words, args.batch_size)
###############################################################################
# Build the model
###############################################################################
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer)
# Path to the saved model
saved_model_path = "/home/hajar.laktaoui/lustre/robust_ml-um6p-st-sccs-iqcbvkbobtq/users/hajar.laktaoui/Implementation4TkL/PEPLERSASRec/TripAdvisor/tripadvisorij/model.pt"
# Check if a saved model exists
if os.path.exists(saved_model_path):
    print(now_time() + "Loading the saved model from:", saved_model_path)
    model = torch.load(saved_model_path).to(device)
    ref_model = torch.load(saved_model_path).to(device)
    # You may need to reinitialize the optimizer after loading the model
    optimizer = AdamW(model.parameters(), lr=args.lr)
else:
    print(now_time() + "No saved model found. Initializing a new model.")
    model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
    model.resize_token_embeddings(ntoken)  # Update embedding table
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

# Proceed with the training as before
# model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)#gpt2-large
# ref_model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)#gpt2-large
# model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
# model.to(device)
# optimizer = AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################
def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_DPO_loss(model_prefered_logprob, model_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob,
                       beta=0.5):

    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean(dim=-1)
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean(dim=-1)

    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean(dim=-1)

    return loss, prefered_relative_logprob.mean(dim=-1), disprefered_relative_logprob.mean(dim=-1), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1).mean(-1)

# def train(model, ref_model, tokenizer, optimizer, data, beta=0.1):
#     model.train()
#     ref_model.eval()
#     while True:
#             batch = data.next_batch()
#             optimizer.zero_grad()

#             users = batch['user'].to(device)
#             items = batch['item'].to(device)
#             ratings = batch['rating'].to(device)
#             seq = batch['seq'].to(device)
#             mask = batch['mask'].to(device)
#             feat = batch['feat'].to(device)

#             prompt_prefered_ids = batch['prompt_prefered_ids'].to(device)
#             prompt_disprefered_ids = batch['prompt_disprefered_ids'].to(device)
#             prompt_prefered_mask = batch['prompt_prefered_mask'].to(device)
#             prompt_disprefered_mask = batch['prompt_disprefered_mask'].to(device)

#             # Forward pass with model
#             model_prefered_output = model(users, items, prompt_prefered_ids, feat,  mask=prompt_prefered_mask)
#             model_disprefered_output = model(users, items, prompt_disprefered_ids, feat, mask=prompt_disprefered_mask)

#             # Compute log probabilities
#             model_prefered_log_prob = get_log_prob(model_prefered_output["logits"], prompt_prefered_ids)
#             model_disprefered_log_prob = get_log_prob(model_disprefered_output["logits"], prompt_disprefered_ids)

#             # Reference model outputs
#             ref_prefered_output = ref_model(users, items, prompt_prefered_ids, feat, mask=prompt_prefered_mask)
#             ref_disprefered_output = ref_model(users, items, prompt_disprefered_ids, feat, mask=prompt_disprefered_mask)

#             ref_prefered_log_prob = get_log_prob(ref_prefered_output["logits"], prompt_prefered_ids)
#             ref_disprefered_log_prob = get_log_prob(ref_disprefered_output["logits"], prompt_disprefered_ids)

#             # Compute DPO loss
#             loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
#                 model_prefered_log_prob, model_disprefered_log_prob,
#                 ref_prefered_log_prob, ref_disprefered_log_prob,
#                 beta=beta)

#             # Backpropagation
#             loss.backward()
#             optimizer.step()

#             wandb.log({
#                 'loss': loss.item(),
#                 'prefered_relative_logprob': prefered_relative_logprob.mean().item(),
#                 'disprefered_relative_logprob': disprefered_relative_logprob.mean().item(),
#                 'reward_accuracy': reward_accuracies.mean().item(),
#                 'reward_margin': reward_margins.mean().item()
#             })
def train(model, ref_model, tokenizer, optimizer, data, beta=0.1):
    model.train()
    ref_model.eval()
    
    text_loss = 0.
    total_sample = 0

    while True:
        batch = data.next_batch()
        optimizer.zero_grad()

        users = batch['user'].to(device)
        items = batch['item'].to(device)
        ratings = batch['rating'].to(device)
        seq = batch['seq'].to(device)
        mask = batch['mask'].to(device)
        feat = batch['feat'].to(device)

        prompt_prefered_ids = batch['prompt_prefered_ids'].to(device)
        prompt_disprefered_ids = batch['prompt_disprefered_ids'].to(device)
        prompt_prefered_mask = batch['prompt_prefered_mask'].to(device)
        prompt_disprefered_mask = batch['prompt_disprefered_mask'].to(device)

        # Forward pass with model
        model_prefered_output = model(users, items, prompt_prefered_ids, feat,  mask=prompt_prefered_mask)
        model_disprefered_output = model(users, items, prompt_disprefered_ids, feat, mask=prompt_disprefered_mask)

        # Compute log probabilities
        model_prefered_log_prob = get_log_prob(model_prefered_output["logits"], prompt_prefered_ids)
        model_disprefered_log_prob = get_log_prob(model_disprefered_output["logits"], prompt_disprefered_ids)

        # Reference model outputs
        ref_prefered_output = ref_model(users, items, prompt_prefered_ids, feat, mask=prompt_prefered_mask)
        ref_disprefered_output = ref_model(users, items, prompt_disprefered_ids, feat, mask=prompt_disprefered_mask)

        ref_prefered_log_prob = get_log_prob(ref_prefered_output["logits"], prompt_prefered_ids)
        ref_disprefered_log_prob = get_log_prob(ref_disprefered_output["logits"], prompt_disprefered_ids)

        # Compute DPO loss
        loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
            model_prefered_log_prob, model_disprefered_log_prob,
            ref_prefered_log_prob, ref_disprefered_log_prob,
            beta=beta)

        # Backpropagation
        loss.backward()
        optimizer.step()

        batch_size = users.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        # Interval-based logging
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'DPO loss {:4.4f} | {:5d}/{:5d} batches'.format(cur_t_loss, data.step, data.total_step))
            text_loss = 0.
            total_sample = 0
        
        # Log additional metrics using wandb
        wandb.log({
            'loss': loss.item(),
            'prefered_relative_logprob': prefered_relative_logprob.mean().item(),
            'disprefered_relative_logprob': disprefered_relative_logprob.mean().item(),
            'reward_accuracy': reward_accuracies.mean().item(),
            'reward_margin': reward_margins.mean().item()
        })

        if data.step == data.total_step:
            break


def evaluate(data):
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            batch = data.next_batch()
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            seq = batch['seq'].to(device)
            mask = batch['mask'].to(device)
            feat = batch['feat'].to(device)

            outputs = model(users, items, seq, feat, mask)
            loss = outputs.loss

            batch_size = users.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break

    return text_loss / total_sample



def generate(data):
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            batch = data.next_batch()
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            seq = batch['seq'].to(device)
            feat = batch['feat'].to(device)
            text = seq[:, :1].to(device)  # Start with <bos>
            for idx in range(seq.size(1)):
                outputs = model(users, items, text, feat, None)
                last_token = outputs.logits[:, -1, :]
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)
                text = torch.cat([text, token], 1)

            ids = text[:, 1:].tolist()  # Remove <bos>
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict



print(now_time() + 'Tuning Prompt Only')
# wandb.init(project="DPO_training", entity="DPOTripAdvisor")
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(model, ref_model, tokenizer, optimizer, train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Stopping early due to no improvement.')
            break

# Load the best saved model.

with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


print(now_time() + 'Tuning both Prompt and LM')
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=args.lr)

# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(model, ref_model, tokenizer, optimizer, train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Stopping early due to no improvement.')
            break

#model_path = '/content/drive/MyDrive/SFTmodel/tripadvisor/model.pt'
# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


# Run on test data.
test_loss = evaluate(test_data)
# test_loss = evaluate(val_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted = generate(test_data) #test_data

tokens_test = [ids2tokensReal(ids[1:], tokenizer, eos_token="<eos>", pad_token="<pad>") for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
# print("Feature Batch:", feature_batch[:5])  # Print first 5 batches of detected features
# print("Test Features:", test_data.feature[:5].tolist())  # Ensure this is a list for easy readability
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''

for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
