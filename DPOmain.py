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
from DPOLs import ContinuousPromptLearning
from DPOutils import rouge_score, bleu_score, DataLoader, DataLoader, Batchify, now_time, ids2tokens, ids2tokensReal, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity, seed_everything

os.environ["WANDB_API_KEY"] = "7c0382c84f747e007f12b15329a10ddad3de09d5"
parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                     help='load indexes')
parser.add_argument('--lr', type=float, default=1e-5, #1e-6, 0.0001
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
parser.add_argument('--endure_times1', type=int, default=1,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--endure_times', type=int, default=1,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
parser.add_argument("--seed", type=int, default=2003)

parser.add_argument("--wandb_project", type=str, default="testlv-dpo")

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
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
corpus = DataLoader(args.data_path, args.index_dir, tokenizer, args.words)
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
saved_model_path = "/home/hajar.laktaoui/lustre/robust_ml-um6p-st-sccs-iqcbvkbobtq/users/hajar.laktaoui/Implementation4TkL/PEPLERSASRec/DPOtripAdvisorGPT/model.pt"

# Check if a saved model exists and load it
if os.path.exists(saved_model_path):
    print(now_time() + " Resuming training from:", saved_model_path)
    model = torch.load(saved_model_path).to(device)  # Load only model
else:
    print(now_time() + " No saved model found. Initializing a new model.")
    model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
    model.resize_token_embeddings(ntoken)
    model.to(device)

# Keep the reference model unchanged
ref_model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
ref_model.resize_token_embeddings(ntoken)
ref_model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

'''
# Proceed with the training as before
model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)#gpt2-large
ref_model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)#gpt2-large
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
ref_model.resize_token_embeddings(ntoken) 
ref_model.to(device)
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
'''

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
                       beta=0.1):
"""
    Compute Direct Preference Optimization (DPO) loss.
    
    Args:
        model_prefered_logprob (torch.FloatTensor): Log probabilities of preferred samples from the model.
        model_disprefered_logprob (torch.FloatTensor): Log probabilities of non-preferred samples from the model.
        ref_prefered_logprob (torch.FloatTensor): Log probabilities of preferred samples from the reference model.
        ref_disprefered_logprob (torch.FloatTensor): Log probabilities of non-preferred samples from the reference model.
        beta (float, optional): Scaling factor for the loss. Default is 0.1.

    Returns:
        Tuple containing loss, mean preferred log probability, mean non-preferred log probability,
        reward accuracy, and reward margin.
"""
    # (1) Compute relative log probabilities
    prefered_relative_logprob = model_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = model_disprefered_logprob - ref_disprefered_logprob
    
    # (2) Compute loss (KEY FIX: Use .mean() not .mean(dim=-1))
    loss = -F.logsigmoid(beta * (prefered_relative_logprob - disprefered_relative_logprob)).mean()
    
    # (3) Compute metrics
    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float().mean()
    reward_margins = (prefered_relative_logprob - disprefered_relative_logprob).mean()
    
    return loss, prefered_relative_logprob.mean(), disprefered_relative_logprob.mean(), reward_accuracies, reward_margins

def get_log_prob(logits, labels):
"""
    Compute the average log probability of each sample's token sequence, masking out padding.

    Specifically, this function:
    1. **Aligns** the `logits` and `labels` by removing the last token from `logits` and
       the first token from `labels`, conforming to the next-token-prediction (autoregressive) setup.
    2. **Applies a mask** to ignore positions where `labels == -100` (commonly used for padding or ignored tokens).
    3. **Computes log probabilities** across the vocabulary, then gathers the log probabilities of the
       correct tokens (from `labels`).
    4. **Averages** these token-level log probabilities over the valid (non-masked) tokens for each sample.

    Args:
        logits (torch.Tensor):
            A tensor of shape `(batch_size, seq_length, vocab_size)` representing the model's
            unnormalized output scores (logits).
        labels (torch.Tensor):
            A tensor of shape `(batch_size, seq_length)` with token IDs. Positions where `labels == -100`
            are considered padding and should be ignored in loss computation.

    Returns:
        torch.Tensor:
            A 1D tensor of shape `(batch_size,)` containing the average log probability
            for each sample in the batch, considering only valid (non-masked) tokens.
"""
    # (1) Align logits and labels
    logits = logits[:, :-1, :]  # Remove last token
    labels = labels[:, 1:].clone()  # Remove first token
    # (2) Mask padding
    mask = (labels != -100)
    labels[labels == -100] = 0  # Avoid index error
    # (3) Compute log probs
    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    # (4) Average over VALID tokens only
    gathered_masked = gathered * mask.float()
    return gathered_masked.sum(-1) / (mask.sum(-1) + 1e-9)  # Avg per example
    
def train(model, ref_model, tokenizer, optimizer, data, beta=0.5):
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

        model_prefered_log_prob = get_log_prob(
            model(users, items, prompt_prefered_ids, feat, prompt_prefered_mask).logits,
            prompt_prefered_ids)

        # Similarly for other log prob calls:
        model_disprefered_log_prob = get_log_prob(
            model(users, items, prompt_disprefered_ids, feat, prompt_disprefered_mask).logits,
            prompt_disprefered_ids)

        ref_prefered_log_prob = get_log_prob(
            ref_model(users, items, prompt_prefered_ids, feat, prompt_prefered_mask).logits,
            prompt_prefered_ids)

        ref_disprefered_log_prob = get_log_prob(
            ref_model(users, items, prompt_disprefered_ids, feat, prompt_disprefered_mask).logits,
            prompt_disprefered_ids)
        # Compute DPO loss
        loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins = calculate_DPO_loss(
            model_prefered_log_prob, model_disprefered_log_prob,
            ref_prefered_log_prob, ref_disprefered_log_prob,
            beta=beta)

        # Backpropagation
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        wandb.log({
            'loss': loss.item(),
            'prefered_relative_logprob': prefered_relative_logprob.mean().item(),
            'disprefered_relative_logprob': disprefered_relative_logprob.mean().item(),
            'reward_accuracy': reward_accuracies.mean().item(),
            'reward_margin': reward_margins.mean().item()
        })


        batch_size = users.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        # Interval-based logging
        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'DPO loss {:4.4f} | {:5d}/{:5d} batches'.format(cur_t_loss, data.step, data.total_step))
            text_loss = 0.
            total_sample = 0

        if data.step == data.total_step:
            break
def evaluate(data):
    model.eval()
    ref_model.eval()
    total_loss = 0.
    total_acc = 0.
    total_margin = 0.
    total_sample = 0

    with torch.no_grad():
        while True:
            batch = data.next_batch()
            users = batch['user'].to(device)
            items = batch['item'].to(device)
            feat = batch['feat'].to(device)

            # Extract paired examples
            prompt_prefered_ids = batch['prompt_prefered_ids'].to(device)
            prompt_disprefered_ids = batch['prompt_disprefered_ids'].to(device)
            prompt_prefered_mask = batch['prompt_prefered_mask'].to(device)
            prompt_disprefered_mask = batch['prompt_disprefered_mask'].to(device)
            optimizer.zero_grad()

            # Forward pass for model and reference model
            model_prefered_log_prob = get_log_prob(
                model(users, items, prompt_prefered_ids, feat, prompt_prefered_mask).logits,
                prompt_prefered_ids)

            model_disprefered_log_prob = get_log_prob(
                model(users, items, prompt_disprefered_ids, feat, prompt_disprefered_mask).logits,
                prompt_disprefered_ids)

            ref_prefered_log_prob = get_log_prob(
                ref_model(users, items, prompt_prefered_ids, feat, prompt_prefered_mask).logits,
                prompt_prefered_ids)

            ref_disprefered_log_prob = get_log_prob(
                ref_model(users, items, prompt_disprefered_ids, feat, prompt_disprefered_mask).logits,
                prompt_disprefered_ids)

            # Calculate DPO metrics
            loss, pref_logp, dispref_logp, acc, margin = calculate_DPO_loss(
                model_prefered_log_prob, model_disprefered_log_prob,
                ref_prefered_log_prob, ref_disprefered_log_prob,
                beta=0.1
            )

            batch_size = users.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.mean().item() if acc.numel() > 0 else 0
            total_margin += margin.mean().item() if margin.numel() > 0 else 0
            total_sample += batch_size

            # Ensure we handle `step` properly
            if getattr(data, "step", 0) == getattr(data, "total_step", 1):  # Default `total_step=1`
                break

    # Avoid division by zero
    if total_sample == 0:
        return 0, 0, 0

    avg_loss = total_loss / total_sample
    avg_acc = total_acc / total_sample
    avg_margin = total_margin / total_sample
    return avg_loss, avg_acc, avg_margin


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
            
            # Generate up to args.words tokens
            for _ in range(args.words):
                outputs = model(users, items, text, feat, None)
                last_token = outputs['logits'][:, -1, :]  # (batch_size, vocab_size)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1)
                text = torch.cat([text, token], dim=1)

            # Convert to list of token IDs
            ids = text[:, 1:].tolist()  # Remove <bos>
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict

print(now_time() + 'Tuning Prompt Only')
endure_count = 0
best_val_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(model, ref_model, tokenizer, optimizer, train_data)
    
    # Evaluate on validation
    val_loss, val_acc, val_margin = evaluate(val_data)
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Margin: {val_margin:.4f}')
    
    # Save best model based on accuracy
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
        # print('Model saved')
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
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False
optimizer = AdamW(model.parameters(), lr=args.lr)

endure_count = 0
best_val_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(model, ref_model, tokenizer, optimizer, train_data)
    
    # Evaluate on validation
    val_loss, val_acc, val_margin = evaluate(val_data)
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Margin: {val_margin:.4f}')
    
    # Save best model based on accuracy
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
        # print('Model saved')
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Stopping early due to no improvement.')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device) #map_location='cpu'

# Run on test data.
test_loss = evaluate(test_data) 
# test_loss = evaluate(val_data)
print('=' * 89)
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
